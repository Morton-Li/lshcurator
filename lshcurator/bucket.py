import threading
from abc import ABC, abstractmethod
from multiprocessing import shared_memory
from pathlib import Path
from time import sleep
from typing import Literal

import numpy

from .algorithms import compute_minhash_signature, compute_band_keys
from .config import BucketConfig
from .utils.readers import iter_parquet_batches, iter_jsonl_rows
from .utils.types import BucketShardMemorySpec, ShmBucketReport, ShmBucketQueueGroups, ShmBucketCommand, ComputeMode


class BucketBase(ABC):
    """Abstract base class for LSH buckets."""
    def __init__(self, config: BucketConfig):
        if config.compute_mode not in ComputeMode:
            raise ValueError(f"Invalid compute_mode: {config.compute_mode}. Must be one of {ComputeMode}")

        self.cfg = config

        self._keys: numpy.ndarray
        self._keys_written: int = 0  # 记录 keys 已写入长度指针

    @abstractmethod
    def append_keys(self, keys: numpy.ndarray) -> None: ...
    @abstractmethod
    def clear(self) -> None: ...
    @property
    def keys(self) -> numpy.ndarray:
        """Return the keys currently stored in the bucket as a numpy array."""
        if not hasattr(self, '_keys'): raise ValueError("Keys have not been initialized yet")
        return self._keys[:self._keys_written]  # 只返回已写入部分
    @property
    def keys_written(self) -> int:
        """Return the number of keys currently written in the bucket."""
        return self._keys_written

    def insert(self, text: str) -> None:
        """Insert a text into the LSH buckets based on its MinHash signature."""
        hash_values: numpy.ndarray = compute_minhash_signature(
            text=text,
            num_perm=self.cfg.bands * self.cfg.rows_per_band,
            shingle_k=self.cfg.shingle_k,
            shingle_step=self.cfg.shingle_step,
            compute_mode=self.cfg.compute_mode,
        ).hashvalues.astype(self.cfg.dtype, copy=False)

        band_keys: numpy.ndarray = compute_band_keys(
            hash_values=hash_values,
            bands=self.cfg.bands,
            rows_per_band=self.cfg.rows_per_band,
        )
        self.append_keys(band_keys)


class Bucket(BucketBase):
    def __init__(self, config: BucketConfig):
        super().__init__(config=config)
        self._keys: numpy.ndarray = numpy.empty(1_000_000, dtype=numpy.uint64)  # 使用 np 替代 list 降低开销

    def append_keys(self, keys: numpy.ndarray) -> None:
        """Append keys to the internal storage, resizing if necessary."""
        if not isinstance(keys, numpy.ndarray): raise ValueError("Keys must be a numpy array")
        new_len = self._keys_written + len(keys)
        if new_len > self._keys.size:
            # Resize by doubling the size
            new_size = max(new_len, self._keys.size * 2)
            new_array = numpy.empty(new_size, dtype=numpy.uint64)
            new_array[:self._keys_written] = self._keys[:self._keys_written]
            self._keys = new_array
        self._keys[self._keys_written:new_len] = keys
        self._keys_written = new_len

    def batch_insert(self, texts: list[str]) -> None:
        """Insert a batch of texts into the LSH buckets."""
        for text in texts: self.insert(text)

    def extract_keys(self, min_hit_count: int | None = None) -> numpy.ndarray:
        """Extract the bucket keys as a numpy array."""
        data = self._keys[:self._keys_written].copy()  # 只考虑已写入部分

        keys, counts = numpy.unique(data, return_counts=True)  # unique 会返回有序结果无需 sort
        if min_hit_count is not None:
            keys = keys[counts >= min_hit_count]
        return keys

    def clear(self) -> None:
        """Clear all keys from the bucket."""
        self._keys = numpy.empty(1_000_000, dtype=numpy.uint64)  # 重置为初始大小防止已创建的数组过大白白占用内存
        self._keys_written = 0


class ShmBucket(BucketBase):
    def __init__(
        self,
        config: BucketConfig,
        shm_spec: BucketShardMemorySpec,
        queue_group: ShmBucketQueueGroups,
        bucket_id: int = 0,
    ):
        super().__init__(config=config)
        self.bucket_id = bucket_id
        self.bucket_status = 'ready'

        self._shm_spec = shm_spec
        self._shm = shared_memory.SharedMemory(name=shm_spec.name, create=False)  # 由上游负责创建和管理共享内存

        if self.cfg.dtype != shm_spec.dtype:
            raise ValueError(f"Bucket dtype {self.cfg.dtype} does not match shared memory dtype {shm_spec.dtype}")
        self._keys: numpy.ndarray = numpy.ndarray((shm_spec.n_elements,), dtype=shm_spec.dtype, buffer=self._shm.buf)  # 使用共享内存作为 keys 存储

        self.queue_group = queue_group

        # 启动一个线程监听上游命令队列，处理扩容和清理等命令
        self._command_listener_thread = threading.Thread(target=self._listen_for_commands, daemon=True)
        self._command_listener_thread.start()

    def _listen_for_commands(self) -> None:
        """监听上游命令队列"""
        while True:
            command: ShmBucketCommand = self.queue_group.command_queue.get()  # 阻塞等待命令
            if command.action == '<|Exit|>':
                self.bucket_status = 'Exit'
                self._shm.close()
                break  # 退出线程
            else:
                fn_name = command.action
                if hasattr(self, fn_name) and callable(getattr(self, fn_name)):
                    fn = getattr(self, fn_name)
                    kwargs = command.kwargs or {}
                    fn(**kwargs)  # 调用对应方法处理命令
                else:
                    print(f"Bucket {self.bucket_id} received unknown command: {command.action}")

    def expand_shm(self) -> None:
        """请求上游扩容共享内存区域以适应更多的 keys"""
        self.bucket_status = 'requesting_expansion'
        self.queue_group.report_queue.put(ShmBucketReport(
            bucket_id=self.bucket_id,
            ShmSpec=self._shm_spec,
            written=self.keys_written,
            status='processing',
            action='expand',
            message=f"Bucket {self.bucket_id} requesting expansion to accommodate more keys",
        ))
        # 等待上游返回 new_shm_spec 并刷新共享内存连接
        while self.bucket_status != 'ready' and self.bucket_status != 'Exit': sleep(1)

    def refresh_shm(self, new_shm_spec: BucketShardMemorySpec) -> None:
        """刷新共享内存连接以适应新的共享内存区域，通常在上游扩容后调用"""
        self.bucket_status = 'expanding'  # 扩容中
        self._shm.close()  # 关闭当前共享内存连接
        self._shm_spec = new_shm_spec
        self._shm = shared_memory.SharedMemory(name=new_shm_spec.name, create=False)  # 连接到新的共享内存
        if self.cfg.dtype != new_shm_spec.dtype:
            raise ValueError(f"Bucket dtype {self.cfg.dtype} does not match new shared memory dtype {new_shm_spec.dtype}")
        self._keys = numpy.ndarray((new_shm_spec.n_elements,), dtype=new_shm_spec.dtype, buffer=self._shm.buf)  # 更新 keys 数组指向新的共享内存
        self._keys_written = 0  # 重置写入指针
        self.bucket_status = 'ready'

    def append_keys(self, keys: numpy.ndarray) -> None:
        while self.bucket_status != 'ready' and self.bucket_status != 'Exit': sleep(1)  # 等待状态变为 ready
        if self.bucket_status == 'Exit': return  # 如果在等待过程中收到退出命令，直接返回不执行写入
        if not isinstance(keys, numpy.ndarray): raise ValueError("Keys must be a numpy array")
        new_len = self.keys_written + len(keys)
        if new_len > self._keys.size:
            # 计算最大允许写入量并部分写入
            deficit = new_len - self._keys.size
            if deficit < len(keys):
                self._keys[self.keys_written:self._keys.size] = keys[:len(keys) - deficit]
                self._keys_written = self._keys.size
                self.expand_shm()  # 请求上游扩容共享内存
                keys = keys[len(keys) - deficit:]  # 更新剩余未写入的 keys
                self.append_keys(keys)
            else:
                # 如果剩余 keys 一个都写不下，直接写入当前能写的部分并通知上游扩容
                self.expand_shm()
                self.append_keys(keys)
        else:
            self._keys[self._keys_written:new_len] = keys
            self._keys_written = new_len

    def run_job(
        self,
        file_path: Path,
        field_name: str | list[str],
        file_format: Literal['parquet', 'jsonl'] = 'parquet',
        **kwargs
    ) -> None:
        """批量插入文本并在完成后向上游报告状态"""
        if isinstance(field_name, str): field_name = [field_name]

        if file_format == 'parquet':
            for batch in iter_parquet_batches(
                parquet_path=file_path,
                batch_size=kwargs.get('batch_size', 2048),
                text_field=field_name,
            ):
                for sample in batch.stack().replace(r'^\s*$', numpy.nan, regex=True).dropna().reset_index(drop=True):
                    while self.bucket_status != 'ready' and self.bucket_status != 'Exit': sleep(1)
                    if self.bucket_status == 'Exit': return None  # 如果在等待过程中收到退出命令，直接返回不执行写入
                    self.insert(text=sample)
        elif file_format == 'jsonl':
            for row in iter_jsonl_rows(file_path=file_path):
                for field in field_name:
                    while self.bucket_status != 'ready' and self.bucket_status != 'Exit': sleep(1)
                    if self.bucket_status == 'Exit': return None  # 如果在等待过程中收到退出命令，直接返回不执行写入
                    content = row.get(field, None)
                    if isinstance(content, str): self.insert(text=content)
        else: raise ValueError(f"Unsupported file format: {file_format}")

        # 向上游报告完成状态
        self.queue_group.report_queue.put(ShmBucketReport(
            bucket_id=self.bucket_id,
            ShmSpec=self._shm_spec,
            written=self.keys_written,
            status='complete',
            message=f"Bucket {self.bucket_id} completed job with {self.keys_written} keys",
        ))
        # 防止在报告完成后继续接受命令导致数据混用
        self.clear()
        return None

    def clear(self) -> None: self._keys_written = 0  # 只需重置写入指针，实际数据会被覆盖无需清零


def bucket_worker(
    config: BucketConfig,
    shm_spec: BucketShardMemorySpec,
    queue_group: ShmBucketQueueGroups,
    job_kwargs: dict,
    bucket_id: int = 0,
) -> None:
    """Bucket 进程的主函数，负责创建 ShmBucket 实例并运行任务"""
    bucket = ShmBucket(config=config, shm_spec=shm_spec, queue_group=queue_group, bucket_id=bucket_id)
    bucket.run_job(**job_kwargs)


__all__ = ['Bucket', 'ShmBucket', 'bucket_worker']
