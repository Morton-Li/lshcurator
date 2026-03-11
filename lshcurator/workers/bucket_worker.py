# Copyright 2026 Morton Li. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import threading
from multiprocessing import shared_memory
from pathlib import Path
from queue import Empty, Queue
from time import sleep
from typing import Literal, Callable

import numpy

from .base import WorkerBase, WorkerManagerBase, _run_worker
from ..bucket import BucketBase
from ..config import BucketConfig, BucketWorkerManagerConfig, BucketWorkerConfig
from ..utils.readers import iter_parquet_batches, iter_jsonl_rows
from ..utils.types import ShardMemorySpec, BucketWorkerReport, BucketWorkerCommand, BucketWorkerSlot


class BucketWorker(WorkerBase, BucketBase):
    _worker_config: BucketWorkerConfig

    def __init__(
        self,
        worker_config: BucketWorkerConfig,
        bucket_config: BucketConfig,
    ):
        super().__init__(worker_config=worker_config, bucket_config=bucket_config)
        self.worker_status = 'ready'

        self._shm_spec = self._worker_config.shm_spec
        self._shm = shared_memory.SharedMemory(name=self._shm_spec.name, create=False)  # 由上游负责创建和管理共享内存
        self._keys: numpy.ndarray = numpy.ndarray((self._shm_spec.n_elements,), dtype=numpy.uint64, buffer=self._shm.buf)  # 使用共享内存作为 keys 存储

        # 启动一个线程监听上游命令队列，处理扩容和清理等命令
        self.command_queue: Queue[BucketWorkerCommand] = self._worker_config.command_queue
        self._command_listener_thread = threading.Thread(target=self._listen_for_commands, daemon=True)
        self._command_listener_thread.start()

    def set_worker_status(self, status: str) -> None: self.worker_status = status

    @property
    def paused(self) -> bool: return self.worker_status != 'ready' and not self.stop_event.is_set()

    def _listen_for_commands(self):
        """监听上游命令队列"""
        while not self.stop_event.is_set():
            try:
                command: BucketWorkerCommand = self.command_queue.get(timeout=1)  # 阻塞等待命令
            except Empty: continue  # 没有命令，继续等待

            fn_name = command.action
            if hasattr(self, fn_name) and callable(getattr(self, fn_name)):
                fn = getattr(self, fn_name)
                kwargs = command.kwargs or {}
                fn(**kwargs)  # 调用对应方法处理命令
            else: print(f"Bucket Worker {self.worker_id} received unknown command: {command.action}")

    def _report_merge_request(self) -> None:
        """向上游报告当前 bucket 的数据已准备好合并到全局 bucket keys 数组中"""
        self.worker_status = 'reporting_merge'
        self.report_queue.put(BucketWorkerReport(
            worker_id=self.worker_id,
            ShmSpec=self._shm_spec,
            written=self.keys_written,
            status='running',
            action='merge',
            message=f"Bucket Worker {self.worker_id} reporting ready for merge with {self.keys_written} keys",
        ))
        while self.paused: sleep(1)  # 阻塞等待上游处理完成后状态变为 ready 或接到 Exit
        self._keys_written = 0

    def append_keys(self, keys: numpy.ndarray[numpy.uint64]) -> None:
        while self.paused: sleep(1)  # 等待状态变为 ready
        if self.stop_event.is_set(): return  # 收到停止事件，退出方法

        if not isinstance(keys, numpy.ndarray): raise ValueError("Keys must be a numpy array")
        new_len = self.keys_written + len(keys)
        if new_len > self._keys.size:
            # 计算最大允许写入量并部分写入
            deficit = new_len - self._keys.size
            if deficit < len(keys):
                self._keys[self.keys_written:self._keys.size] = keys[:len(keys) - deficit]
                self._keys_written = self._keys.size
                self._report_merge_request()  # 向上游报告当前数据已准备好合并到全局 bucket keys 数组中
                keys = keys[len(keys) - deficit:]  # 更新剩余未写入的 keys
                self.append_keys(keys)
            else:
                # 如果剩余 keys 一个都写不下，直接通知上游合并当前数据
                self._report_merge_request()
                self.append_keys(keys)
        else:
            self._keys[self._keys_written:new_len] = keys
            self._keys_written = new_len

    def complete(self):
        """有序完成当前任务，向上游报告状态并发送退出命令"""
        if not self.stop_event.is_set():
            self.report_queue.put(BucketWorkerReport(
                worker_id=self.worker_id,
                ShmSpec=self._shm_spec,
                written=self.keys_written,
                status='complete',
                message=f"Bucket Worker {self.worker_id} completed job with {self.keys_written} keys",
            ))
            self._command_listener_thread.join()  # 等待命令监听线程退出
        self._shm.close()

    def job(
        self,
        file_path: Path,
        field_name: str | list[str],
        file_format: Literal['parquet', 'jsonl'] = 'parquet',
        **kwargs
    ):
        """
        批量插入文本并在完成后向上游报告状态
        Args:
            file_path: 输入文件路径，支持单个文件路径
            field_name: 需要处理的文本字段名称，支持单个字段或字段列表
            file_format: 输入文件格式，支持 'parquet' 和 'jsonl'
            kwargs:
                batch_size: 仅对 parquet 有效，指定批处理大小以控制内存使用，默认为 2048
        """
        if isinstance(field_name, str): field_name = [field_name]

        if file_format == 'parquet':
            for batch in iter_parquet_batches(
                parquet_path=file_path,
                batch_size=kwargs.get('batch_size', 2048),
                text_field=field_name,
            ):
                for sample in batch.stack().replace(r'^\s*$', numpy.nan, regex=True).dropna().reset_index(drop=True):
                    while self.paused: sleep(1)
                    if self.stop_event.is_set(): return self.complete()  # 收到停止事件，退出方法
                    self.insert(text=str(sample))
        elif file_format == 'jsonl':
            for row in iter_jsonl_rows(file_path=file_path):
                for field in field_name:
                    while self.paused: sleep(1)
                    if self.stop_event.is_set(): return self.complete()  # 收到停止事件，退出方法
                    content = row.get(field, '').strip()
                    if not content: continue
                    self.insert(text=str(content))
        else: raise ValueError(f"Unsupported file format: {file_format}")

        # 有序退出
        return self.complete()


class BucketWorkerManager(WorkerManagerBase):
    # 显式声明以满足类型检查器
    _worker_slots: dict[int, BucketWorkerSlot]
    worker_slots: dict[int, BucketWorkerSlot]
    _worker_manager_config: BucketWorkerManagerConfig

    def __init__(self, bucket_config: BucketConfig, bucket_worker_manager_config: BucketWorkerManagerConfig):
        super().__init__(worker_manager_config=bucket_worker_manager_config)

        self.bucket_config: BucketConfig = bucket_config  # 全局 bucket 配置
        self.bucket_keys: list[numpy.ndarray] = []  # 全局 bucket keys 列表

    def work_report_handler(self, report: BucketWorkerReport) -> None:
        if report.status == 'running':
            report_action = report.action
            if report_action == 'merge':
                self._merge_bucket_keys(worker_id=report.worker_id, n_written=report.written)
                command_queue = self.worker_slots[report.worker_id].command_queue
                command_queue.put(BucketWorkerCommand(
                    action='set_worker_status', kwargs={'status': 'ready'}
                ))
            else: print(f'Bucket Worker {report.worker_id} reported unknown action: {report_action}, message: {report.message}')
        elif report.status == 'complete': self._merge_bucket_keys(worker_id=report.worker_id, n_written=report.written)
        elif report.status == 'error': print(f'Bucket Worker {report.worker_id} reported error: {report.message}')
        else: raise RuntimeError(f'Bucket Worker {report.worker_id} reported unknown status: {report.status}, message: {report.message}')

    def _merge_bucket_keys(self, worker_id: int, n_written: int) -> None:
        """将指定 bucket 的 keys 合并到全局 bucket keys 数组中，按需扩展全局数组大小。"""
        worker_slot = self.worker_slots.get(worker_id, None)
        if worker_slot is None: raise RuntimeError(f"Worker {worker_id} not found")

        shm = worker_slot.shared_memory
        # 从共享内存中读取 bucket keys 数据
        bucket_keys_array = numpy.ndarray(shape=(self._worker_manager_config.chunk_elements,), dtype=numpy.uint64, buffer=shm.buf)
        if n_written > 0:
            # 将 bucket keys 合并到全局数组中
            self.bucket_keys.append(bucket_keys_array[:n_written].copy())  # 复制数据以避免共享内存被覆盖

    def add_subprocess(self, worker_cls: Callable[..., WorkerBase], worker_init_kwargs: dict | None = None, job_kwargs: dict | None = None) -> int:
        """创建一个新的 worker slot 并添加到队列中，target 是 worker 进程的入口函数，args 和 kwargs 是传递给 target 的参数"""
        worker_id = self._allocate_worker_slot_id()  # 自动分配一个新的 worker slot ID，确保不与现有 ID 冲突
        worker_stop_event = self._multiprocessing_context.Event()  # 创建一个新的事件对象，用于通知 worker 进程停止
        command_queue = self._multiprocessing_context.Queue()  # 创建一个新的命令队列，用于向 worker 进程发送命令
        new_shm = shared_memory.SharedMemory(create=True, size=self._worker_manager_config.shm_chunk_nbytes)
        shm_spec = ShardMemorySpec(name=new_shm.name, n_elements=self._worker_manager_config.chunk_elements)

        self.add_worker_slot(BucketWorkerSlot(
            process=self._multiprocessing_context.Process(
                target=_run_worker,
                kwargs={
                    'worker_cls': worker_cls,
                    'worker_config': BucketWorkerConfig(
                        stop_event=worker_stop_event,
                        report_queue=self._worker_report_queue,
                        worker_id=worker_id,
                        command_queue=command_queue,
                        shm_spec=shm_spec,
                    ),
                    'worker_init_kwargs': worker_init_kwargs or {},  # 传递给 Worker 实例的初始化参数
                    'job_kwargs': job_kwargs or {},  # 传递给 Worker.job 方法的参数
                }
            ),
            stop_event=worker_stop_event,
            worker_id=worker_id,
            command_queue=command_queue,
            shared_memory=new_shm
        ))
        return worker_id

    def remove_worker_slot_extra(self, slot: BucketWorkerSlot) -> None:
        """移除 worker slot 以外的资源，确保 worker slot 中的共享内存和命令队列被正确关闭和清理"""
        slot.shared_memory.close()  # 关闭共享内存对象
        slot.shared_memory.unlink()  # 解除共享内存对象，确保系统资源被释放
        slot.command_queue.close()  # 关闭命令队列，确保系统资源被释放

    def run(
        self,
        corpus_files_path: str | Path | list[str | Path],
        corpus_field_name: str | list[str],
        corpus_file_format: Literal['parquet', 'jsonl'] = 'parquet',
        **kwargs
    ) -> numpy.ndarray[numpy.uint64]:
        # 标准化输入路径格式为 list[Path]
        _corpus_files_path = corpus_files_path
        if isinstance(_corpus_files_path, str):  _corpus_files_path = [Path(_corpus_files_path)]
        elif isinstance(_corpus_files_path, Path): _corpus_files_path = [_corpus_files_path]
        elif isinstance(_corpus_files_path, list):
            for idx, path in enumerate(_corpus_files_path):
                if isinstance(path, str): _corpus_files_path[idx] = Path(path)
        else: raise ValueError(f"Invalid corpus_files_path type: {type(corpus_files_path)}")

        self.bucket_keys.clear()  # 清空全局 bucket keys 数组

        file_path_worker_mapping = {}
        for file_idx, file_path in enumerate(_corpus_files_path):
            worker_id = self.add_subprocess(
                worker_cls=BucketWorker,
                worker_init_kwargs={
                    'bucket_config': self.bucket_config,
                },
                job_kwargs={
                    'file_path': file_path,
                    'field_name': corpus_field_name,
                    'file_format': corpus_file_format,
                    **kwargs
                },
            )
            # 记录文件路径与 worker_id 的映射关系，以便后续使用
            file_path_worker_mapping[file_path] = worker_id

        while not self.is_complete: sleep(1)  # 等待所有 worker 完成任务
        self.stop()

        return numpy.concatenate(self.bucket_keys) if len(self.bucket_keys) > 0 else numpy.array([], dtype=numpy.uint64)
