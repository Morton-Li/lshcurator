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
import multiprocessing
import threading
from multiprocessing import Queue, shared_memory
from multiprocessing.connection import wait
from pathlib import Path
from queue import Empty
from time import sleep
from typing import Literal

import numpy

from .bucket import bucket_worker
from .config import CuratorConfig, BucketConfig
from .utils.types import (
    ShmBucketReport, CuratorWorkerSlot, BucketShardMemorySpec, ShmBucketCommand, ShmBucketQueueGroups,
)


class CuratorBase:
    def __init__(self, config: CuratorConfig):
        self.config = config  # config 已配置 frozen 不可变，可以放心暴露

        self._is_running = False  # 进程是否正在运行的标志，控制报告处理线程的生命周期

        self._multiprocessing_context = multiprocessing.get_context(method='spawn')  # 使用 'spawn' 启动多进程，确保兼容性和稳定性

        self.bucket_keys: list[numpy.ndarray] = []  # 全局 bucket keys 列表

        self._report_queue: Queue[ShmBucketReport] = self._multiprocessing_context.Queue()
        self._workers_lock = threading.Lock()
        self._workers: dict[int, CuratorWorkerSlot] = {}

        # 负责监听 worker 进程报告的线程
        self._report_handler_thread: threading.Thread

    @property
    def worker_count(self) -> int:
        with self._workers_lock:
            return len(self._workers)
    def get_worker(self, worker_id: int) -> CuratorWorkerSlot | None:
        with self._workers_lock:
            return self._workers.get(worker_id, None)
    def worker_slots(self, snapshot: bool = True) -> list[CuratorWorkerSlot]:
        with self._workers_lock:
            return list(self._workers.values()) if snapshot else self._workers.values()
    def pop_worker(self, worker_id: int) -> CuratorWorkerSlot | None:
        slot = self.get_worker(worker_id=worker_id)
        if slot is None: raise KeyError(f"Worker {worker_id} not found")
        slot.process.join(timeout=8)
        if slot.process.is_alive():
            # 等待多时进程未退出
            print(f'Worker process {slot.process.name} did not exit in time, terminating forcefully.')
            # 强制终止
            slot.process.terminate()
            slot.process.join()  # 确保进程资源被回收
        if slot.process.exitcode != 0:
            print(f'Worker process {slot.process.name} exited with code {slot.process.exitcode}')
        slot.command_queue.close()
        slot.shared_memory.close()
        slot.shared_memory.unlink()
        with self._workers_lock:
            return self._workers.pop(worker_id, None)
    @property
    def worker_mapping(self) -> dict[int, CuratorWorkerSlot]:
        with self._workers_lock:
            return dict(self._workers)  # 返回 workers 的浅复制，避免外部修改原始字典
    def set_worker(self, worker_id: int, worker_slot: CuratorWorkerSlot) -> None:
        with self._workers_lock:
            self._workers[worker_id] = worker_slot


class Curator(CuratorBase):
    def _report_handler(self):
        """持续监听 worker 进程的报告队列"""
        while self._is_running:
            try:
                report: ShmBucketReport = self._report_queue.get(timeout=1)  # 等待报告，超时后继续循环检查
            except Empty: continue  # 没有报告，继续等待

            if report.status == 'processing':
                if report.action == 'merge':
                    self._merge_bucket_keys(bucket_id=report.bucket_id, n_written=report.written)
                    target_slot = self.get_worker(report.bucket_id)
                    if target_slot is None:
                        print(f'Received merge report for unknown bucket {report.bucket_id}, message: {report.message}')
                        self._is_running = False
                        raise RuntimeError(f'Unknown bucket ID in merge report: {report.bucket_id}')
                    target_slot.command_queue.put(ShmBucketCommand(
                        action='ready', kwargs={'is_ready': True}
                    ))
                    continue
                else:
                    print(f'Received unknown processing action: {report.action} from bucket {report.bucket_id}, message: {report.message}')
                    self._is_running = False
                    raise RuntimeError(f'Unknown processing action: {report.action}')
            elif report.status == 'complete':
                self._merge_bucket_keys(bucket_id=report.bucket_id, n_written=report.written)

                target_slot = self.get_worker(report.bucket_id)
                if target_slot is None:
                    print(f'Received complete report for unknown bucket {report.bucket_id}, message: {report.message}')
                    self._is_running = False
                    raise RuntimeError(f'Unknown bucket ID in complete report: {report.bucket_id}')
                target_slot.command_queue.put(ShmBucketCommand(action='<|Exit|>'))  # 命令 worker 进程退出
                self.pop_worker(report.bucket_id)
                continue
            else:
                # 未知状态
                print(f'Received unknown report status: {report.status} from bucket {report.bucket_id}, message: {report.message}')
                self._is_running = False
                raise RuntimeError(f'Unknown report status: {report.status}')

    def _merge_bucket_keys(self, bucket_id: int, n_written: int) -> None:
        """将指定 bucket 的 keys 合并到全局 bucket keys 数组中，按需扩展全局数组大小。"""
        worker_slot = self.get_worker(worker_id=bucket_id)
        if worker_slot is None:
            raise RuntimeError(f"Bucket {bucket_id} not found")

        shm = worker_slot.shared_memory
        # 从共享内存中读取 bucket keys 数据
        bucket_keys_array = numpy.ndarray(shape=(self.config.chunk_elements,), dtype=numpy.uint64, buffer=shm.buf)
        if n_written > 0:
            # 将 bucket keys 合并到全局数组中
            self.bucket_keys.append(bucket_keys_array[:n_written].copy())  # 复制数据以避免共享内存被覆盖

    def _alloc_shared_memory_for_bucket(self, bucket_id: int) -> None:
        """为指定 bucket 分配新的共享内存，并通过命令队列通知对应的 worker 进程刷新其共享内存映射。"""
        worker_slot = self.get_worker(worker_id=bucket_id)
        if worker_slot is None:
            raise RuntimeError(f"Bucket {bucket_id} not found")

        worker_slot.shared_memory.close()
        worker_slot.shared_memory.unlink()
        shm = shared_memory.SharedMemory(create=True, size=self.config.shm_chunk_nbytes)
        worker_slot.shared_memory = shm

        shm_spec = BucketShardMemorySpec(name=shm.name, n_elements=self.config.chunk_elements)
        worker_slot.command_queue.put(ShmBucketCommand(
            action='refresh_shm',
            kwargs={'new_shm_spec': shm_spec}
        ))

        self.set_worker(worker_id=bucket_id, worker_slot=worker_slot)  # 更新 worker 映射中的共享内存信息

    def process_corpus(
        self,
        corpus_files_path: str | Path | list[str | Path],
        corpus_field_name: str | list[str],
        corpus_file_format: Literal['parquet', 'jsonl'] = 'parquet',
        **kwargs
    ):
        self._is_running = True
        self._report_handler_thread = threading.Thread(target=self._report_handler, daemon=True)
        self._report_handler_thread.start()  # 启动报告处理线程

        # 1. 计算 bucket keys
        self._compute_bucket_keys(
            corpus_files_path=corpus_files_path,
            corpus_field_name=corpus_field_name,
            corpus_file_format=corpus_file_format,
            **kwargs
        )
        if len(self.bucket_keys) == 0:
            print("No bucket keys were computed.")
            self._is_running = False
            self._report_handler_thread.join()  # 等待报告处理线程退出，设置超时以防止死锁
            return None
        bucket_keys = numpy.concatenate(self.bucket_keys)  # 将所有 bucket keys 合并成一个大数组
        print(f"Total bucket keys computed: {len(bucket_keys)}")
        # TODO: 完整的处理语料流程设计

        self._is_running = False
        self._report_handler_thread.join()  # 等待报告处理线程退出，设置超时以防止死锁

    def _compute_bucket_keys(
        self,
        corpus_files_path: str | Path | list[str | Path],
        corpus_field_name: str | list[str],
        corpus_file_format: Literal['parquet', 'jsonl'] = 'parquet',
        **kwargs
    ) -> None:
        """计算 bucket keys"""
        _corpus_files_path = corpus_files_path
        if isinstance(_corpus_files_path, str): _corpus_files_path = [Path(_corpus_files_path)]
        elif isinstance(_corpus_files_path, Path): _corpus_files_path = [_corpus_files_path]
        elif isinstance(_corpus_files_path, list):
            for idx, path in enumerate(_corpus_files_path):
                if isinstance(path, str): _corpus_files_path[idx] = Path(path)
        else: raise ValueError(f"Invalid corpus_files_path type: {type(corpus_files_path)}")

        self.bucket_keys.clear()  # 清空全局 bucket keys 数组

        # 创建多进程
        max_workers = min(self.config.max_workers, len(corpus_files_path))
        for file_idx, file_path in enumerate(corpus_files_path):
            while self.worker_count >= max_workers:
                if not self._is_running: return None
                # 先取镜像避免被报告处理进程 delete worker_slot 后访问空槽位导致 KeyError
                worker_slots = self.worker_slots(snapshot=True)
                sentinels = [slot.process.sentinel for slot in worker_slots]
                if sentinels: wait(sentinels)  # 阻塞等待任一 worker 进程完成
                sleep(1)  # 等待一段时间让报告处理线程处理完成报告，确保资源被正确回收

            worker_idx = file_idx
            bucket_config = BucketConfig(
                shingle_k=self.config.shingle_k,
                shingle_step=self.config.shingle_step,
                bands=self.config.bands,
                rows_per_band=self.config.rows_per_band,
                compute_mode=self.config.compute_mode,
            )
            new_shm = shared_memory.SharedMemory(create=True, size=self.config.shm_chunk_nbytes)
            shm_spec = BucketShardMemorySpec(
                name=new_shm.name,
                n_elements=self.config.chunk_elements,
            )
            bucket_command_queue = self._multiprocessing_context.Queue()
            queue_group = ShmBucketQueueGroups(
                report_queue=self._report_queue,
                command_queue=bucket_command_queue
            )
            job_kwargs = {
                'file_path': file_path,
                'field_name': corpus_field_name,
                'file_format': corpus_file_format,
            }
            if kwargs is not None:
                job_kwargs.update(kwargs)

            process = self._multiprocessing_context.Process(
                target=bucket_worker,
                kwargs={
                    'config': bucket_config,
                    'shm_spec': shm_spec,
                    'queue_group': queue_group,
                    'job_kwargs': job_kwargs,
                    'bucket_id': worker_idx
                },
                name=f'BucketWorker-{worker_idx}'
            )
            process.start()

            self.set_worker(
                worker_id=worker_idx,
                worker_slot=CuratorWorkerSlot(
                    bucket_id=worker_idx,
                    process=process,  # 由 executor 启动后返回的 Future 对象管理
                    command_queue=bucket_command_queue,
                    shared_memory=new_shm
                )
            )

        # 确保所有进程均运行完毕后再返回
        while self.worker_count > 0: sleep(1)
        return None

    def compute_bucket_keys(
        self,
        corpus_files_path: str | Path | list[str | Path],
        corpus_field_name: str | list[str],
        corpus_file_format: Literal['parquet', 'jsonl'] = 'parquet',
        **kwargs
    ) -> numpy.ndarray:
        """
        计算 bucket keys 的独立接口，适用于只需要计算 bucket keys 的场景。
        """
        self._is_running = True
        self._report_handler_thread = threading.Thread(target=self._report_handler, daemon=True)
        self._report_handler_thread.start()  # 启动报告处理线程

        self._compute_bucket_keys(
            corpus_files_path=corpus_files_path,
            corpus_field_name=corpus_field_name,
            corpus_file_format=corpus_file_format,
            **kwargs
        )

        self._is_running = False
        self._report_handler_thread.join()  # 等待报告处理线程退出，设置超时以防止死锁

        return numpy.concatenate(self.bucket_keys) if len(self.bucket_keys) > 0 else numpy.array([], dtype=numpy.uint64)
