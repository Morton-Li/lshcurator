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
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Queue, shared_memory
from pathlib import Path
from typing import Literal

import numpy

from .bucket import bucket_worker
from .config import CuratorConfig, BucketConfig
from .utils.types import (
    ShmBucketReport, CuratorWorkerSlot, BucketShardMemorySpec, ShmBucketCommand, ShmBucketQueueGroups,
)


class Curator:
    def __init__(
        self,
        config: CuratorConfig
    ):
        self.config = config
        self._is_running = False

        self.bucket_keys = numpy.empty(0, dtype=numpy.uint64)  # 全局 bucket keys 数组，按需扩展

        self._report_queue: Queue[ShmBucketReport] = Queue()
        self._workers: dict[int, CuratorWorkerSlot] = {}

        # 负责监听 worker 进程报告的线程
        self._report_handler_thread = threading.Thread(target=self._report_handler, daemon=True)

    def _report_handler(self):
        """持续监听 worker 进程的报告队列"""
        while self._is_running:
            report: ShmBucketReport = self._report_queue.get(timeout=1)  # 等待报告，超时后继续循环检查
            if report is None: continue  # 收到 None 报告，可能是信号或占位，忽略

            if report.status == 'processing':
                if report.action == 'merge':
                    self._merge_bucket_keys(bucket_id=report.bucket_id, n_written=report.written)
                    self._workers[report.bucket_id].command_queue.put(ShmBucketCommand(
                        action='ready', kwargs={'is_ready': True}
                    ))
                    continue
                else:
                    print(f'Received unknown processing action: {report.action} from bucket {report.bucket_id}, message: {report.message}')
                    raise RuntimeError(f'Unknown processing action: {report.action}')
            elif report.status == 'complete':
                self._merge_bucket_keys(bucket_id=report.bucket_id, n_written=report.written)
                self._workers[report.bucket_id].command_queue.put(ShmBucketCommand(action='<|Exit|>'))  # 命令 worker 进程退出
                continue
            else:
                # 未知状态
                print(f'Received unknown report status: {report.status} from bucket {report.bucket_id}, message: {report.message}')
                raise RuntimeError(f'Unknown report status: {report.status}')

    def _merge_bucket_keys(self, bucket_id: int, n_written: int) -> None:
        """将指定 bucket 的 keys 合并到全局 bucket keys 数组中，按需扩展全局数组大小。"""
        worker_slot = self._workers.get(bucket_id, None)
        if worker_slot is None:
            raise RuntimeError(f"Bucket {bucket_id} not found")

        shm = worker_slot.shared_memory
        # 从共享内存中读取 bucket keys 数据
        bucket_keys_array = numpy.ndarray(shape=(self.config.chunk_elements,), dtype=numpy.uint64, buffer=shm.buf)
        # 将 bucket keys 合并到全局数组中
        self.bucket_keys = numpy.concatenate((self.bucket_keys, bucket_keys_array[:n_written]))

    def _alloc_shared_memory_for_bucket(self, bucket_id: int) -> None:
        """为指定 bucket 分配新的共享内存，并通过命令队列通知对应的 worker 进程刷新其共享内存映射。"""
        worker_slot = self._workers.get(bucket_id, None)
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

        self._workers[bucket_id] = worker_slot

    def process_corpus(
        self,
        corpus_files_path: str | Path | list[str | Path],
        corpus_field_name: str | list[str],
        corpus_file_format: Literal['parquet', 'jsonl'] = 'parquet',
        **kwargs
    ):
        # 1. 计算 bucket keys
        bucket_keys = self.compute_bucket_keys(
            corpus_files_path=corpus_files_path,
            corpus_field_name=corpus_field_name,
            corpus_file_format=corpus_file_format,
            **kwargs
        )
        # TODO: 完整的处理语料流程设计

    def compute_bucket_keys(
        self,
        corpus_files_path: str | Path | list[str | Path],
        corpus_field_name: str | list[str],
        corpus_file_format: Literal['parquet', 'jsonl'] = 'parquet',
        **kwargs
    ) -> numpy.ndarray:
        self._is_running = True
        self._report_handler_thread.start()  # 启动报告处理线程
        # 创建多进程
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []

            for worker_idx in range(self.config.max_workers):
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
                bucket_command_queue = Queue()
                queue_group = ShmBucketQueueGroups(
                    report_queue=self._report_queue,
                    command_queue=bucket_command_queue
                )
                job_kwargs = {
                    'file_path': corpus_files_path,
                    'field_name': corpus_field_name,
                    'file_format': corpus_file_format,
                }
                if kwargs is not None:
                    job_kwargs.update(kwargs)

                process = executor.submit(
                    bucket_worker,
                    config=bucket_config,
                    shm_spec=shm_spec,
                    queue_group=queue_group,
                    job_kwargs=job_kwargs,
                    bucket_id=worker_idx
                )

                self._workers[worker_idx] = CuratorWorkerSlot(
                    bucket_id=worker_idx,
                    process=process,  # 由 executor 启动后返回的 Future 对象管理
                    command_queue=bucket_command_queue,
                    shared_memory=new_shm
                )

                futures.append(process)

            for future in as_completed(futures):
                future.result()  # 等待 worker 进程完成任务

        self._is_running = False

        return self.bucket_keys
