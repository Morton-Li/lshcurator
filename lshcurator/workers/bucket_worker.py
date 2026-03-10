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
from typing import Literal

import numpy

from .base import WorkerBase, WorkerManagerBase
from ..bucket import BucketBase
from ..config import BucketConfig
from ..utils.readers import iter_parquet_batches, iter_jsonl_rows
from ..utils.types import WorkerReport, ShardMemorySpec, BucketWorkerReport, BucketWorkerCommand


class BucketWorker(WorkerBase, BucketBase):
    def __init__(
        self,
        config: BucketConfig,
        shm_spec: ShardMemorySpec,
        worker_id: int,
        command_queue: Queue[BucketWorkerCommand],
        report_queue: Queue[BucketWorkerReport],
    ):
        super().__init__(worker_id=worker_id, report_queue=report_queue, config=config)
        self.worker_status = 'ready'

        self._shm_spec = shm_spec
        self._shm = shared_memory.SharedMemory(name=shm_spec.name, create=False)  # 由上游负责创建和管理共享内存
        self._keys: numpy.ndarray = numpy.ndarray((shm_spec.n_elements,), dtype=numpy.uint64, buffer=self._shm.buf)  # 使用共享内存作为 keys 存储

        # 启动一个线程监听上游命令队列，处理扩容和清理等命令
        self.command_queue: Queue[BucketWorkerCommand] = command_queue
        self._command_listener_thread = threading.Thread(target=self._listen_for_commands, daemon=True)
        self._command_listener_thread.start()

    def set_worker_status(self, status: str) -> None: self.worker_status = status

    @property
    def paused(self) -> bool: return self.worker_status not in ['ready', 'Exit']

    def _listen_for_commands(self):
        """监听上游命令队列"""
        while True:
            try:
                command: BucketWorkerCommand = self.command_queue.get(timeout=1)  # 阻塞等待命令
            except Empty: continue  # 没有命令，继续等待

            if command.action == '<|Exit|>':
                self.worker_status = 'Exit'
                sleep(2)  # 确保正在执行的操作有时间完成
                self._shm.close()
                break  # 退出线程
            else:
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
        if self.worker_status == 'Exit': return  # 收到退出命令

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
        self.command_queue.put(BucketWorkerCommand(action='<|Exit|>'))  # 向命令队列发送退出命令，触发清理和退出流程
        self.report_queue.put(BucketWorkerReport(
            worker_id=self.worker_id,
            ShmSpec=self._shm_spec,
            written=self.keys_written,
            status='complete',
            message=f"Bucket Worker {self.worker_id} completed job with {self.keys_written} keys",
        ))

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
                    if self.worker_status == 'Exit': return  # 收到退出命令
                    self.insert(text=str(sample))
        elif file_format == 'jsonl':
            for row in iter_jsonl_rows(file_path=file_path):
                for field in field_name:
                    while self.paused: sleep(1)
                    if self.worker_status == 'Exit': return  # 收到退出命令
                    content = row.get(field, '').strip()
                    if not content: continue
                    self.insert(text=str(content))
        else: raise ValueError(f"Unsupported file format: {file_format}")

        # 有序退出
        self.command_queue.put(BucketWorkerCommand(action='<|Exit|>'))
        self.complete()


class BucketKeyWorkerManager(WorkerManagerBase):
    def __init__(self, max_workers: int = 1):
        super().__init__(max_workers=max_workers)

    def work_report_handler(self, report: WorkerReport):
        pass
