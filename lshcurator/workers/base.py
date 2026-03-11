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
from abc import ABC, abstractmethod
from queue import Empty, Queue
from time import sleep
from typing import Callable

from ..config import WorkerConfig
from ..utils.types import WorkerSlot, WorkerReport


class WorkerBase(ABC):
    """Base class for workers."""
    def __init__(self, *, worker_config: WorkerConfig, **kwargs):
        """
        Args:
            worker_id (int): Worker 的唯一标识符，主进程通过此 ID 来区分不同的 worker 进程。
            report_queue (multiprocessing.Queue): 用于向主进程发送报告的队列，worker 进程通过此队列将状态和结果报告给主进程。
        """
        super().__init__(**kwargs)
        self._worker_config: WorkerConfig = worker_config

    @property
    def worker_id(self) -> int: return self._worker_config.worker_id
    @property
    def report_queue(self) -> multiprocessing.Queue[WorkerReport]: return self._worker_config.report_queue
    @property
    def stop_event(self) -> multiprocessing.Event: return self._worker_config.stop_event

    @abstractmethod
    def job(self, **kwargs):
        """Worker 的具体任务逻辑"""
        raise NotImplementedError('Subclasses must implement job to define the worker\'s task logic')

    def complete(self):
        """Report completion to the main process. Should be called by the worker process when its task is complete."""
        self.report_queue.put(WorkerReport(worker_id=self.worker_id, status='complete', message='Worker task completed successfully'))


def _run_worker(
    worker_cls: Callable[..., WorkerBase],
    worker_config: WorkerConfig,
    worker_init_kwargs: dict,
    job_kwargs: dict,
):
    """Worker 进程的入口函数，负责创建 Worker 实例并执行任务"""
    try:
        worker = worker_cls(worker_config=worker_config, **worker_init_kwargs)  # 创建 Worker 实例，传递 worker_id 和 report_queue 以及其他必要参数
        worker.job(**job_kwargs)  # 执行 Worker 的具体任务逻辑
    except Exception as e:
        worker_config.report_queue.put(WorkerReport(worker_id=worker_config.worker_id, status='error', message=f'Worker encountered an error: {str(e)}'))  # 向主进程报告错误


class WorkerManagerBase(ABC):
    """Base class for worker managers."""
    def __init__(self, max_workers: int = 1):
        """
        Args:
            max_workers (int): 最大 worker 数量，默认为 1。WorkerManagerBase 将根据此参数限制同时运行的 worker 进程数量，确保资源被合理利用。
        """
        self.max_workers = max_workers

        self._worker_slots_lock = threading.Lock()  # 防止主进程内多线程竞争

        self._multiprocessing_context = multiprocessing.get_context(method='spawn')  # 使用 'spawn' 启动多进程，确保兼容性和稳定性
        self._worker_slots: dict[int, WorkerSlot] = {}
        self._worker_slots_ids: list[int] = []  # 维护一个单独的 worker slot ID 列表，方便快速检查和分配新的 ID，避免与现有 ID 冲突

        self._worker_report_queue: multiprocessing.Queue[WorkerReport] = self._multiprocessing_context.Queue()  # 用于接收 worker 进程的报告，主进程监听此队列以获取 worker 状态和结果
        self._add_subprocess_queue: Queue[WorkerSlot] = Queue()  # 用于接收需要添加的 worker slot，主进程监听此队列以动态添加 worker slot 和启动对应的 worker 进程

        self._worker_report_handler_is_running = True  # 控制 worker 报告处理线程的运行状态
        self._add_subprocess_handler_is_running = True
        self._worker_report_handler_thread: threading.Thread = threading.Thread(target=self._worker_report_handler, daemon=True)  # 后台线程，专门处理 worker 报告队列中的消息
        self._worker_report_handler_thread.start()
        self._add_subprocess_thread: threading.Thread = threading.Thread(target=self._add_subprocess_handler, daemon=True)  # 后台线程，专门处理添加 worker slot 的请求
        self._add_subprocess_thread.start()

    @property
    def worker_slots(self) -> dict[int, WorkerSlot]:
        with self._worker_slots_lock:
            return dict(self._worker_slots)  # 返回 _worker_slots 的浅复制，避免外部修改原始字典

    def _allocate_worker_slot_id(self) -> int:
        """自动分配一个新的 worker slot ID，确保不与现有 ID 冲突"""
        with self._worker_slots_lock:
            new_id = max(self._worker_slots_ids, default=-1) + 1
            self._worker_slots_ids.append(new_id)
        return new_id

    def set_worker_slot(self, slot: WorkerSlot) -> None:
        """设置指定 idx 的 worker slot"""
        worker_id = slot.worker_id
        if worker_id is None: raise ValueError('Worker slot must have a worker_id')
        if worker_id not in self.worker_slots: self.add_worker_slot(slot)  # 如果 worker_id 不存在，则添加新的 worker slot
        else:
            with self._worker_slots_lock:
                self._worker_slots[worker_id] = slot

    def add_worker_slot(self, slot: WorkerSlot) -> None:
        """添加一个新的 worker slot"""
        worker_id = slot.worker_id
        if worker_id is not None and worker_id in self.worker_slots: raise IndexError(f'Worker slot {worker_id} already exists')
        self._add_subprocess_queue.put(slot)

    def add_subprocess(self, worker_cls: Callable[..., WorkerBase], worker_init_kwargs: dict | None = None, job_kwargs: dict | None = None) -> int:
        """创建一个新的 worker slot 并添加到队列中，target 是 worker 进程的入口函数，args 和 kwargs 是传递给 target 的参数"""
        worker_id = self._allocate_worker_slot_id()  # 自动分配一个新的 worker slot ID，确保不与现有 ID 冲突
        worker_stop_event = self._multiprocessing_context.Event()  # 创建一个新的事件对象，用于通知 worker 进程停止
        self.add_worker_slot(WorkerSlot(
            process=self._multiprocessing_context.Process(
                target=_run_worker,
                kwargs={
                    'worker_cls': worker_cls,
                    'worker_config': WorkerConfig(
                        stop_event=worker_stop_event,
                        report_queue=self._worker_report_queue,
                        worker_id=worker_id,
                    ),
                    'worker_init_kwargs': worker_init_kwargs or {},  # 传递给 Worker 实例的初始化参数
                    'job_kwargs': job_kwargs or {},  # 传递给 Worker.job 方法的参数
                }
            ),
            stop_event=worker_stop_event,
            worker_id=worker_id,
        ))
        return worker_id

    def _add_subprocess_handler(self):
        """后台线程方法，持续监听添加 worker slot 的请求队列并处理消息"""
        while self._add_subprocess_handler_is_running:
            if len(self.worker_slots) >= self.max_workers:
                sleep(1)
                continue  # 已达到最大 worker 数量，等待 1 秒后继续检查

            try: slot: WorkerSlot = self._add_subprocess_queue.get(timeout=1)  # 等待 1 秒获取添加请求，避免无限阻塞
            except Empty: continue  # 队列为空，继续等待

            # 再次确认管理器仍然在运行
            if not self._add_subprocess_handler_is_running: break

            if slot.worker_id is None: slot.worker_id = self._allocate_worker_slot_id()  # 自动分配一个新的 idx，确保不与现有 idx 冲突
            with self._worker_slots_lock:
                self._worker_slots[slot.worker_id] = slot
            slot.process.start()  # 启动对应的 worker 进程

    def remove_worker_slot(self, worker_id: int) -> None:
        """移除指定 idx 的 worker slot"""
        with self._worker_slots_lock:  # 全程持锁防止同时释放同一 worker slot 导致的竞争条件
            if worker_id not in self._worker_slots: return  # worker_id 不存在，无需移除
            self.stop_subprocesses(worker_id=worker_id)  # 停止 worker 进程，确保资源被正确释放
            del self._worker_slots[worker_id]  # 从 worker_slots 中移除 worker slot
            self._worker_slots_ids.remove(worker_id)  # 从 worker slot ID 列表中移除 worker_id，确保未来分配的新 ID 不会与已存在的 ID 冲突

    def _worker_report_handler(self):
        """后台线程方法，持续监听 worker 报告队列并处理消息"""
        while self._worker_report_handler_is_running:
            try: report: WorkerReport = self._worker_report_queue.get(timeout=1)  # 等待 1 秒获取报告，避免无限阻塞
            except Empty: continue  # 队列为空，继续等待
            self.work_report_handler(report)
            # 根据报告状态自动移除已完成的 worker slot，确保资源被及时释放
            if report.status == 'complete' or report.status == 'error': self.remove_worker_slot(report.worker_id)

    @abstractmethod
    def work_report_handler(self, report: WorkerReport):
        """处理 worker 报告的具体逻辑"""
        raise NotImplementedError('Subclasses must implement work_report_handler to process worker reports')

    def stop_subprocesses(self, worker_id: int):
        """
        停止指定 worker 进程，确保资源被正确释放
        注意：
        全程不持锁，需要上游保证不会同时调用 stop_subprocesses 来停止同一个 worker_id，避免竞争条件导致的异常行为
        """
        slot = self._worker_slots.get(worker_id, None)
        if slot is None: return  # worker_id 不存在，无需停止
        slot.stop_event.set()  # 设置停止事件，通知 worker 进程尽快完成当前任务并退出
        process = slot.process
        if process.is_alive():  # 确保进程仍然存活后才尝试等待和终止，避免不必要的操作
            process.join(timeout=8)  # 等待 worker 进程结束
            if process.is_alive():
                process.terminate()  # 强制终止仍然存活的 worker 进程
                process.join(timeout=8)
                if process.is_alive(): print(f'Warning: Worker process {process.pid} is still alive after termination attempt')

    def run(self, *args, **kwargs):
        """启动 worker 进程的逻辑(如需要)"""
        raise NotImplementedError('Subclasses must implement run to define how to start worker processes if needed')

    def stop(self):
        """等待所有 worker 进程结束，并停止报告处理线程"""
        self._add_subprocess_handler_is_running = False  # 停止添加 worker slot 的线程
        self._add_subprocess_thread.join()  # 等待添加 worker slot 的线程结束
        self._add_subprocess_queue.shutdown(immediate=True)  # 关闭添加 worker slot 的队列，确保没有未处理的添加请求

        for slot_idx in self.worker_slots.keys(): self.remove_worker_slot(slot_idx)  # 移除 worker slot 时会自动停止对应的 worker 进程

        self._worker_report_handler_is_running = False  # 停止报告处理线程
        self._worker_report_handler_thread.join()  # 等待报告处理线程结束
