from dataclasses import dataclass, field
from multiprocessing import Process, shared_memory, Event
from multiprocessing.queues import Queue
from typing import TypeAlias, Literal

import numpy


ComputeModeSet: set[str] = {'char', 'byte'}
ComputeMode: TypeAlias = Literal['char', 'byte']


@dataclass(slots=True)
class HashRepresentatives:
    representatives: list[numpy.ndarray] = field(default_factory=list)  # representative hash_values arrays (uint64, length=num_perm)
    def add_representative(self, hash_values: numpy.ndarray) -> None:
        self.representatives.append(hash_values)


@dataclass(frozen=True, slots=True)
class ShardMemorySpec:
    name: str
    n_elements: int


@dataclass(frozen=True, slots=True)
class ShardMemoryReport:
    ShmSpec: ShardMemorySpec
    written: int
    message: str | None = None


@dataclass(slots=True, kw_only=True)
class WorkerSlot:
    process: Process
    stop_event: Event  # 用于通知 worker 进程停止的事件，主进程设置此事件后 worker 进程应尽快完成当前任务并退出
    worker_id: int | None = None


@dataclass(slots=True, kw_only=True)
class WorkerReport:
    worker_id: int
    status: Literal['complete', 'error', 'running']
    message: str | None = None


@dataclass(frozen=True, slots=True)
class BucketWorkerCommand:
    action: str
    kwargs: dict | None = None


@dataclass(slots=True, kw_only=True)
class BucketWorkerReport(WorkerReport):
    ShmSpec: ShardMemorySpec
    written: int
    action: Literal['merge'] | None = None  # 请求主进程的动作，仅在 status='running' 时有效

    def __post_init__(self):
        if self.status != 'running' and self.action is not None:
            raise ValueError('Action must be None when status is not "running"')


@dataclass(slots=True, kw_only=True)
class BucketWorkerSlot(WorkerSlot):
    command_queue: Queue[BucketWorkerCommand]
    shared_memory: shared_memory.SharedMemory


@dataclass(slots=True, kw_only=True)
class CuratorWorkerSlot(WorkerSlot):
    command_queue: Queue[BucketWorkerCommand]
    shared_memory: shared_memory.SharedMemory
