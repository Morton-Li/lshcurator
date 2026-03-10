from dataclasses import dataclass, field
from multiprocessing import Process, shared_memory
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
    worker_id: int | None = None


@dataclass(slots=True, kw_only=True)
class WorkerReport:
    worker_id: int
    status: Literal['complete', 'error', 'running']
    message: str | None = None


@dataclass(frozen=True, slots=True)
class ShmBucketCommand:
    action: str
    kwargs: dict | None = None


@dataclass(frozen=True, slots=True)
class ShmBucketReport:
    worker_idx: int
    ShmSpec: ShardMemorySpec
    written: int
    status: Literal['complete', 'error', 'processing']
    action: Literal['merge', 'clear'] | None = None  # 请求主进程的动作(扩容、清理等)，仅在 status='processing' 时有效
    message: str | None = None


@dataclass(frozen=True, slots=True)
class ShmBucketQueueGroups:
    report_queue: Queue[ShmBucketReport]  # 从 bucket 进程到主进程的报告队列
    command_queue: Queue[ShmBucketCommand]  # 从主进程到 bucket 进程的命令队列


@dataclass(slots=True)
class CuratorWorkerSlot:
    worker_id: int
    process: Process
    command_queue: Queue[ShmBucketCommand]
    shared_memory: shared_memory.SharedMemory
