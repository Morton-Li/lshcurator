from dataclasses import dataclass
from multiprocessing import Process, shared_memory
from multiprocessing.queues import Queue
from typing import TypeAlias, Literal

import numpy


ComputeModeSet: set[str] = {'char', 'byte'}
ComputeMode: TypeAlias = Literal['char', 'byte']


@dataclass(slots=True)
class BucketState:
    representatives: list[numpy.ndarray]  # representative hash_values arrays (uint64, length=num_perm)
    hit_count: int = 0  # number of times this bucket has been hit


@dataclass(frozen=True, slots=True)
class BucketShardMemorySpec:
    name: str
    n_elements: int
    dtype: numpy.dtype = numpy.uint64


@dataclass(frozen=True, slots=True)
class ShmBucketCommand:
    action: str
    kwargs: dict | None = None


@dataclass(frozen=True, slots=True)
class ShmBucketReport:
    bucket_id: int
    ShmSpec: BucketShardMemorySpec
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
    bucket_id: int
    process: Process
    command_queue: Queue[ShmBucketCommand]
    shared_memory: shared_memory.SharedMemory
