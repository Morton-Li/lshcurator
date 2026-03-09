from dataclasses import dataclass

import numpy

from .utils.types import ComputeMode


@dataclass(frozen=True, slots=True)
class BucketConfig:
    """
    Args:
        shingle_k (int):
            Shingle size (number of characters or bytes)
            按字符切割成长度为 k 的连续子串集合
            例如 text="abcdef", k=3, step=1 时，生成的 shingles 包括 "abc", "bcd", "cde", "def"
        shingle_step (int):
            Step size for downsampling shingles (e.g., 3 means take every 3rd shingle)
            滑窗每次移动 step 个字符，枚举所有 k-gram。
            当 step > 1 时，表示对 shingles 进行下采样，更快但信息更少
            当 step=1 时，枚举所有 k-gram
            当 step=k 时，枚举不重叠的 k-gram，例如 text="abcdef", k=3, step=3 时，生成的 shingles 包括 "abc", "def"（跳过 "bcd", "cde"）
            当 step>k 时，枚举稀疏的 k-gram，例如 text="abcdefgh", k=3, step=4 时，生成的 shingles 包括 "abc", "efg"（跳过 "bcd", "cde", "def", "fgh"）
        bands (int): Number of bands for LSH.
        rows_per_band (int): Number of rows per band for LSH.
        compute_mode (str):
            "char" for character-level shingles, "byte" for byte-level shingles.
            计算模式，"char" 表示按字符切割 shingles，"byte" 表示按字节切割 shingles。
    """
    shingle_k: int
    shingle_step: int
    bands: int
    rows_per_band: int
    compute_mode: ComputeMode = 'char'


@dataclass(frozen=True, slots=True)
class DeduperConfig:
    bands: int
    rows_per_band: int
    shingle_k: int
    shingle_step: int
    similarity_threshold: float
    compute_mode: ComputeMode = 'char'
    max_representatives_per_bucket: int | None = None

    @property
    def num_perm(self) -> int: return self.bands * self.rows_per_band


@dataclass(frozen=True, slots=True)
class CuratorConfig:
    shingle_k: int
    shingle_step: int
    bands: int
    rows_per_band: int
    similarity_threshold: float
    compute_mode: ComputeMode = 'char'

    max_workers: int = 8
    chunk_elements: int = 1_000_000  # 每次分配共享内存的元素数量

    max_representatives_per_bucket: int | None = None

    @property
    def shm_chunk_nbytes(self) -> int: return self.chunk_elements * numpy.dtype('uint64').itemsize
