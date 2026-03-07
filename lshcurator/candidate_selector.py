from multiprocessing import shared_memory
from pathlib import Path
from typing import Literal

import numpy

from .algorithms import compute_minhash_signature, encode_band_key
from .config import CandidateSelectorConfig
from .utils.types import ComputeMode, ShardMemorySpec


def match_keys(
    text: str,
    bucket_keys: numpy.ndarray[numpy.uint64],
    bands: int,
    rows_per_band: int,
    shingle_k: int,
    shingle_step: int,
    compute_mode: ComputeMode = 'char',
) -> list[numpy.uint64]:
    """
    Compute the MinHash signature for the given text and return the list of matched keys that exist in bucket_keys.
    Args:
        text (str): The text to be hashed.
        bucket_keys (numpy.ndarray[numpy.uint64]): The sorted array of bucket keys to match against.
        bands (int): Number of bands for LSH.
        rows_per_band (int): Number of rows per band for LSH.
        shingle_k (int): Shingle size (number of characters or bytes).
        shingle_step (int): Step size for downsampling shingles.
        compute_mode (ComputeMode): "char" for character-level shingles, "byte" for byte-level shingles.
    Returns:
        list[numpy.uint64]: The list of matched keys that exist in bucket_keys.
    """
    hash_values = compute_minhash_signature(
        text=text,
        num_perm=bands * rows_per_band,
        shingle_k=shingle_k,
        shingle_step=shingle_step,
        compute_mode=compute_mode,
    ).hashvalues.astype(numpy.uint64, copy=False)

    matched_keys: list[numpy.uint64] = []
    for band_idx in range(bands):
        digest8_key = encode_band_key(
            hash_values=hash_values,
            rows_per_band=rows_per_band,
            band_idx=band_idx,
            output_type='digest8',
        )
        key: numpy.uint64 = numpy.frombuffer(digest8_key, dtype='<u8', count=1)[0]
        # 只考虑在 bucket_keys 中存在的 key
        key_index = numpy.searchsorted(bucket_keys, key)
        if not (key_index < bucket_keys.size and bucket_keys[key_index] == key): continue  # key 不在 bucket_keys 中，跳过（在使用正确的情况下此情况为超低频，直接跳过可以节省大量资源）
        matched_keys.append(key)

    return matched_keys


class CandidateSelectorBase:
    def __init__(self, config: CandidateSelectorConfig, bucket_keys: numpy.ndarray[numpy.uint64]):
        self.config = config

        self._matched_keys: list[numpy.uint64] = []
        self._bucket_keys: numpy.ndarray[numpy.uint64] = bucket_keys
        self._bucket_keys.sort()

    @property
    def bucket_keys(self) -> numpy.ndarray[numpy.uint64]: return self._bucket_keys


class CandidateSelector(CandidateSelectorBase):
    def match_keys(self, text: str) -> None:
        self._matched_keys.extend(match_keys(
            text=text,
            bucket_keys=self.bucket_keys,
            bands=self.config.bands,
            rows_per_band=self.config.rows_per_band,
            shingle_k=self.config.shingle_k,
            shingle_step=self.config.shingle_step,
            compute_mode=self.config.compute_mode,
        ))


class ShmBucketSelector(CandidateSelectorBase):
    def __init__(self, config: CandidateSelectorConfig, shm_spec: ShardMemorySpec):
        shm = shared_memory.SharedMemory(name=shm_spec.name, create=False)  # 由上游负责提供共享内存
        bucket_keys: numpy.ndarray = numpy.ndarray((shm_spec.n_elements,), dtype=numpy.uint64, buffer=shm.buf)  # 使用共享内存作为 bucket_keys

        super().__init__(config=config, bucket_keys=bucket_keys)

    def run_job(
        self,
        file_path: Path,
        field_name: str | list[str],
        file_format: Literal['parquet', 'jsonl'] = 'parquet',
        **kwargs
    ) -> None: raise NotImplementedError()
