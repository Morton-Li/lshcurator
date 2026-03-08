from multiprocessing import shared_memory
from pathlib import Path
from typing import Literal

import numpy

from .algorithms import compute_minhash_signature, encode_band_key
from .config import CandidateSelectorConfig
from .utils.readers import iter_parquet_batches, iter_jsonl_rows
from .utils.types import ComputeMode, ShardMemorySpec


def match_keys(
    text: str,
    bands: int,
    rows_per_band: int,
    shingle_k: int,
    shingle_step: int,
    bucket_keys: numpy.ndarray[numpy.uint64] | None = None,
    compute_mode: ComputeMode = 'char',
) -> list[numpy.uint64]:
    """
    Compute the MinHash signature for the given text and return the list of matched keys that exist in bucket_keys.
    Args:
        text (str): The text to be hashed.
        bands (int): Number of bands for LSH.
        rows_per_band (int): Number of rows per band for LSH.
        shingle_k (int): Shingle size (number of characters or bytes).
        shingle_step (int): Step size for downsampling shingles.
        bucket_keys (numpy.ndarray[numpy.uint64] | None): The sorted array of bucket keys to match against. If None, all keys will be considered as matched.
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
        if bucket_keys is not None:
            # 只考虑在 bucket_keys 中存在的 key
            key_index = numpy.searchsorted(bucket_keys, key)
            if not (key_index < bucket_keys.size and bucket_keys[key_index] == key): continue  # key 不在 bucket_keys 中，跳过（在使用正确的情况下此情况为超低频，直接跳过可以节省大量资源）
        matched_keys.append(key)

    return matched_keys


class CandidateSelectorBase:
    def __init__(self, config: CandidateSelectorConfig, bucket_keys: numpy.ndarray[numpy.uint64]):
        """
        Args:
            config (CandidateSelectorConfig): The configuration for the candidate selector.
            bucket_keys (numpy.ndarray[numpy.uint64]): The sorted array of bucket keys to match against.
        """
        self.config = config

        self._matched_keys: list[numpy.uint64] = []
        self._bucket_keys: numpy.ndarray[numpy.uint64] = bucket_keys

    @property
    def bucket_keys(self) -> numpy.ndarray[numpy.uint64]: return self._bucket_keys
    @property
    def matched_keys(self) -> list[numpy.uint64]: return self._matched_keys

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


class CandidateSelector(CandidateSelectorBase):
    def __init__(self, config: CandidateSelectorConfig, bucket_keys: numpy.ndarray[numpy.uint64]):
        super().__init__(config=config, bucket_keys=bucket_keys)
        self._bucket_keys.sort()  # 确保 bucket_keys 是有序的，以便后续使用 searchsorted 进行高效查找


class ShmCandidateSelector(CandidateSelectorBase):
    def __init__(self, config: CandidateSelectorConfig, bucket_keys_shm_spec: ShardMemorySpec):
        self._shm = shared_memory.SharedMemory(name=bucket_keys_shm_spec.name, create=False)  # 由上游负责提供共享内存
        bucket_keys: numpy.ndarray = numpy.ndarray((bucket_keys_shm_spec.n_elements,), dtype=numpy.uint64, buffer=self._shm.buf)  # 使用共享内存作为 bucket_keys

        super().__init__(config=config, bucket_keys=bucket_keys)

    def run_job(
        self,
        file_path: Path,
        field_name: str | list[str],
        file_format: Literal['parquet', 'jsonl'] = 'parquet',
        **kwargs
    ) -> None:
        self._matched_keys.clear()
        if isinstance(field_name, str): field_name = [field_name]

        if file_format == 'parquet':
            for batch in iter_parquet_batches(
                parquet_path=file_path,
                batch_size=kwargs.get('batch_size', 2048),
                text_field=field_name,
            ):
                for sample in batch.stack().replace(r'^\s*$', numpy.nan, regex=True).dropna().reset_index(drop=True):
                    self.match_keys(text=str(sample))
        elif file_format == 'jsonl':
            for row in iter_jsonl_rows(file_path=file_path):
                for field in field_name:
                    content = row.get(field, '').strip()
                    if not content: continue
                    self.match_keys(text=str(content))
        else:
            self._shm.close()
            raise ValueError(f"Unsupported file format: {file_format}")

        self._shm.close()  # 进程结束前关闭共享内存


def candidate_selector_worker(
    config: CandidateSelectorConfig,
    bucket_keys_shm_spec: ShardMemorySpec,
    job_kwargs: dict,
) -> None:
    """CandidateSelector 进程的主函数，负责创建 ShmCandidateSelector 实例并运行任务"""
    worker = ShmCandidateSelector(config=config, bucket_keys_shm_spec=bucket_keys_shm_spec)
    worker.run_job(**job_kwargs)
    matched_keys: list[numpy.uint64] = worker.matched_keys


__all__ = ['CandidateSelector', 'ShmCandidateSelector', 'candidate_selector_worker']
