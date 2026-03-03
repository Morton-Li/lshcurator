import numpy

from .algorithms import compute_minhash_signature, compute_band_keys
from .config import BucketConfig


class Bucket:
    def __init__(self, config: BucketConfig):
        self._keys: numpy.ndarray = numpy.empty(1_000_000, dtype=numpy.uint64)  # 使用 np 替代 list 降低开销
        self._keys_written: int = 0  # 记录 keys 已写入长度指针

        self.cfg = config

    def append_keys(self, keys: numpy.ndarray) -> None:
        """Append keys to the internal storage, resizing if necessary."""
        new_len = self._keys_written + len(keys)
        if new_len > self._keys.size:
            # Resize by doubling the size
            new_size = max(new_len, self._keys.size * 2)
            new_array = numpy.empty(new_size, dtype=numpy.uint64)
            new_array[:self._keys_written] = self._keys[:self._keys_written]
            self._keys = new_array
        self._keys[self._keys_written:new_len] = keys
        self._keys_written = new_len

    def insert(self, text: str) -> None:
        """Insert a text into the LSH buckets based on its MinHash signature."""
        hash_values = compute_minhash_signature(
            text=text,
            num_perm=self.cfg.bands * self.cfg.rows_per_band,
            shingle_k=self.cfg.shingle_k,
            shingle_step=self.cfg.shingle_step,
            compute_mode=self.cfg.compute_mode,
        ).hashvalues.astype(numpy.uint64, copy=False)

        band_keys = compute_band_keys(
            hash_values=hash_values,
            bands=self.cfg.bands,
            rows_per_band=self.cfg.rows_per_band,
        )
        self.append_keys(band_keys)

    def batch_insert(self, texts: list[str]) -> None:
        """Insert a batch of texts into the LSH buckets."""
        for text in texts: self.insert(text)

    def extract_keys(self, min_hit_count: int | None = None) -> numpy.ndarray:
        """Extract the bucket keys as a numpy array."""
        data = self._keys[:self._keys_written].copy()  # 只考虑已写入部分
        data.sort()

        keys, counts = numpy.unique(data, return_counts=True)
        if min_hit_count is not None:
            keys = keys[counts >= min_hit_count]
        return keys

    def clear(self) -> None:
        """Clear all keys from the bucket."""
        self._keys = numpy.empty(1_000_000, dtype=numpy.uint64)  # 重置为初始大小
        self._keys_written = 0
