from typing import Literal

import numpy

from .utils import compute_minhash_signature, encode_band_key


class Bucket:
    def __init__(
        self,
        shingle_k: int,
        shingle_step: int,
        bands: int,
        rows_per_band: int,
        compute_mode: Literal['char', 'byte'] = 'char',
    ):
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
        self._keys: numpy.ndarray = numpy.empty(1_000_000, dtype=numpy.uint64)  # 使用 np 替代 list 降低开销
        self._keys_written: int = 0  # 记录 keys 已写入长度指针

        self.shingle_k = shingle_k
        self.shingle_step = shingle_step
        self.bands = bands
        self.rows_per_band = rows_per_band
        self.num_perm = bands * rows_per_band

        if compute_mode not in ('char', 'byte'):
            raise ValueError(f'Invalid compute_mode: {compute_mode}, expected "char" or "byte"')
        self.compute_mode = compute_mode

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
            num_perm=self.num_perm,
            shingle_k=self.shingle_k,
            shingle_step=self.shingle_step,
            compute_mode=self.compute_mode,
        ).hashvalues.astype(numpy.uint64, copy=False)

        keys = numpy.empty(self.bands, dtype=numpy.uint64)  # 使用 np 以替代 list 降低开销
        for band_idx in range(self.bands):
            digest8_key = encode_band_key(
                hash_values=hash_values,
                rows_per_band=self.rows_per_band,
                band_idx=band_idx,
                output_type='digest8',
            )
            keys[band_idx] = numpy.frombuffer(digest8_key, dtype='<u8', count=1)[0]
        self.append_keys(keys)

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
