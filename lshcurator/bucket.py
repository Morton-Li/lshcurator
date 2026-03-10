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
from abc import ABC, abstractmethod

import numpy

from .algorithms import compute_minhash_signature, compute_band_keys
from .config import BucketConfig
from .utils.types import ComputeModeSet


class BucketBase(ABC):
    """Abstract base class for LSH buckets."""
    def __init__(self, *, config: BucketConfig, **kwargs):
        super().__init__(**kwargs)
        if config.compute_mode not in ComputeModeSet:
            raise ValueError(f"Invalid compute_mode: {config.compute_mode}. Must be one of {ComputeModeSet}")

        self.cfg = config

        self._keys: numpy.ndarray
        self._keys_written: int = 0  # 记录 keys 已写入长度指针

    @abstractmethod
    def append_keys(self, keys: numpy.ndarray[numpy.uint64]) -> None: ...
    @property
    def keys(self) -> numpy.ndarray:
        """Return the keys currently stored in the bucket as a numpy array."""
        if not hasattr(self, '_keys'): raise ValueError("Keys have not been initialized yet")
        return self._keys[:self._keys_written]  # 只返回已写入部分
    @property
    def keys_written(self) -> int:
        """Return the number of keys currently written in the bucket."""
        return self._keys_written

    def insert(self, text: str) -> None:
        """Insert a text into the LSH buckets based on its MinHash signature."""
        hash_values: numpy.ndarray[numpy.uint64] = compute_minhash_signature(
            text=text,
            num_perm=self.cfg.bands * self.cfg.rows_per_band,
            shingle_k=self.cfg.shingle_k,
            shingle_step=self.cfg.shingle_step,
            compute_mode=self.cfg.compute_mode,
        ).hashvalues.astype(numpy.uint64, copy=False)

        band_keys: numpy.ndarray[numpy.uint64] = compute_band_keys(
            hash_values=hash_values,
            bands=self.cfg.bands,
            rows_per_band=self.cfg.rows_per_band,
        )
        self.append_keys(band_keys)


class Bucket(BucketBase):
    def __init__(self, config: BucketConfig):
        super().__init__(config=config)
        self._keys: numpy.ndarray = numpy.empty(1_000_000, dtype=numpy.uint64)  # 使用 np 替代 list 降低开销

    def append_keys(self, keys: numpy.ndarray[numpy.uint64]) -> None:
        """Append keys to the internal storage, resizing if necessary."""
        if not isinstance(keys, numpy.ndarray): raise ValueError("Keys must be a numpy array")
        new_len = self._keys_written + len(keys)
        if new_len > self._keys.size:
            # Resize by doubling the size
            new_size = max(new_len, self._keys.size * 2)
            new_array = numpy.empty(new_size, dtype=numpy.uint64)
            new_array[:self._keys_written] = self._keys[:self._keys_written]
            self._keys = new_array
        self._keys[self._keys_written:new_len] = keys
        self._keys_written = new_len

    def batch_insert(self, texts: list[str]) -> None:
        """Insert a batch of texts into the LSH buckets."""
        for text in texts: self.insert(text)

    def extract_keys(self, min_hit_count: int | None = None) -> numpy.ndarray:
        """Extract the bucket keys as a numpy array."""
        data = self._keys[:self._keys_written].copy()  # 只考虑已写入部分

        keys, counts = numpy.unique(data, return_counts=True)  # unique 会返回有序结果无需 sort
        if min_hit_count is not None:
            keys = keys[counts >= min_hit_count]
        return keys

    def clear(self) -> None:
        """Clear all keys from the bucket."""
        self._keys = numpy.empty(1_000_000, dtype=numpy.uint64)  # 重置为初始大小防止已创建的数组过大白白占用内存
        self._keys_written = 0
