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
    def __init__(self, *, bucket_config: BucketConfig, **kwargs):
        super().__init__(**kwargs)
        if bucket_config.compute_mode not in ComputeModeSet:
            raise ValueError(f"Invalid compute_mode: {bucket_config.compute_mode}. Must be one of {ComputeModeSet}")

        self._bucket_config = bucket_config

        self._keys: numpy.ndarray
        self._keys_written: int = 0  # 记录 keys 已写入长度指针

    @property
    def keys_written(self) -> int:
        """Return the number of keys currently written in the bucket."""
        return self._keys_written

    def insert(self, text: str) -> None:
        """Insert a text into the LSH buckets based on its MinHash signature."""
        hash_values: numpy.ndarray[numpy.uint64] = compute_minhash_signature(
            text=text,
            num_perm=self._bucket_config.bands * self._bucket_config.rows_per_band,
            shingle_k=self._bucket_config.shingle_k,
            shingle_step=self._bucket_config.shingle_step,
            compute_mode=self._bucket_config.compute_mode,
        ).hashvalues.astype(numpy.uint64, copy=False)

        band_keys: numpy.ndarray[numpy.uint64] = compute_band_keys(
            hash_values=hash_values,
            bands=self._bucket_config.bands,
            rows_per_band=self._bucket_config.rows_per_band,
        )

        self.append_keys(band_keys)

    @abstractmethod
    def append_keys(self, keys: numpy.ndarray[numpy.uint64]) -> None: ...

    def extract_keys(self) -> numpy.ndarray:
        """
        Return the keys currently stored in the bucket as a numpy array.
        Returns:
            numpy.ndarray:
                Array of keys.
                当 key_layout='flat' 时，返回 shape=(num_keys,) 的 1D 数组，每个元素是一个 key。
                当 key_layout='row_bands' 时，返回 shape=(num_samples, bands) 的 2D 数组，每行对应一个样本的所有 band keys，列数等于 bands。
        """
        if not hasattr(self, '_keys'): raise ValueError("Keys have not been initialized yet")

        data = self._keys[:self._keys_written].copy()  # 只返回已写入部分

        if self._bucket_config.key_layout == 'flat': return data
        elif self._bucket_config.key_layout == 'row_bands':
            # 将 flat keys 重新组织成 (num_keys, bands) 的二维结构，每行对应一个样本的所有 band keys
            if self._bucket_config.bands <= 0: raise ValueError("Bands must be positive for row_bands layout")
            if self._keys_written % self._bucket_config.bands != 0:
                raise ValueError(f"Keys written ({self._keys_written}) is not a multiple of bands ({self._bucket_config.bands})")
            return data.reshape(-1, self._bucket_config.bands)
        else: raise ValueError(f"Invalid key_layout: {self._bucket_config.key_layout}")

    def clear(self) -> None:
        """Clear all keys from the bucket."""
        # 子类应根据实际存储结构实现清空逻辑，这里只重置了 keys_written 指针（简单有效但不彻底）
        self._keys_written = 0


class Bucket(BucketBase):
    def __init__(self, bucket_config: BucketConfig):
        super().__init__(bucket_config=bucket_config)
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

    def clear(self) -> None:
        """Clear all keys from the bucket."""
        self._keys = numpy.empty(1_000_000, dtype=numpy.uint64)  # 重置为初始大小防止已创建的数组过大白白占用内存
        super().clear()
