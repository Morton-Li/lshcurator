from typing import Literal

import numpy

from .types import BucketState
from .utils import compute_minhash_signature, encode_band_key


class Deduper:
    def __init__(
        self,
        bucket_keys: numpy.ndarray,
        bands: int,
        rows_per_band: int,
        shingle_k: int,
        shingle_step: int,
        similarity_threshold: float,
        compute_mode: Literal['char', 'byte'] = 'char',
        max_representatives_per_bucket: int | None = None,
    ):
        self.num_perm = bands * rows_per_band
        self.bands = bands
        self.rows_per_band = rows_per_band
        self.shingle_k = shingle_k
        self.shingle_step = shingle_step
        if not (0 <= similarity_threshold <= 1):
            raise ValueError(f'similarity_threshold must be in [0, 1], got {similarity_threshold}')
        self.similarity_threshold = similarity_threshold
        self.compute_mode = compute_mode
        self.max_representatives_per_bucket = max_representatives_per_bucket

        self._bucket_keys: numpy.ndarray = bucket_keys
        self._bucket_keys.sort()
        self._buckets: dict[int, BucketState] = {}

    def __call__(self, text: str) -> bool:
        hash_values = compute_minhash_signature(
            text=text,
            num_perm=self.num_perm,
            shingle_k=self.shingle_k,
            shingle_step=self.shingle_step,
            compute_mode=self.compute_mode,
        ).hashvalues.astype(numpy.uint64, copy=False)

        keys: list[numpy.uint64] = []
        seen_rep_ids: set[int] = set()
        for band_idx in range(self.bands):
            digest8_key = encode_band_key(
                hash_values=hash_values,
                rows_per_band=self.rows_per_band,
                band_idx=band_idx,
                output_type='digest8',
            )
            key: numpy.uint64 = numpy.frombuffer(digest8_key, dtype='<u8', count=1)[0]
            # 只考虑在 bucket_keys 中存在的 key
            key_index = numpy.searchsorted(self._bucket_keys, key)
            if not (key_index < self._bucket_keys.size and self._bucket_keys[key_index] == key): continue  # key 不在 bucket_keys 中，跳过（在使用正确的情况下此情况为超低频，直接跳过可以节省大量资源）
            keys.append(key)

            st = self._buckets.get(int(key), None)
            if st is None: continue
            for rep in st.representatives:
                rid = id(rep)
                if rid not in seen_rep_ids:
                    seen_rep_ids.add(rid)
                    if float(numpy.mean(hash_values == rep)) >= self.similarity_threshold: return False

        # 到这里没有触发 return False，说明没有找到相似文本，可以认为是一个新的文本，接下来将其加入桶中
        for key in keys:
            st = self._buckets.get(int(key))
            if st is None: self._buckets[int(key)] = BucketState(representatives=[hash_values], hit_count=1)
            else:
                st.hit_count += 1
                if self.max_representatives_per_bucket is None or len(st.representatives) < self.max_representatives_per_bucket:
                    st.representatives.append(hash_values)

        return True
