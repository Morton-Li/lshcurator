import numpy

from .algorithms import compute_minhash_signature, encode_band_key
from .config import DeduperConfig
from .utils.types import HashRepresentatives


class Deduper:
    def __init__(
        self,
        bucket_keys: numpy.ndarray,
        config: DeduperConfig,
    ):
        """
        Args:
            bucket_keys (numpy.ndarray): 预先计算好的桶键列表，必须是 numpy.uint64 类型的 1D 数组，且已经排序（升序）。Deduper 将只考虑这些桶键来判断文本是否重复。
            config (DeduperConfig): Deduper 的配置对象，包含 LSH 参数和相似度阈值等设置。
        """
        self.config = config
        if not (0 <= self.config.similarity_threshold <= 1):
            raise ValueError(f'similarity_threshold must be in [0, 1], got {self.config.similarity_threshold}')

        self._bucket_keys: numpy.ndarray = bucket_keys
        self._buckets: dict[int, HashRepresentatives] = {}

    @property
    def bucket_keys(self) -> numpy.ndarray: return self._bucket_keys
    @property
    def buckets(self) -> dict[int, HashRepresentatives]: return self._buckets
    @property
    def num_buckets(self) -> int: return len(self._buckets)
    @property
    def num_bucket_keys(self) -> int: return self._bucket_keys.size

    def __call__(self, text: str) -> bool:
        """
        Args:
            text (str): 待检查的文本字符串。
        Returns:
            bool: 如果文本被认为是新的（不重复），返回 True；如果文本被认为是重复的，返回 False。
        """
        hash_values = compute_minhash_signature(
            text=text,
            num_perm=self.config.num_perm,
            shingle_k=self.config.shingle_k,
            shingle_step=self.config.shingle_step,
            compute_mode=self.config.compute_mode,
        ).hashvalues.astype(numpy.uint64, copy=False)

        matched_keys: list[numpy.uint64] = []
        seen_rep_ids: set[int] = set()
        for band_idx in range(self.config.bands):
            digest8_key = encode_band_key(
                hash_values=hash_values,
                rows_per_band=self.config.rows_per_band,
                band_idx=band_idx,
                output_type='digest8',
            )
            key: numpy.uint64 = numpy.frombuffer(digest8_key, dtype='<u8', count=1)[0]
            # 只考虑在 bucket_keys 中存在的 key
            key_index = numpy.searchsorted(self.bucket_keys, key)
            if not (key_index < self.bucket_keys.size and self.bucket_keys[key_index] == key): continue  # key 不在 bucket_keys 中，跳过（在使用正确的情况下此情况为超低频，直接跳过可以节省大量资源）
            matched_keys.append(key)

            st = self._buckets.get(int(key), None)
            if st is None: continue
            for rep in st.representatives:
                rid = id(rep)
                if rid not in seen_rep_ids:
                    seen_rep_ids.add(rid)
                    if float(numpy.mean(hash_values == rep)) >= self.config.similarity_threshold: return False

        # 到这里没有触发 return False，说明没有找到相似文本，可以认为是一个新的文本，接下来将其加入桶中
        for key in matched_keys:
            hr = self._buckets.get(int(key), None)
            if hr is None: self._buckets[int(key)] = HashRepresentatives(representatives=[hash_values])
            elif self.config.max_representatives_per_bucket is None or len(hr.representatives) < self.config.max_representatives_per_bucket:
                    hr.add_representative(hash_values)

        return True
