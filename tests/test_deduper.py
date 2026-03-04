import struct
import sys
from pathlib import Path

import numpy
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lshcurator import Deduper
from lshcurator.algorithms import compute_minhash_signature, encode_band_key


def test_similarity_threshold_validation():
    with pytest.raises(ValueError):
        Deduper(
            bucket_keys=numpy.array([1], dtype=numpy.uint64),
            bands=4, rows_per_band=4,
            shingle_k=5, shingle_step=1,
            similarity_threshold=-0.01,
        )
    with pytest.raises(ValueError):
        Deduper(
            bucket_keys=numpy.array([1], dtype=numpy.uint64),
            bands=4, rows_per_band=4,
            shingle_k=5, shingle_step=1,
            similarity_threshold=1.01,
        )


def test_bucket_keys_sorted_in_place():
    arr = numpy.array([9, 2, 7, 2], dtype=numpy.uint64)  # 未排序
    _ = Deduper(
        bucket_keys=arr,
        bands=2, rows_per_band=4,
        shingle_k=5, shingle_step=1,
        similarity_threshold=0.9,
    )
    # Deduper.__init__ 会对传入数组原地 sort()
    assert numpy.all(arr[:-1] <= arr[1:])


def test_when_key_not_in_bucket_keys_should_keep_and_not_build_buckets():
    # bucket_keys 为空 => 所有 key 都会被跳过 => 永远 keep，且 _buckets 为空
    d = Deduper(
        bucket_keys=numpy.empty((0,), dtype=numpy.uint64),
        bands=8, rows_per_band=4,
        shingle_k=5, shingle_step=1,
        similarity_threshold=0.9,
        compute_mode="char",
    )
    assert d("same text") is True
    assert d("same text") is True
    assert len(d._buckets) == 0


def test_drop_identical_text_when_keys_present():
    bands, rows = 8, 4
    num_perm = bands * rows

    # 预先为目标文本算出其 keys，并作为 bucket_keys（确保 membership 命中）
    hash_values = compute_minhash_signature(
        text="hello", num_perm=num_perm,
        shingle_k=5, shingle_step=1, compute_mode="char"
    ).hashvalues.astype(numpy.uint64, copy=False)

    keys = []
    for band_idx in range(bands):
        digest8_key = encode_band_key(
            hash_values=hash_values,
            rows_per_band=rows,
            band_idx=band_idx,
            output_type='digest8',
        )
        keys.append(numpy.frombuffer(digest8_key, dtype="<u8", count=1)[0])

    bucket_keys = numpy.array(keys, dtype=numpy.uint64)

    d = Deduper(
        bucket_keys=bucket_keys,
        bands=bands, rows_per_band=rows,
        shingle_k=5, shingle_step=1,
        similarity_threshold=0.95,  # identical => 1.0 >= 0.95 => drop
        compute_mode="char",
    )

    assert d("hello") is True
    assert d("hello") is False
    assert len(d._buckets) > 0  # 已经入桶


def test_max_representatives_per_bucket_caps_growth(monkeypatch):
    """
    用“每个 band 固定 key”的 encode，使所有文本都落到相同 buckets，
    这样可以测试 reps 上限是否生效，以及 hit_count 是否随 keep 增长。
    """
    def _make_encode_band_key_constant_per_band():
        """
        返回一个 encode_band_key 实现：
        - 只与 band_idx 相关（每个 band 一个固定 key）
        - 这样不同文本会落入同一组桶，便于测试 reps 上限/命中计数
        """
        def _enc(*, hash_values: numpy.ndarray, rows_per_band: int, band_idx: int, output_type: str) -> bytes:
            assert output_type == "digest8"
            key = numpy.uint64(0xABCD0000_00000000 | (band_idx & 0xFFFF))
            return struct.pack("<Q", int(key))
        return _enc

    from lshcurator import deduper
    monkeypatch.setattr(deduper, "encode_band_key", _make_encode_band_key_constant_per_band())

    bands, rows = 4, 4
    # bucket_keys 必须包含这些固定 key，否则不会入桶
    fixed_keys = numpy.array(
        [numpy.uint64(0xABCD0000_00000000 | (i & 0xFFFF)) for i in range(bands)],
        dtype=numpy.uint64,
    )

    d = Deduper(
        bucket_keys=fixed_keys,
        bands=bands, rows_per_band=rows,
        shingle_k=5, shingle_step=1,
        similarity_threshold=0.99,   # 不同文本几乎不相等 => 都会 keep
        compute_mode="byte",
        max_representatives_per_bucket=2,
    )

    texts = [f"text-{i}" for i in range(6)]
    for t in texts:
        assert d(t) is True

    # 每个 key 一个桶；每个桶 reps 最多 2 个
    assert len(d._buckets) == bands
    for st in d._buckets.values():
        assert st.hit_count == len(texts)
        assert len(st.representatives) <= 2


def test_compute_mode_is_forwarded_to_minhash(monkeypatch):
    seen = {"mode": None}

    def _spy_minhash(*, text: str, num_perm: int, shingle_k: int, shingle_step: int, compute_mode: str):
        seen["mode"] = compute_mode
        return compute_minhash_signature(
            text=text, num_perm=num_perm,
            shingle_k=shingle_k, shingle_step=shingle_step,
            compute_mode=compute_mode,
        )

    from lshcurator import deduper
    monkeypatch.setattr(deduper, "compute_minhash_signature", _spy_minhash)

    d = Deduper(
        bucket_keys=numpy.array([numpy.uint64(0xABCD0000_00000000)], dtype=numpy.uint64),
        bands=1, rows_per_band=4,
        shingle_k=5, shingle_step=1,
        similarity_threshold=0.9,
        compute_mode="byte",
    )
    d("x")
    assert seen["mode"] == "byte"
