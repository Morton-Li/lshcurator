import sys
from pathlib import Path

import numpy
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lshcurator import Bucket, BucketConfig


def test_invalid_compute_mode_raises():
    with pytest.raises(ValueError):
        bucket_config = BucketConfig(shingle_k=5, shingle_step=1, bands=8, rows_per_band=4, compute_mode="invalid_mode")
        Bucket(bucket_config=bucket_config)  # type: ignore[arg-type]


def test_append_keys_resizes_and_keeps_existing_data():
    bucket_config = BucketConfig(shingle_k=5, shingle_step=1, bands=4, rows_per_band=4, compute_mode="char")
    b = Bucket(bucket_config=bucket_config)
    b._keys = numpy.empty(4, dtype=numpy.uint64)
    b._keys_written = 0

    b.append_keys(numpy.array([10, 11, 12, 13], dtype=numpy.uint64))
    assert b._keys_written == 4
    assert b._keys.size == 4

    # 再追加 6 个 -> new_len=10，需要扩容
    b.append_keys(numpy.array([20, 21, 22, 23, 24, 25], dtype=numpy.uint64))
    assert b._keys_written == 10
    assert b._keys.size >= 10
    assert numpy.array_equal(b._keys[:4], numpy.array([10, 11, 12, 13], dtype=numpy.uint64))
    assert numpy.array_equal(b._keys[4:10], numpy.array([20, 21, 22, 23, 24, 25], dtype=numpy.uint64))


def test_clear_resets_state():
    bucket_config = BucketConfig(shingle_k=5, shingle_step=1, bands=4, rows_per_band=4, compute_mode="char")
    b = Bucket(bucket_config=bucket_config)
    b._keys = numpy.empty(8, dtype=numpy.uint64)
    b._keys_written = 0
    b.append_keys(numpy.array([7, 8, 9], dtype=numpy.uint64))
    assert b._keys_written == 3

    b.clear()
    assert b._keys_written == 0
    assert b._keys.dtype == numpy.uint64
    assert b._keys.size == 1_000_000
