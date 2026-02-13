import numpy
import pytest

from lshcurator import Bucket


def test_invalid_compute_mode_raises():
    with pytest.raises(ValueError):
        Bucket(shingle_k=5, shingle_step=1, bands=8, rows_per_band=4, compute_mode="token")  # type: ignore[arg-type]


def test_append_keys_resizes_and_keeps_existing_data():
    b = Bucket(shingle_k=5, shingle_step=1, bands=4, rows_per_band=4, compute_mode="char")
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


def test_insert_same_text_twice_extract_min_hit_count():
    bands, rows = 8, 4
    b = Bucket(shingle_k=5, shingle_step=1, bands=bands, rows_per_band=rows, compute_mode="byte")
    b._keys = numpy.empty(64, dtype=numpy.uint64)
    b._keys_written = 0

    b.insert("same text")
    assert b._keys_written == bands  # 每条样本写入 bands 个 key
    b.insert("same text")

    # 所有 band 的 key 都会出现 2 次
    keys = b.extract_keys(min_hit_count=2)
    assert keys.dtype == numpy.uint64
    assert keys.size == bands


def test_extract_keys_filters_counts_correctly():
    b = Bucket(shingle_k=5, shingle_step=1, bands=4, rows_per_band=4, compute_mode="char")
    b._keys = numpy.empty(16, dtype=numpy.uint64)
    b._keys_written = 0

    # 构造：1出现3次，2出现1次，3出现2次
    b.append_keys(numpy.array([1, 1, 1, 2, 3, 3], dtype=numpy.uint64))

    keys_ge2 = b.extract_keys(min_hit_count=2)
    assert numpy.array_equal(keys_ge2, numpy.array([1, 3], dtype=numpy.uint64))


def test_clear_resets_state():
    b = Bucket(shingle_k=5, shingle_step=1, bands=4, rows_per_band=4, compute_mode="char")
    b._keys = numpy.empty(8, dtype=numpy.uint64)
    b._keys_written = 0
    b.append_keys(numpy.array([7, 8, 9], dtype=numpy.uint64))
    assert b._keys_written == 3

    b.clear()
    assert b._keys_written == 0
    assert b._keys.dtype == numpy.uint64
    assert b._keys.size == 1_000_000
