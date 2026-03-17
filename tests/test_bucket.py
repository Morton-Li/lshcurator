import sys
from pathlib import Path

import numpy
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lshcurator import Bucket, BucketConfig


def _u64(values: list[int]) -> numpy.ndarray:
    return numpy.array(values, dtype=numpy.uint64).astype(numpy.uint64, copy=False)


def test_invalid_compute_mode_raises():
    with pytest.raises(ValueError):
        bucket_config = BucketConfig(shingle_k=5, shingle_step=1, bands=8, rows_per_band=4, compute_mode="invalid_mode")  # type: ignore[arg-type]
        Bucket(bucket_config=bucket_config)  # type: ignore[arg-type]


def test_append_keys_resizes_and_keeps_existing_data():
    bucket_config = BucketConfig(shingle_k=5, shingle_step=1, bands=4, rows_per_band=4, compute_mode="char")
    b = Bucket(bucket_config=bucket_config)
    b._keys = numpy.empty(4, dtype=numpy.uint64)
    b._keys_written = 0

    b.append_keys(_u64([10, 11, 12, 13]))  # type: ignore[arg-type]
    assert b.keys_written == 4

    # 再追加 6 个 -> new_len=10，需要扩容
    b.append_keys(_u64([20, 21, 22, 23, 24, 25]))  # type: ignore[arg-type]
    assert b.keys_written == 10
    assert numpy.array_equal(
        b.extract_keys(),
        _u64([10, 11, 12, 13, 20, 21, 22, 23, 24, 25]),
    )


def test_extract_keys_returns_row_bands_matrix():
    bucket_config = BucketConfig(
        shingle_k=5,
        shingle_step=1,
        bands=2,
        rows_per_band=4,
        compute_mode="char",
        key_layout='row_bands',
    )
    b = Bucket(bucket_config=bucket_config)
    b._keys = numpy.empty(8, dtype=numpy.uint64)
    b._keys_written = 0

    b.append_keys(_u64([10, 11, 20, 21]))  # type: ignore[arg-type]

    extracted = b.extract_keys()
    assert extracted.shape == (2, 2)
    assert extracted.dtype == numpy.uint64
    assert numpy.array_equal(extracted, _u64([10, 11, 20, 21]).reshape(2, 2))


def test_extract_keys_row_bands_requires_multiple_of_bands():
    bucket_config = BucketConfig(
        shingle_k=5,
        shingle_step=1,
        bands=2,
        rows_per_band=4,
        compute_mode="char",
        key_layout='row_bands',
    )
    b = Bucket(bucket_config=bucket_config)
    b._keys = _u64([10, 11, 12])
    b._keys_written = 3

    with pytest.raises(ValueError, match='is not a multiple of bands'):
        b.extract_keys()


def test_clear_resets_state():
    bucket_config = BucketConfig(shingle_k=5, shingle_step=1, bands=4, rows_per_band=4, compute_mode="char")
    b = Bucket(bucket_config=bucket_config)
    b._keys = numpy.empty(8, dtype=numpy.uint64)
    b._keys_written = 0
    b.append_keys(_u64([7, 8, 9]))  # type: ignore[arg-type]
    assert b.keys_written == 3

    b.clear()
    assert b.keys_written == 0
    assert b.extract_keys().size == 0
