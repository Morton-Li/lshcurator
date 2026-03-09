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
import hashlib
from typing import Literal, Callable, Iterator

import numpy
from datasketch import MinHash

from .utils.types import ComputeMode


def iter_char_shingles(text: str, *, k_gram: int, step: int = 1) -> Iterator[bytes]:
    """Character-level k-gram shingles."""
    n = len(text)
    if n < k_gram: return
    for i in range(0, n - k_gram + 1, step):
        yield text[i:i + k_gram].encode("utf-8", errors="ignore")


def iter_byte_shingles(text: str, *, k_gram: int, step: int = 1) -> Iterator[bytes]:
    """Byte-level k-gram shingles."""
    data = text.encode("utf-8", errors="ignore")
    n = len(data)
    if n < k_gram: return
    for i in range(0, n - k_gram + 1, step):
        yield data[i:i + k_gram]


COMPUTE_FN_MAPPING: dict[ComputeMode, Callable[..., Iterator[bytes]]] = {
    'char': iter_char_shingles,
    'byte': iter_byte_shingles,
}


def compute_minhash_signature(
    text: str,
    *,
    num_perm: int,
    shingle_k: int,
    shingle_step: int = 1,
    compute_mode: ComputeMode = 'char',
) -> MinHash:
    """Compute MinHash signature for the given text."""
    mh = MinHash(num_perm=num_perm, seed=36)
    for sh in COMPUTE_FN_MAPPING[compute_mode](text=text, k_gram=shingle_k, step=shingle_step):
        mh.update(sh)
    return mh


def encode_band_key(
    hash_values: numpy.ndarray,
    *,
    rows_per_band: int,
    band_idx: int,
    output_type: Literal['raw', 'digest8']
) -> bytes:
    """
    Encode the band key for the given band index.
    """
    start = band_idx * rows_per_band
    end = start + rows_per_band
    # 2-byte band prefix is enough for usual band counts
    if band_idx >= 2**16: raise ValueError(f'band_idx {band_idx} exceeds maximum of 65535 for 2-byte prefix')
    prefix = band_idx.to_bytes(2, "little", signed=False)
    hash_values_bytes = hash_values[start:end].astype('<u8', copy=False).tobytes()
    if output_type == 'raw': return prefix + hash_values_bytes
    elif output_type == 'digest8': return hashlib.blake2b(hash_values_bytes, digest_size=8, person=prefix).digest()
    else: raise ValueError(f'Invalid output_type: {output_type}, expected "raw" or "digest8"')


def compute_band_keys(
    hash_values: numpy.ndarray[numpy.uint64],
    *,
    bands: int,
    rows_per_band: int,
) -> numpy.ndarray[numpy.uint64]:
    """
    Compute band keys for the given hash values.
    Args:
        hash_values (numpy.ndarray): 1D array of uint64 hash values (MinHash signature).
        bands (int): Number of bands.
        rows_per_band (int): Number of rows per band.
    """
    if hash_values.dtype != numpy.uint64:
        hash_values = hash_values.astype(numpy.uint64, copy=False)

    expected = bands * rows_per_band
    if hash_values.ndim != 1 or hash_values.size != expected:
        raise ValueError(f'hash_values must be 1D of length {expected}, got shape={hash_values.shape}')

    keys = numpy.empty(bands, dtype=numpy.uint64)
    for band_idx in range(bands):
        digest8_key = encode_band_key(
            hash_values=hash_values,
            rows_per_band=rows_per_band,
            band_idx=band_idx,
            output_type='digest8',
        )
        keys[band_idx] = numpy.frombuffer(digest8_key, dtype='<u8', count=1)[0]
    return keys
