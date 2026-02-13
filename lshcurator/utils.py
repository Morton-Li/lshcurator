import hashlib
from typing import Generator, Any, Literal, Callable

import numpy
from datasketch import MinHash


def iter_char_shingles(text: str, k_gram: int, step: int = 1) -> Generator[bytes, Any, None]:
    """Character-level k-gram shingles."""
    n = len(text)
    if n < k_gram: return
    for i in range(0, n - k_gram + 1, step):
        yield text[i:i + k_gram].encode("utf-8", errors="ignore")


def iter_byte_shingles(text: str, k_gram: int, step: int = 1) -> Generator[bytes, Any, None]:
    """Byte-level k-gram shingles."""
    data = text.encode("utf-8", errors="ignore")
    n = len(data)
    if n < k_gram: return
    for i in range(0, n - k_gram + 1, step):
        yield data[i:i + k_gram]


def compute_minhash_signature(
    text: str,
    num_perm: int,
    shingle_k: int,
    shingle_step: int = 1,
    compute_mode: Literal['char', 'byte'] = 'char',
) -> MinHash:
    """Compute MinHash signature for the given text."""
    mh = MinHash(num_perm=num_perm, seed=36)
    for sh in COMPUTE_FN_MAPPING[compute_mode](text=text, k_gram=shingle_k, step=shingle_step):
        mh.update(sh)
    return mh


COMPUTE_FN_MAPPING: dict[str, Callable[..., Generator[bytes, Any, None]]] = {
    'char': iter_char_shingles,
    'byte': iter_byte_shingles,
}


def encode_band_key(hash_values: numpy.ndarray, rows_per_band: int, band_idx: int, output_type: Literal['raw', 'digest8']) -> bytes:
    """
    Band key = band_idx prefix + raw bytes of rows uint64 values.
    Prefix avoids collisions across bands.
    """
    start = band_idx * rows_per_band
    end = start + rows_per_band
    # 2-byte band prefix is enough for usual band counts
    prefix = band_idx.to_bytes(2, "little", signed=False)
    hash_values_bytes = hash_values[start:end].tobytes()
    if output_type == 'raw': return prefix + hash_values_bytes
    elif output_type == 'digest8': return hashlib.blake2b(hash_values_bytes, digest_size=8, person=prefix).digest()
    else: raise ValueError(f'Invalid output_type: {output_type}, expected "raw" or "digest"')

