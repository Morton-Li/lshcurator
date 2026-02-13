from dataclasses import dataclass

import numpy


@dataclass(slots=True)
class BucketState:
    representatives: list[numpy.ndarray]  # representative hash_values arrays (uint64, length=num_perm)
    hit_count: int = 0  # number of times this bucket has been hit
