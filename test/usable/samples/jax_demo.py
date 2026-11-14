import heapq
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np


def truncate_bond(tensor, chi):
    """Keep the chi largest singular values of the bond between two sites."""
    u, s, vh = jnp.linalg.svd(tensor, full_matrices=False)
    return u[:, :chi], s[:chi], vh[:chi]


def sample_shots(probabilities, shots):
    counts = Counter()
    for _ in range(shots):
        counts[int(np.random.choice(len(probabilities), p=probabilities))] += 1
    return counts
