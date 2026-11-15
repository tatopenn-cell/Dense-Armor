"""Vector Symbolic Algebras: HRR and Spatial Semantic Pointers.

References:
    Joffe & Eliasmith (2025), "Vector Symbolic Algebras for the
    Abstraction and Reasoning Corpus", arXiv:2511.08747.
"""

from __future__ import annotations

import numpy as np

N_DIMENSIONS = 4096
LENGTH_SCALE = 0.25
MAX_GRID_SIZE = 30
SEED = 0


def similarity(a, b):
    return float(np.dot(a, b))


def bind(a, b):
    c = np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=len(a))
    n = np.linalg.norm(c)
    return c / n if n > 0 else c


def unbind(a, b):
    c = np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(b)), n=len(a))
    n = np.linalg.norm(c)
    return c / n if n > 0 else c


def invert(a):
    return np.roll(a[::-1], 1)


def bundle(*vectors):
    s = np.sum(vectors, axis=0)
    n = np.linalg.norm(s)
    return s / n if n > 0 else s


def random_unit_vector(rng, n=N_DIMENSIONS):
    v = rng.normal(size=n)
    return v / np.linalg.norm(v)


class SSPSpace:
    def __init__(self, n=N_DIMENSIONS, length_scale=LENGTH_SCALE, seed=SEED):
        self.n = n
        self.length_scale = np.full(2, float(length_scale))
        rng = np.random.default_rng(seed)
        self.phase = rng.uniform(-np.pi, np.pi, size=(2, n))

    def encode(self, positions):
        p = np.atleast_2d(np.asarray(positions, dtype=float))
        angles = p @ self.phase
        angles = angles / self.length_scale.mean()
        sp = np.fft.irfft(np.exp(1j * angles), n=self.n, axis=-1)
        norms = np.linalg.norm(sp, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        sp = sp / norms
        if np.ndim(positions) == 1:
            return sp[0]
        return sp
