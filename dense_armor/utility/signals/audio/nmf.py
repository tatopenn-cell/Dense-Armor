"""Non-negative matrix factorisation with multiplicative updates, and source separation.

Source: Lee and Seung (2001), "Algorithms for Non-negative Matrix Factorization", NIPS 13,
Eq. (4) (Euclidean distance) and Eq. (5) (divergence), PDF page 3; Theorems 1 and 2 state
that the cost does not increase under these updates.
"""

import numpy as np

from dense_armor.utility.signals.audio.fourier import istft, stft
from dense_armor.utility.signals.audio.signal import Audio


def nmf(
    V: np.ndarray, rank: int, n_iter: int = 100, cost: str = "euclidean", seed: int = 0
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Factorise ``V ~ W H`` with ``W`` of shape ``(n, rank)`` and ``H`` of shape ``(rank, m)``.

    Euclidean update (Eq. 4): ``H <- H * (W^T V) / (W^T W H)`` and
    ``W <- W * (V H^T) / (W H H^T)``. Divergence update (Eq. 5):
    ``H <- H * (W^T (V / WH)) / sum_i W_ia`` and ``W <- W * ((V / WH) H^T) / sum_v H_av``.
    Choice stated: a denominator is floored at ``1e-12`` to avoid division by zero, which
    the paper does not discuss.

    Args:
        V: non-negative matrix.
        rank: number of components.
        n_iter: number of updates of ``H`` and ``W``.
        cost: ``"euclidean"`` or ``"divergence"``.
        seed: seed of the uniform initialisation in ``(0, 1)``.

    Returns:
        ``W``, ``H`` and the cost after every iteration.

    Raises:
        ValueError: if ``V`` has a negative entry or ``cost`` is unknown.

    Example:
        >>> W, H, c = nmf(np.ones((3, 4)), 1, 5)
        >>> W.shape, H.shape, len(c)
        ((3, 1), (1, 4), 5)
    """
    V = np.asarray(V, dtype=float)
    if np.any(V < 0):
        raise ValueError("V must be non-negative")
    if cost not in ("euclidean", "divergence"):
        raise ValueError(f"unknown cost {cost!r}")
    eps = 1e-12
    rng = np.random.default_rng(seed)
    W = rng.random((V.shape[0], rank)) + eps
    H = rng.random((rank, V.shape[1])) + eps
    hist = []
    for _ in range(n_iter):
        if cost == "euclidean":
            H = H * (W.T @ V) / np.maximum(W.T @ W @ H, eps)
            W = W * (V @ H.T) / np.maximum(W @ H @ H.T, eps)
            hist.append(float(np.sum((V - W @ H) ** 2)))
        else:
            R = V / np.maximum(W @ H, eps)
            H = H * (W.T @ R) / np.maximum(W.sum(axis=0)[:, None], eps)
            R = V / np.maximum(W @ H, eps)
            W = W * (R @ H.T) / np.maximum(H.sum(axis=1)[None, :], eps)
            WH = np.maximum(W @ H, eps)
            hist.append(float(np.sum(V * np.log(np.maximum(V, eps) / WH) - V + WH)))
    return W, H, hist


def separate(
    audio: Audio, rank: int, n_fft: int = 512, hop: int = 256, n_iter: int = 100
) -> list[np.ndarray]:
    """Split a mono sound into ``rank`` parts with the NMF of its magnitude spectrogram.

    Component ``j`` has the magnitude ``W[:, j] H[j, :]`` and the phase of the mixture,
    and is rebuilt with :func:`istft`. Choice stated: the phase is the phase of the mixture,
    so the parts add back to the mixture only up to the NMF error.

    Args:
        audio: one-channel sound.
        rank: number of parts.
        n_fft: frame length (Hann window, ``hop = n_fft / 2`` satisfies overlap-add).
        hop: frame step.
        n_iter: updates.

    Returns:
        A list of ``rank`` arrays, each as long as the rebuilt signal.
    """
    if audio.channels != 1:
        raise ValueError("separate expects one channel")
    S = stft(audio.x, n_fft, hop, "hann")
    W, H, _ = nmf(np.abs(S).T, rank, n_iter, "euclidean")
    phase = np.exp(1j * np.angle(S))
    return [
        istft((np.outer(W[:, j], H[j]).T) * phase, hop, "hann", n_fft)
        for j in range(rank)
    ]
