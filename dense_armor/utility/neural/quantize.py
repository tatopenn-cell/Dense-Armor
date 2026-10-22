"""The vector quantizer of van den Oord et al. 2017.

A vector ``z_e`` is mapped to the nearest code of a codebook
(Eq.(1) and Eq.(2), page 3), the gradient of the decoder is copied
to the encoder (the straight-through estimator, page 3), and the
training objective adds the codebook and commitment terms of Eq.(3),
page 4.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from dense_armor.utility.neural.params import Param


class VectorQuantizer:
    """Nearest-code quantizer with the straight-through gradient.

    ``forward(z_e, training=True)`` takes vectors on the last axis
    (shape ``(..., dim)``), picks for each vector the index of the
    nearest code in the Euclidean norm (the smallest index on a tie,
    Eq.(1), page 3 of van den Oord et al. 2017) and returns the codes
    ``codebook[index]`` (Eq.(2), page 3). It keeps ``indices_`` and
    ``z_e_`` for the backward.

    ``backward(g)`` is the straight-through estimator (page 3: the
    gradient of ``z_q`` is copied to ``z_e``) plus the gradient of the
    commitment term, and adds to ``codebook.grad`` the gradient of the
    codebook term. ``loss()`` returns the codebook term plus the
    commitment term (Eq.(3), page 4), averaged over the vectors of the
    last call; the term ``log p(x | z_q)`` belongs to the loss of the
    decoder.

    Args:
        n_codes: the number of codes, at least 1.
        dim: the dimension of each code, at least 1.
        key: a ``jax.random`` key; the codebook starts from a
            zero-mean Gaussian (our choice, the paper gives no
            initialisation).
        beta: the weight of the commitment term, default 0.25
            (page 4: results unchanged for ``beta`` from 0.1 to 2.0).

    Raises:
        ValueError: if ``n_codes < 1``, ``dim < 1`` or ``beta < 0``.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from dense_armor.utility.neural.quantize import VectorQuantizer
        >>> vq = VectorQuantizer(2, 1, jax.random.PRNGKey(0), beta=0.25)
        >>> vq.codebook.value = jnp.array([[0.0], [1.0]])
        >>> vq.forward(jnp.array([[0.2], [0.9]])).shape
        (2, 1)
        >>> vq.indices_.tolist()
        [0, 1]
    """

    def __init__(
        self,
        n_codes: int,
        dim: int,
        key: Any,
        beta: float = 0.25,
    ) -> None:
        if n_codes < 1:
            raise ValueError(f"n_codes must be at least 1, got {n_codes}")
        if dim < 1:
            raise ValueError(f"dim must be at least 1, got {dim}")
        if beta < 0:
            raise ValueError(f"beta must be non-negative, got {beta}")
        self.n_codes = n_codes
        self.dim = dim
        self.beta = float(beta)
        self.codebook = Param(jax.random.normal(key, (n_codes, dim)))
        self.indices_: jax.Array | None = None
        self.z_e_: jax.Array | None = None
        self._n_vectors: int = 0

    def __repr__(self) -> str:
        return (
            f"VectorQuantizer(n_codes={self.n_codes}, dim={self.dim}, beta={self.beta})"
        )

    def params(self) -> list[Param]:
        """The codebook, as a single ``Param``."""
        return [self.codebook]

    def _nearest(self, z_flat: jax.Array) -> jax.Array:
        """Indices of the nearest codes, smallest index on a tie."""
        d = ((z_flat[:, None, :] - self.codebook.value[None, :, :]) ** 2).sum(-1)
        return jnp.argmin(d, axis=-1)

    def forward(self, z_e: jax.Array, training: bool = True) -> jax.Array:
        """Return the nearest codes; keep the indices."""
        del training
        z_e = jnp.asarray(z_e)
        shape = z_e.shape
        z_flat = z_e.reshape(-1, self.dim)
        idx = self._nearest(z_flat)
        z_q = self.codebook.value[idx]
        self.indices_ = idx
        self.z_e_ = z_e
        self._n_vectors = z_flat.shape[0]
        return z_q.reshape(shape)

    def backward(self, g: jax.Array) -> jax.Array:
        """Straight-through gradient plus the commitment term."""
        if self.indices_ is None or self.z_e_ is None:
            raise RuntimeError("call forward() before backward()")
        g = jnp.asarray(g)
        N = self._n_vectors
        z_e_flat = self.z_e_.reshape(-1, self.dim)
        e_k = self.codebook.value[self.indices_]
        diff = z_e_flat - e_k
        dz_e = g.reshape(-1, self.dim) + (2.0 * self.beta / N) * diff
        cb_grad = jnp.zeros_like(self.codebook.value)
        cb_grad = cb_grad.at[self.indices_].add(-2.0 / N * diff)
        self.codebook.grad = self.codebook.grad + cb_grad
        return dz_e.reshape(self.z_e_.shape)

    def loss(self) -> jax.Array:
        """Codebook term plus commitment term of Eq.(3), page 4."""
        if self.indices_ is None or self.z_e_ is None:
            raise RuntimeError("call forward() before loss()")
        z_e_flat = self.z_e_.reshape(-1, self.dim)
        e_k = self.codebook.value[self.indices_]
        diff = z_e_flat - e_k
        codebook_term = jnp.mean(jnp.sum(diff**2, axis=-1))
        commit_term = self.beta * codebook_term
        return codebook_term + commit_term

    def encode_indices(self, z_e: jax.Array) -> jax.Array:
        """Indices of the nearest codes of ``z_e``."""
        z_e = jnp.asarray(z_e)
        z_flat = z_e.reshape(-1, self.dim)
        return self._nearest(z_flat).reshape(z_e.shape[:-1])

    def decode_indices(self, idx: jax.Array) -> jax.Array:
        """Codes of the indices ``idx``."""
        return self.codebook.value[jnp.asarray(idx)]
