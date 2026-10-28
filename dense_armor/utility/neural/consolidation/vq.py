"""Vector quantisation memories: VQMemory, PQMemory.

VQMemory: VQ-VAE, van den Oord et al. 2017, Eq. (1)-(4) page 3.
PQMemory: product quantisation, Jegou et al. 2011, IEEE TPAMI 33(1).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from dense_armor.utility.neural.activations import Relu
from dense_armor.utility.neural.chain import Sequential
from dense_armor.utility.neural.layers import Dense
from dense_armor.utility.neural.params import Param


class VQMemory:
    """Vector-quantised memory (van den Oord et al. 2017, Eq. 1-4 p. 3).

    Args:
        n_x: number of input features.
        d_model: dimension of the latent code.
        n_codes: size of the codebook.
        hidden: hidden units of the encoder and the decoder.
        commitment: weight of the commitment term.
        seed: seed of the encoder, decoder and codebook.

    References:
        van den Oord, Vinyals, Kavukcuoglu 2017, Neural Discrete
        Representation Learning, NeurIPS, Eq. (1)-(4) page 3.
    """

    def __init__(
        self,
        n_x: int,
        d_model: int = 8,
        n_codes: int = 32,
        hidden: int = 16,
        commitment: float = 0.25,
        seed: int = 0,
    ) -> None:
        self.n_x = int(n_x)
        self.d_model = int(d_model)
        self.n_codes = int(n_codes)
        self.hidden = int(hidden)
        self.commitment = float(commitment)
        self.seed = int(seed)
        key = jax.random.PRNGKey(seed)
        self.encoder = Sequential(
            [Dense(n_x, hidden, key), Relu(), Dense(hidden, d_model, key)]
        )
        self.decoder = Sequential(
            [Dense(d_model, hidden, key), Relu(), Dense(hidden, n_x, key)]
        )
        codebook_init = jax.random.normal(key, (n_codes, d_model)) * 0.1
        self.codebook = Param(jnp.asarray(codebook_init))
        self._z_e: jnp.ndarray | None = None
        self._z_q: jnp.ndarray | None = None
        self._k: jnp.ndarray | None = None
        self._counts = jnp.zeros(n_codes)
        self._sums = jnp.zeros((n_codes, d_model))

    def encode(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return ``z_e = encoder(x)``, shape ``(n, d_model)``."""
        return self.encoder.forward(x, training=False)

    def quantize(
        self, z_e: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return ``(z_q, k, e_k)`` with ``k = argmin_j ||z_e - e_j||^2``."""
        diff = z_e[:, None, :] - self.codebook.value[None, :, :]
        dist = jnp.sum(diff * diff, axis=-1)
        k = jnp.argmin(dist, axis=-1)
        e_k = self.codebook.value[k]
        return e_k, k, self.codebook.value

    def decode(self, z_q: jnp.ndarray) -> jnp.ndarray:
        """Return ``decoder(z_q)``, shape ``(n, n_x)``."""
        return self.decoder.forward(z_q, training=False)

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """Encoder, quantiser, decoder; caches the intermediate arrays."""
        x = jnp.asarray(x)
        z_e = self.encode(x)
        z_q, k, _ = self.quantize(z_e)
        x_hat = self.decode(z_q)
        self._z_e = z_e
        self._z_q = z_q
        self._k = k
        self._counts = self._counts.at[k].add(1.0)
        self._sums = self._sums.at[k].add(z_e)
        return x_hat

    def backward(self, g: jnp.ndarray) -> jnp.ndarray:
        """Backward through the decoder, straight-through, and the encoder.

        Returns the gradient on ``x``. Adds to ``self.codebook.grad`` the
        codebook term ``-2 (z_e - e_k)`` and to the encoder gradient the
        commitment term ``2 beta (z_e - e_k)`` (straight-through).
        """
        if self._z_e is None or self._z_q is None or self._k is None:
            raise RuntimeError("call forward() before backward()")
        dz_q = self.decoder.backward(g)
        z_e = self._z_e
        e_k = self.codebook.value[self._k]
        diff = z_e - e_k
        dz_e = dz_q + 2.0 * self.commitment * diff
        self.codebook.grad = self.codebook.grad + jnp.zeros_like(self.codebook.grad).at[
            self._k
        ].add(-2.0 * diff)
        dx = self.encoder.backward(dz_e)
        return dx

    def fit(self, x: jnp.ndarray, n_steps: int = 5, lr: float = 0.01) -> VQMemory:
        """Train the encoder, decoder and codebook on ``x`` for ``n_steps``.

        Helper for the tests and the ``ResidualTree``: uses the EMA
        codebook update and the straight-through gradient on the encoder
        and decoder. One ``Adam`` per ``Param`` (the shapes differ, and a
        shared optimiser fails on the first parameter whose shape does not
        match its state).
        """
        from dense_armor.utility.optim import Adam

        X = jnp.asarray(x, dtype=jnp.float32)
        if not hasattr(self, "_opt_enc"):
            self._opt_enc = [Adam(lr=lr) for _ in self.encoder.params()]
            self._opt_dec = [Adam(lr=lr) for _ in self.decoder.params()]
        else:
            for o in self._opt_enc + self._opt_dec:
                if hasattr(o, "lr"):
                    o.lr = lr
        for _ in range(int(n_steps)):
            x_hat = self.forward(X)
            self.backward(x_hat - X)
            self.update_codebook()
            for p, o in zip(self.encoder.params(), self._opt_enc):
                p.update(o)
                p.zero_grad()
            for p, o in zip(self.decoder.params(), self._opt_dec):
                p.update(o)
                p.zero_grad()
        return self

    def update_codebook(self) -> VQMemory:
        """EMA update: ``e_k = sums_k / counts_k`` for the used codes."""
        mask = self._counts > 0
        means = jnp.where(
            mask[:, None],
            self._sums / jnp.maximum(self._counts[:, None], 1.0),
            self.codebook.value,
        )
        self.codebook.value = means
        return self

    def params(self) -> list[Param]:
        """The encoder and decoder parameters (the codebook is updated by EMA)."""
        return self.encoder.params() + self.decoder.params()


class PQMemory:
    """Product quantisation (Jégou, Douze, Schmid 2011, IEEE TPAMI 33(1)).

    Bibliographic only: the paper is not on arXiv, the formula is derived
    from section 3 and declared below. The deviations are declared in
    ``Args``.

    Each ``d_model``-dimensional vector is split into ``n_subspaces``
    disjoint subspaces of dimension ``d_model // n_subspaces``. Each
    subspace has its own codebook of ``n_centroids`` centroids, trained
    by k-means. A vector is encoded as one centroid index per subspace,
    and the asymmetric distance ``ADC(q, code) = sum_s ||q_s - c_{s,i_s}||^2``
    is the squared reconstruction error, computed without decoding.

    Args:
        d_model: dimension of the vector, must be divisible by ``n_subspaces``.
        n_subspaces: number of disjoint subspaces.
        n_centroids: codebook size of one subspace (``256`` in the paper).
        n_iter: k-means iterations.
        seed: seed of the k-means initialisation.

    Raises:
        ValueError: if ``d_model`` is not divisible by ``n_subspaces``.

    References:
        Jégou, H., Douze, M., Schmid, C. (2011). Product Quantization for
        Nearest Neighbor Search. IEEE TPAMI 33(1):117-128,
        DOI 10.1109/TPAMI.2011.144. Bibliographic only: not in the folder,
        formula derived from section 3.
    """

    def __init__(
        self,
        d_model: int,
        n_subspaces: int = 4,
        n_centroids: int = 16,
        n_iter: int = 20,
        seed: int = 0,
    ) -> None:
        if d_model % n_subspaces != 0:
            raise ValueError(
                f"d_model {d_model} not divisible by n_subspaces {n_subspaces}"
            )
        self.d_model = int(d_model)
        self.n_subspaces = int(n_subspaces)
        self.n_centroids = int(n_centroids)
        self.n_iter = int(n_iter)
        self.seed = int(seed)
        self.d_sub = d_model // n_subspaces
        self.centroids_: list[jnp.ndarray] = []
        self._fitted = False

    def _kmeans(self, X: jnp.ndarray, rng: np.random.RandomState) -> jnp.ndarray:
        n, d = X.shape
        k = min(self.n_centroids, n)
        idx = rng.choice(n, size=k, replace=False)
        C = jnp.asarray(np.asarray(X)[idx], dtype=jnp.float32)
        for _ in range(self.n_iter):
            diff = X[:, None, :] - C[None, :, :]
            dist = jnp.sum(diff * diff, axis=-1)
            assign = jnp.argmin(dist, axis=-1)
            newC = []
            for j in range(k):
                mask = assign == j
                if bool(jnp.any(mask)):
                    newC.append(jnp.mean(X[mask], axis=0))
                else:
                    newC.append(C[j])
            C = jnp.stack(newC, axis=0)
        return C

    def fit(self, documents: jnp.ndarray) -> PQMemory:
        """Train one codebook per subspace by k-means."""
        X = jnp.asarray(documents, dtype=jnp.float32)
        if X.shape[-1] != self.d_model:
            raise ValueError(f"expected d_model={self.d_model}, got {X.shape[-1]}")
        rng = np.random.RandomState(self.seed)
        self.centroids_ = []
        for s in range(self.n_subspaces):
            a = s * self.d_sub
            b = (s + 1) * self.d_sub
            C = self._kmeans(X[:, a:b], rng)
            self.centroids_.append(C)
        self._fitted = True
        return self

    def encode(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return ``(n_subspaces,)`` or ``(n, n_subspaces)`` indices."""
        if not self._fitted:
            raise RuntimeError("call fit() before encode()")
        single = len(x.shape) == 1
        X = jnp.asarray(x, dtype=jnp.float32).reshape(-1, self.d_model)
        codes = []
        for s in range(self.n_subspaces):
            a = s * self.d_sub
            b = (s + 1) * self.d_sub
            xa = X[:, a:b]
            diff = xa[:, None, :] - self.centroids_[s][None, :, :]
            dist = jnp.sum(diff * diff, axis=-1)
            codes.append(jnp.argmin(dist, axis=-1))
        out = jnp.stack(codes, axis=-1)
        return out[0] if single else out

    def decode(self, code: jnp.ndarray) -> jnp.ndarray:
        """Reconstruct the concatenation of the centroids."""
        if not self._fitted:
            raise RuntimeError("call fit() before decode()")
        single = len(code.shape) == 1
        C = jnp.asarray(code).reshape(-1, self.n_subspaces)
        parts = []
        for s in range(self.n_subspaces):
            parts.append(self.centroids_[s][C[:, s]])
        out = jnp.concatenate(parts, axis=-1)
        return out[0] if single else out

    def asymmetric_distance(self, q: jnp.ndarray, code: jnp.ndarray) -> jnp.ndarray:
        """``ADC(q, code) = ||q - decode(code)||^2``, computed in subspaces."""
        if not self._fitted:
            raise RuntimeError("call fit() before asymmetric_distance()")
        q = jnp.asarray(q, dtype=jnp.float32).reshape(-1, self.d_model)
        code = jnp.asarray(code).reshape(-1, self.n_subspaces)
        total = jnp.zeros(q.shape[0])
        for s in range(self.n_subspaces):
            a = s * self.d_sub
            b = (s + 1) * self.d_sub
            qs = q[:, a:b]
            cs = self.centroids_[s][code[:, s]]
            d = qs - cs
            total = total + jnp.sum(d * d, axis=-1)
        return total
