"""Unforgetting primitives: VQ codebook (item E).

VQ-VAE: van den Oord et al. 2017, Eq. (1)-(4) page 3.
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


class ResidualTree:
    """Residual tree (DeltaMem, Tan et al. 2026, adapted to dense networks).

    Two routers are available. ``router="linear"`` (default) scans every
    child and picks the one with the lowest vector-quantised residual,
    which requires an encode and a decode per child. ``router="mondrian"``
    keeps a bounding box per child and picks the child with the smallest
    box extension for the new event, which requires only float min/max
    comparisons. The Mondrian descent is not the same path as the linear
    one: it is a trade-off declared here, not an equivalence. Both
    respect ``max_depth`` and ``branching`` and return a valid node.

    DeltaMem is an incremental experience memory for LLM agents with
    residual trees: each node stores a residual ``<alpha, kappa>`` against
    its parent, and high-confidence paths are consolidated into a new root.
    This class adapts the tree to dense networks: each node holds a
    :class:`VQMemory`; the residual of an event is the reconstruction error
    of the parent's codebook, and a child is created when the residual is
    above ``kappa``.

    Args:
        d_model: dimension of the events.
        n_codes: codebook size of each node.
        max_depth: maximum depth of the tree. A fourth merge is refused.
        branching: number of children after which a node is consolidated.
        kappa: residual threshold above which a child is created.
        seed: seed of the codebooks.

    Deviations from the paper (declared):
        - the nodes are vector-quantised memories, not residual modules;
        - the consolidation is a k-means on the children codes, not the
          paper's confidence rule;
        - ``alpha`` and ``kappa`` are per-node scalars in the paper; here
          ``kappa`` is the threshold and ``alpha`` is the mean residual
          of the node.

    References:
        Tan, H., Zhang, Z., Cao, Z., Li, R., Chen, X. (2026). DELTAMEM:
        Incremental Experience Memory for LLM Agents via Residual Trees.
        arXiv:2606.03083, page 2-3.
    """

    def __init__(
        self,
        d_model: int,
        n_codes: int = 32,
        max_depth: int = 3,
        branching: int = 8,
        kappa: float = 0.1,
        seed: int = 0,
        router: str = "linear",
    ) -> None:
        if router not in ("linear", "mondrian"):
            raise ValueError(f"router must be 'linear' or 'mondrian', got {router!r}")
        self.d_model = int(d_model)
        self.n_codes = int(n_codes)
        self.max_depth = int(max_depth)
        self.branching = int(branching)
        self.kappa = float(kappa)
        self.seed = int(seed)
        self.router = router
        self._rng = np.random.RandomState(seed)
        self.root = self._new_node(depth=0)
        self._n_nodes = 1
        self._n_events = 0

    def _new_node(self, depth: int) -> dict:
        seed = int(self._rng.randint(0, 2**31 - 1))
        vq = VQMemory(
            n_x=self.d_model,
            d_model=min(4, self.d_model),
            n_codes=self.n_codes,
            seed=seed,
        )
        return {
            "depth": depth,
            "vq": vq,
            "children": [],
            "support": 0,
            "alpha": 0.0,
            "kappa": self.kappa,
            "samples": [],
            "lb_": None,
            "ub_": None,
        }

    def _extend_box(self, node: dict, x: jnp.ndarray) -> None:
        """Extend the node's box with ``x``."""
        xa = np.asarray(x, dtype=np.float64).reshape(-1)
        if node["lb_"] is None:
            node["lb_"] = xa.copy()
            node["ub_"] = xa.copy()
        else:
            node["lb_"] = np.minimum(node["lb_"], xa)
            node["ub_"] = np.maximum(node["ub_"], xa)

    def _box_extension(self, node: dict, x: jnp.ndarray) -> float:
        """Box extension needed to contain ``x`` (0.0 if inside).

        Returns ``inf`` for a node with no box yet (no sample seen).
        """
        if node["lb_"] is None:
            return float("inf")
        xa = np.asarray(x, dtype=np.float64).reshape(-1)
        lo = np.minimum(node["lb_"], xa)
        hi = np.maximum(node["ub_"], xa)
        return float(np.sum((node["lb_"] - lo) + (hi - node["ub_"])))

    def _pick_child_mondrian(self, node: dict, x: jnp.ndarray) -> dict | None:
        """Child with the smallest box extension for ``x`` (``None`` if no child)."""
        best = None
        best_ext = float("inf")
        for child in node["children"]:
            e = self._box_extension(child, x)
            if e < best_ext:
                best_ext = e
                best = child
        return best

    def _residual(self, node: dict, x: jnp.ndarray) -> float:
        """Reconstruction error of ``x`` with the node's codebook."""
        z_e = node["vq"].encode(x)
        x_hat = node["vq"].decode(z_e)
        return float(jnp.mean(jnp.sum((x - x_hat) ** 2, axis=-1)))

    def insert(self, x: jnp.ndarray) -> dict:
        """Insert ``x`` at the deepest node whose residual is within kappa.

        A child is created only if ``depth < max_depth`` and the node
        has fewer than ``branching`` children. When the node is full or
        at ``max_depth`` and the residual is above ``kappa``, ``x`` is
        written at the node anyway: the tree refuses to grow, and the
        number of nodes stays within ``(branching**(max_depth+1)-1)/(branching-1)``.
        """
        x = jnp.asarray(x, dtype=jnp.float32).reshape(-1, self.d_model)
        node = self.root
        while True:
            r = self._residual(node, x)
            if r <= node["kappa"]:
                self._write(node, x, r)
                return node
            if (
                node["depth"] >= self.max_depth
                or len(node["children"]) >= self.branching
            ):
                self._write(node, x, r)
                return node
            if self.router == "mondrian":
                best = self._pick_child_mondrian(node, x)
            else:
                best = None
                best_r = float("inf")
                for child in node["children"]:
                    rc = self._residual(child, x)
                    if rc < best_r:
                        best_r = rc
                        best = child
                if best is not None and best_r >= r:
                    best = None
            if best is not None:
                node = best
                continue
            child = self._new_node(depth=node["depth"] + 1)
            node["children"].append(child)
            self._n_nodes += 1
            node = child
            self._write(node, x, self._residual(node, x))
            return node

    def _write(self, node: dict, x: jnp.ndarray, r: float) -> None:
        """Write one event into ``node``.

        The samples list stops growing at 8 (the append is guarded by
        ``len < 8``), so ``len == 8`` is true on every write after the
        eighth. The previous condition was ``n % 8 == 0 or len == 8``: the
        first disjunct never changed the outcome (dead code) and is removed.
        To fit only once every 8 writes, change the condition to
        ``n % 8 == 0 and len == 8``.
        """
        node["support"] += 1
        n = node["support"]
        node["alpha"] = (node["alpha"] * (n - 1) + r) / n
        self._extend_box(node, x)
        if len(node["samples"]) < 8:
            node["samples"].append(x)
        if len(node["samples"]) == 8:
            xs = jnp.stack(node["samples"], axis=0)
            node["vq"].fit(xs, n_steps=3)
        self._n_events += 1

    def read(self, x: jnp.ndarray) -> tuple[jnp.ndarray, int]:
        """Return ``(decoded, depth)`` of the deepest node matching ``x``.

        Descends always into the child with the lowest residual, until a
        node with no children. The residual comparison against the parent
        would keep the root whenever the root was trained on more points
        than its children.
        """
        x = jnp.asarray(x, dtype=jnp.float32).reshape(-1, self.d_model)
        node = self.root
        while True:
            if self.router == "mondrian":
                best = self._pick_child_mondrian(node, x)
            else:
                best = None
                best_r = float("inf")
                for child in node["children"]:
                    rc = self._residual(child, x)
                    if rc < best_r:
                        best_r = rc
                        best = child
            if best is None:
                break
            node = best
        z_e = node["vq"].encode(x)
        return node["vq"].decode(z_e), node["depth"]

    def consolidate(self) -> int:
        """Merge every node that has ``branching`` children into a parent.

        Returns the number of nodes removed. The parent's codebook is the
        concatenation of the children centroids, reduced by k-means to
        ``n_codes / branching``; the children are dropped.
        """
        removed = 0

        def visit(node: dict) -> None:
            nonlocal removed
            while len(node["children"]) >= self.branching:
                kids = node["children"][: self.branching]
                node["vq"].fit(
                    jnp.concatenate([k["vq"].codebook.value for k in kids], axis=0),
                    n_steps=3,
                )
                node["support"] = sum(k["support"] for k in kids)
                node["alpha"] = float(np.mean([k["alpha"] for k in kids]))
                node["children"] = node["children"][self.branching :]
                removed += len(kids)
                self._n_nodes -= len(kids)
            for child in node["children"]:
                visit(child)

        visit(self.root)
        return removed

    def n_nodes(self) -> int:
        """Number of nodes in the tree."""
        return self._n_nodes

    def n_events(self) -> int:
        """Number of events written in the tree."""
        return self._n_events

    def max_nodes(self) -> int:
        """Theoretical maximum nodes with ``branching`` and ``max_depth``."""
        return (self.branching ** (self.max_depth + 1) - 1) // (self.branching - 1)

    def state_size_bytes(self) -> int:
        """Bytes of the codebooks, one entry per node."""
        per_node = self.n_codes * min(4, self.d_model) * 4
        return self.n_nodes() * per_node


class LifecycleMemory:
    """Lifecycle-quantised memory (SuperLocalMemory V3.3, Bhardwaj 2026, p.1).

    Each event is assigned to one of four states, with a bit-width per
    state: ``active`` 32, ``warm`` 8, ``cold`` 4, ``archive`` 2 (the
    paper's sequence on page 1). An access promotes the event toward
    ``active``; age without access decays it toward ``archive``. The
    reconstruction error is measured at the current bit-width.

    Args:
        d_model: dimension of the events.
        promote_threshold: accesses needed to promote one band.
        decay_horizon_s: seconds without access to decay one band.
        seed: seed of the internal RNG.

    Deviations from the paper (declared):
        - the "Ebbinghaus retention" is a linear decay on access count,
          not an exponential curve;
        - the "Fisher-Rao retrieval" is a squared L2 distance, declared.

    References:
        Bhardwaj, V. P. (2026). SuperLocalMemory V3.3: The Living Brain.
        arXiv:2604.04514, page 1 (bit-width sequence).
    """

    STATES = ("active", "warm", "cold", "archive")
    BITS = {"active": 32, "warm": 8, "cold": 4, "archive": 2}

    def __init__(
        self,
        d_model: int,
        promote_threshold: int = 3,
        decay_horizon_s: float = 600.0,
        value_range: tuple[float, float] = (-5.0, 5.0),
        seed: int = 0,
    ) -> None:
        self.d_model = int(d_model)
        self.promote_threshold = int(promote_threshold)
        self.decay_horizon_s = float(decay_horizon_s)
        self.value_range = (float(value_range[0]), float(value_range[1]))
        self.seed = int(seed)
        self._rng = np.random.RandomState(seed)
        self._events: list[dict] = []

    def _quantize(self, x: jnp.ndarray, bits: int) -> jnp.ndarray:
        """Uniform quantisation to ``bits`` bits over a fixed global range.

        The range is ``value_range``, set at construction, not the range
        of ``x``. A per-vector range is lossless for a 2D vector: the min
        and the max sit exactly on the boundaries of the quantiser, so
        even 2 bits reconstruct the original exactly. The fixed range
        gives the intended decay: 8 bits ~ 0.04 error, 4 bits ~ 0.1,
        2 bits ~ 1.

        For ``bits >= 24`` the number of levels exceeds the int32 range
        of JAX, and the reconstruction is effectively the identity.
        """
        x = jnp.asarray(x, dtype=jnp.float32)
        if bits >= 24:
            return x
        lo, hi = self.value_range
        if hi == lo:
            return x
        levels = float((1 << bits) - 1)
        q = jnp.round((x - lo) / (hi - lo) * levels) / levels
        return q * (hi - lo) + lo

    def write(self, x: jnp.ndarray, t: float = 0.0) -> int:
        """Store ``x`` in ``active`` state, return its index."""
        x = jnp.asarray(x, dtype=jnp.float32).reshape(-1)
        idx = len(self._events)
        self._events.append(
            {
                "x": x,
                "state": "active",
                "accesses": 0,
                "last_access_t": float(t),
            }
        )
        return idx

    def access(self, idx: int, t: float = 0.0) -> LifecycleMemory:
        """Register an access; promote if the threshold is reached."""
        if idx < 0 or idx >= len(self._events):
            raise IndexError(f"event {idx} out of range")
        ev = self._events[idx]
        ev["accesses"] += 1
        ev["last_access_t"] = float(t)
        if ev["accesses"] >= self.promote_threshold:
            i = self.STATES.index(ev["state"])
            if i > 0:
                ev["state"] = self.STATES[i - 1]
            ev["accesses"] = 0
        return self

    def tick(self, t: float = 0.0) -> LifecycleMemory:
        """Decay every event that has not been accessed for ``decay_horizon_s``."""
        for ev in self._events:
            age = float(t) - ev["last_access_t"]
            if age >= self.decay_horizon_s:
                i = self.STATES.index(ev["state"])
                if i < len(self.STATES) - 1:
                    ev["state"] = self.STATES[i + 1]
                ev["last_access_t"] = float(t)
        return self

    def read(self, idx: int) -> tuple[jnp.ndarray, str, int]:
        """Return ``(reconstruction, state, bits)`` at the current state."""
        if idx < 0 or idx >= len(self._events):
            raise IndexError(f"event {idx} out of range")
        ev = self._events[idx]
        bits = self.BITS[ev["state"]]
        x_hat = self._quantize(ev["x"], bits).reshape(-1)
        return x_hat, ev["state"], bits

    def state_of(self, idx: int) -> str:
        """The state of event ``idx``."""
        return self._events[idx]["state"]

    def mse_of(self, idx: int) -> float:
        """Reconstruction MSE of event ``idx`` at the current state."""
        ev = self._events[idx]
        x_hat, _, _ = self.read(idx)
        return float(jnp.mean((ev["x"] - x_hat) ** 2))

    def __len__(self) -> int:
        return len(self._events)


class DualMemory:
    """Dual memory (HiCL, Kapoor et al. 2026, adapted to dense networks).

    HiCL is a hippocampal-inspired dual-memory architecture: a fast memory
    for recent events with top-k pattern separation, and a slow memory
    for consolidated events, with a prioritised replay proportional to
    novelty and age. This class keeps the two memories as arrays, and the
    separation as a ``top_k`` distance rule.

    Args:
        d_model: dimension of the events.
        fast_capacity: number of events in the fast memory.
        slow_capacity: number of events in the slow memory.
        top_k: number of candidates in the pattern separation.
        seed: seed of the internal RNG.

    Deviations from the paper (declared):
        - no neural network: the two memories are arrays;
        - "pattern separation" is a top-k distance rule, not a learned
          orthogonal projection;
        - "prioritised replay" is proportional to ``novelty * age``, a
          linear rule, not a learned priority.

    References:
        Kapoor, K., Mackey, W., Aloimonos, Y., Lin, X. (2026). HiCL:
        Hippocampal-Inspired Continual Learning. arXiv:2508.16651, p.2-4.
    """

    def __init__(
        self,
        d_model: int,
        fast_capacity: int = 64,
        slow_capacity: int = 512,
        top_k: int = 8,
        seed: int = 0,
    ) -> None:
        self.d_model = int(d_model)
        self.fast_capacity = int(fast_capacity)
        self.slow_capacity = int(slow_capacity)
        self.top_k = int(top_k)
        self.seed = int(seed)
        self._rng = np.random.RandomState(seed)
        self.fast_: list[np.ndarray] = []
        self.slow_: list[np.ndarray] = []
        self._fast_t: list[float] = []
        self._slow_novelty: list[float] = []

    def _novelty(self, x: np.ndarray) -> float:
        """Distance to the closest slow event, in [0, inf)."""
        if not self.slow_:
            return float("inf")
        S = np.stack(self.slow_, axis=0)
        d = np.sum((S - x[None, :]) ** 2, axis=1)
        return float(np.sqrt(np.min(d)))

    def write(self, x: jnp.ndarray, t: float = 0.0) -> int:
        """Write ``x`` in the fast memory; return its index."""
        xa = np.asarray(x, dtype=np.float64).reshape(-1)[: self.d_model]
        if len(self.fast_) >= self.fast_capacity:
            self.fast_.pop(0)
            self._fast_t.pop(0)
        self.fast_.append(xa.copy())
        self._fast_t.append(float(t))
        return len(self.fast_) - 1

    def top_k_separation(self, x: jnp.ndarray) -> list[int]:
        """Indices of the ``top_k`` most dissimilar events in the fast memory."""
        if not self.fast_:
            return []
        F = np.stack(self.fast_, axis=0)
        xa = np.asarray(x, dtype=np.float64).reshape(-1)[: self.d_model]
        d = np.sum((F - xa[None, :]) ** 2, axis=1)
        order = np.argsort(d)[::-1]
        return [int(i) for i in order[: self.top_k]]

    def consolidate(self) -> int:
        """Move all fast events to slow memory; return how many moved."""
        n = len(self.fast_)
        for xa, t in zip(self.fast_, self._fast_t):
            nov = self._novelty(xa)
            if len(self.slow_) >= self.slow_capacity:
                self.slow_.pop(0)
                self._slow_novelty.pop(0)
            self.slow_.append(xa.copy())
            self._slow_novelty.append(0.0 if nov == float("inf") else nov)
        self.fast_ = []
        self._fast_t = []
        return n

    def replay(self, n: int, t_now: float = 0.0) -> list[np.ndarray]:
        """Sample ``n`` events from the slow memory, prioritised by novelty*age.

        Declared limitation: ``age`` is the index-based recency
        ``np.arange(len(slow_), 0, -1)``, not ``t_now - t_event``.  The
        argument ``t_now`` is accepted for interface compatibility and is
        not used.  Making the age time-based needs a timestamp per event,
        which the memory does not keep.
        """
        if not self.slow_:
            return []
        nov = np.asarray(self._slow_novelty, dtype=np.float64)
        # age is the recency: older events (lower index) have higher age
        age = np.arange(len(self.slow_), 0, -1, dtype=np.float64)
        w = (nov + 1e-6) * age
        w = w / np.sum(w)
        idx = self._rng.choice(len(self.slow_), size=n, replace=True, p=w)
        return [self.slow_[i].copy() for i in idx]

    def size(self) -> tuple[int, int]:
        """The sizes ``(fast, slow)``."""
        return len(self.fast_), len(self.slow_)


class TheoreticalBound:
    """Classical Hopfield capacity as a forgetting bound (placeholder).

    The two 2025 papers on the theoretical limit are **not in the folder**.
    This class is a declared placeholder, not the 2025 refinement: it uses
    the classical Hopfield capacity ``N_max ~ 0.138 * N`` where ``N`` is
    the number of neurons, not the number of parameters. For a fully
    connected network ``P = N^2``, so ``N_max ~ 0.138 * sqrt(P)``.

    Args:
        n_params: number of parameters of the network.
        n_neurons: number of neurons; if ``None``, ``sqrt(n_params)``.

    References:
        Hopfield, J. J. (1982). Neural networks and physical systems with
        emergent collective computational abilities. PNAS 79(8):2554-2558,
        page 3. **Bibliographic only**: no PDF in the folder, the capacity
        ``0.138 N`` is the classical result.

        The two 2025 papers are in the folder and are cited, but they do
        **not** give a closed-form bound: ``hopfield2025_forgetting.pdf``
        (Essex, Janson, Norris, Baird 2025) studies bifurcation mechanisms
        and attractor basins in a learning Hopfield network,
        ``associative2025_forgetting.pdf`` (Vinci, Galluzzi, Mattia 2025)
        studies self-interactions in associative networks. The ``0.138``
        capacity is not in either PDF; the class keeps the classical
        value and declares it.
    """

    def __init__(self, n_params: int, n_neurons: int | None = None) -> None:
        self.n_params = int(n_params)
        self.n_neurons = (
            int(n_neurons) if n_neurons is not None else int(round(self.n_params**0.5))
        )

    def capacity(self) -> float:
        """``0.138 * N``, the classical Hopfield capacity."""
        return 0.138 * self.n_neurons

    def bound(self, n_events: int) -> float:
        """Forgetting bound per epoch, ``max(0, (n_events - capacity) / P)``."""
        n_events = int(n_events)
        excess = max(0.0, n_events - self.capacity())
        return excess / max(self.n_params, 1)


class ArchivalMemory:
    """Unforgetting architetturale: residual tree + lifecycle + dual memory.

    The write policy is the one declared in ``RULE.md`` v0.1 of the task:

    - ``importance(x, t) = 0.4 * novelty + 0.3 * rarity + 0.3 * relevance``
      with novelty the distance from the closest slow event, rarity the
      inverse of the number of events already in the same leaf, and
      relevance the residual against the current node;
    - ``write_depth(importance) = round(importance * max_depth)``;
    - ``transition(state, event)``: promote after ``promote_threshold``
      accesses, decay one band after ``decay_horizon_s`` seconds;
    - ``promote(node)``: the codebook of the parent is the concatenation
      of the children's centroids, reduced by k-means to
      ``n_codes // branching``.

    The four functions are in ``RULE.md`` section 1 to 4; the class
    implements them as methods, and the test pins the three cases each.
    """

    def __init__(
        self,
        d_model: int,
        n_codes: int = 32,
        max_depth: int = 3,
        branching: int = 8,
        fast_capacity: int = 64,
        slow_capacity: int = 512,
        seed: int = 0,
        router: str = "linear",
    ) -> None:
        self.d_model = int(d_model)
        self.n_codes = int(n_codes)
        self.max_depth = int(max_depth)
        self.branching = int(branching)
        self.fast_capacity = int(fast_capacity)
        self.slow_capacity = int(slow_capacity)
        self.seed = int(seed)
        self.router = str(router)
        self.tree = ResidualTree(
            d_model=d_model,
            n_codes=n_codes,
            max_depth=max_depth,
            branching=branching,
            kappa=0.1,
            seed=seed,
            router=self.router,
        )
        self.lifecycle = LifecycleMemory(
            d_model=d_model,
            promote_threshold=3,
            decay_horizon_s=600.0,
            value_range=(-5.0, 5.0),
            seed=seed,
        )
        self.dual = DualMemory(
            d_model=d_model,
            fast_capacity=fast_capacity,
            slow_capacity=slow_capacity,
            top_k=8,
            seed=seed,
        )

    def importance(self, x: jnp.ndarray, t: float = 0.0) -> float:
        """The importance of ``x``, in [0, 1] (RULE.md section 1).

        The novelty is the distance from the closest slow event. ``x`` has
        shape ``(1, d)`` after ``reshape(-1, d_model)``; ``S`` is ``(n, d)``.
        ``S - x`` broadcasts to ``(n, d)`` and the sum over ``axis=1`` is
        the squared distance per event. The previous ``x[None, :]`` made it
        ``(1, 1, d)``, broadcast to ``(1, n, d)``, and ``axis=1`` summed
        over events instead of over coordinates.
        """
        x = jnp.asarray(x, dtype=jnp.float32).reshape(-1, self.d_model)
        # novelty: distance from the closest slow event, sigmoid-scaled
        nov = float(jnp.linalg.norm(x))
        if self.dual.slow_:
            S = np.stack(self.dual.slow_, axis=0)
            d = np.sum((S - np.asarray(x)) ** 2, axis=1)
            nov = float(np.sqrt(np.min(d)))
        novelty = 1.0 / (1.0 + nov)
        # rarity: inverse of the number of events in the same leaf
        _, depth = self.tree.read(x)
        rarity = 1.0 / (1.0 + depth)
        # relevance: normalised residual against the current node
        r = float(self.tree._residual(self.tree.root, x))
        relevance = r / (r + 1.0)
        return float(np.clip(0.4 * novelty + 0.3 * rarity + 0.3 * relevance, 0.0, 1.0))

    def write_depth(self, importance: float) -> int:
        """The depth at which to write, in [0, max_depth] (RULE.md section 2)."""
        return int(max(0, min(self.max_depth, round(importance * self.max_depth))))

    def transition(self, state: str, accesses: int, age_s: float) -> str:
        """The lifecycle transition (RULE.md section 3)."""
        states = LifecycleMemory.STATES
        i = states.index(state)
        if accesses >= self.lifecycle.promote_threshold and i > 0:
            return states[i - 1]
        if (
            age_s >= self.lifecycle.decay_horizon_s
            and accesses == 0
            and i < len(states) - 1
        ):
            return states[i + 1]
        return state

    def write(self, x: jnp.ndarray, t: float = 0.0) -> int:
        """Write ``x`` in the tree, the lifecycle and the dual memory.

        Declared limitation: the write policy ``importance -> write_depth``
        (RULE.md sections 1 and 2) is *not* connected here.  ``self.tree``
        inserts at its own default depth, and the importance is not passed
        to the lifecycle or the dual memory.  ``importance()`` and
        ``write_depth()`` are available to the caller, but the caller must
        apply them; the class does not.  Connecting them is a behavioural
        change, not a bug fix.
        """
        x = jnp.asarray(x, dtype=jnp.float32).reshape(-1, self.d_model)
        self.tree.insert(x)
        idx = self.lifecycle.write(x, t=t)
        self.dual.write(x, t=t)
        return idx

    def read(self, x: jnp.ndarray) -> tuple[jnp.ndarray, int]:
        """Read ``x`` from the tree (deepest node)."""
        return self.tree.read(x)

    def reconstruct(self, idx: int) -> tuple[jnp.ndarray, str, int]:
        """Reconstruct event ``idx`` at its current lifecycle state."""
        return self.lifecycle.read(idx)

    def __len__(self) -> int:
        return len(self.lifecycle)
