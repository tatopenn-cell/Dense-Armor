"""Residual tree over VQMemory."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from dense_armor.utility.neural.activations import Relu
from dense_armor.utility.neural.chain import Sequential
from dense_armor.utility.neural.layers import Dense
from dense_armor.utility.neural.params import Param

from .vq import VQMemory

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
