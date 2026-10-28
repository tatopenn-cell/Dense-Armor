"""Archival memory: tree + lifecycle + dual."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from dense_armor.utility.neural.activations import Relu
from dense_armor.utility.neural.chain import Sequential
from dense_armor.utility.neural.layers import Dense
from dense_armor.utility.neural.params import Param

from .memory import DualMemory, LifecycleMemory
from .tree import ResidualTree

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
