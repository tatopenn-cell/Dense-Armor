"""Lifecycle and dual memory."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from dense_armor.utility.neural.activations import Relu
from dense_armor.utility.neural.chain import Sequential
from dense_armor.utility.neural.layers import Dense
from dense_armor.utility.neural.params import Param

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
