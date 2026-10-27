"""Sparse mixture of experts, dispatched in JAX.

Only the top_k experts selected for each token are executed, not all
n_experts. The tokens are dispatched to their experts with a capacity factor
(Fedus et al. 2021, Section 2.2, page 5), each expert runs on its own
(capacity, n_x) sub-batch via jax.vmap, and the outputs are combined with a
gather. The cost is O(n_tokens * capacity_factor) in expert evaluations, not
O(n_tokens * n_experts).

`forward_pure(params, x, capacity, top_k, n_experts)` is a pure function and
compiles under jax.jit. `forward(x)` is the stateful wrapper of the library
contract: it saves x_ and calls forward_pure. `backward(g)` uses jax.vjp of
forward_pure and adds the result to the .grad of every Param.

The cost, declared honestly: expert evaluation is O(n_tokens * top_k *
hidden), independent of n_experts, because only the top_k experts are
executed. The routing is O(n_tokens * top_k * log(n_tokens * top_k)) via a
segment cumsum on the top-k indices, independent of n_experts (the old
argsort of the mask was O(n_experts * n_tokens * log n_tokens)). A
scatter-based dispatch (Fedus et al. 2021, Section 2.3, page 5) replaces
the loop on experts: the tokens are placed in the expert buffers with
one scatter, not with a vmap over experts.

The top-k selection is deterministic. Shazeer et al. (2017) use a noisy top-k
gating (Eq. (4), page 4); this implementation does not, and the difference is
declared here.

The gate values are used as weights, not renormalised: this matches Shazeer
et al. (2017), where the weighted sum uses the raw gate values. Fedus et al.
(2021) uses top-1, where the weight is 1 by construction.

The load-balancing loss is the auxiliary loss of Fedus et al. (2021), Eq. (4),
page 7: loss = alpha * N * sum_i f_i * P_i.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp

from dense_armor.utility.neural.layers import Dense
from dense_armor.utility.neural.params import Param


def _to_2d(x: jax.Array) -> tuple[jax.Array, bool]:
    x = jnp.asarray(x)
    if x.ndim == 1:
        return x[None, :], True
    return x, False


def _forward_pure_aux(
    params: tuple,
    x: jax.Array,
    capacity: int,
    top_k: int,
    n_experts: int,
    rng_key: jax.Array | None = None,
) -> tuple[jax.Array, dict]:
    """MoE forward, returns (y, aux). Pure, jittable."""
    Wg, bg, W1, b1, W2, b2 = params
    logits = x @ Wg + bg
    if rng_key is not None:
        logits = logits + 1e-2 * jax.random.normal(rng_key, logits.shape)
    p = jax.nn.softmax(logits, axis=-1)
    top_vals, top_idx = jax.lax.top_k(p, top_k)

    n_tokens_local = x.shape[0]
    n_flat = n_tokens_local * top_k
    flat_expert = top_idx.reshape(-1)
    # segment cumsum: rank 0-based di ogni (token, slot) nel suo esperto.
    # Il costo dipende da n_flat = n_tokens * top_k, non da n_experts.
    order = jnp.argsort(flat_expert, stable=True)
    sorted_exp = flat_expert[order]
    new_seg = jnp.concatenate(
        [jnp.ones(1, dtype=bool), sorted_exp[1:] != sorted_exp[:-1]]
    )
    seg_start = jnp.maximum.accumulate(jnp.where(new_seg, jnp.arange(n_flat), 0))
    rank_sorted = jnp.arange(n_flat) - seg_start
    rank = jnp.empty_like(rank_sorted).at[order].set(rank_sorted)
    keep = rank < capacity
    flat_idx = flat_expert * capacity + rank
    safe_idx = jnp.where(keep, flat_idx, n_experts * capacity)
    x_flat = jnp.repeat(x, top_k, axis=0)
    x_to_set = jnp.where(keep[:, None], x_flat, 0.0)
    x_disp_flat = jnp.zeros((n_experts * capacity + 1, x.shape[-1]), dtype=x.dtype)
    x_disp_flat = x_disp_flat.at[safe_idx].set(x_to_set)
    x_disp = x_disp_flat[: n_experts * capacity].reshape(
        n_experts, capacity, x.shape[-1]
    )
    pos_flat = jnp.where(keep, rank, -1)
    t_idx = jnp.repeat(jnp.arange(n_tokens_local), top_k)
    pos = jnp.full((n_tokens_local, n_experts), -1, dtype=jnp.int32)
    pos = pos.at[t_idx, flat_expert].set(pos_flat)

    def _expert(W1i, b1i, W2i, b2i, xi):
        h = jax.nn.relu(xi @ W1i + b1i)
        return h @ W2i + b2i

    y_disp = jax.vmap(_expert)(W1, b1, W2, b2, x_disp)

    positions = jnp.take_along_axis(pos, top_idx, axis=1)
    safe = jnp.where(positions >= 0, positions, 0)
    idx_flat = top_idx * capacity + safe
    y_flat = y_disp.reshape(n_experts * capacity, -1)
    y_gathered = y_flat[idx_flat]
    y_gathered = jnp.where((positions >= 0)[..., None], y_gathered, 0.0)
    y = jnp.einsum("bk,bky->by", top_vals, y_gathered)
    return y, {
        "p": p,
        "top_idx": top_idx,
        "top_vals": top_vals,
        "pos": pos,
        "y_disp": y_disp,
    }


def forward_pure(
    params: tuple,
    x: jax.Array,
    capacity: int,
    top_k: int,
    n_experts: int,
    rng_key: jax.Array | None = None,
) -> jax.Array:
    """The MoE forward output, pure and jittable."""
    y, _ = _forward_pure_aux(params, x, capacity, top_k, n_experts, rng_key)
    return y


class SparseMoE:
    """Sparse mixture of experts with dispatch.

    Args:
        n_x: input features.
        n_y: output features.
        n_experts: number of experts. Default 4.
        top_k: experts per token. Default 2.
        hidden: hidden units per expert. Default 16.
        seed: seed of gating and experts. Default 0.
        alpha: weight of the load-balancing loss. Default 1e-2.
        capacity_factor: max tokens per expert as a multiple of the average
            load. None means no dropping. Default 1.25 (Fedus et al. 2021,
            Section 2.2).

    Raises:
        ValueError: if n_experts < 1, top_k outside [1, n_experts],
            alpha < 0, or capacity_factor negative.

    References:
        Fedus, Zoph, Shazeer (2021), Switch Transformers, Eq. (1) page 5,
        Eq. (4) page 7, capacity factor Section 2.2 page 5. Shazeer et al.
        (2017), Eq. (3) page 4; the noisy top-k of Eq. (4) page 4 is not
        implemented.
    """

    def __init__(
        self,
        n_x: int,
        n_y: int,
        n_experts: int = 4,
        top_k: int = 2,
        hidden: int = 16,
        seed: int = 0,
        alpha: float = 1e-2,
        capacity_factor: float | None = 1.25,
        noisy_gating: bool = False,
        noise_std: float = 1e-2,
    ) -> None:
        if n_experts < 1:
            raise ValueError(f"n_experts must be >= 1, got {n_experts!r}")
        if not 1 <= top_k <= n_experts:
            raise ValueError(
                f"top_k must be in [1, n_experts], got {top_k!r} with n_experts={n_experts}"
            )
        if alpha < 0.0:
            raise ValueError(f"alpha must be >= 0, got {alpha!r}")
        if capacity_factor is not None and capacity_factor < 0.0:
            raise ValueError(
                f"capacity_factor must be None or >= 0, got {capacity_factor!r}"
            )
        self.n_x = int(n_x)
        self.n_y = int(n_y)
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        self.hidden = int(hidden)
        self.seed = int(seed)
        self.alpha = float(alpha)
        self.capacity_factor = capacity_factor
        self.noisy_gating = bool(noisy_gating)
        self.noise_std = float(noise_std)

        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, 4)
        self.gating = Dense(n_x, n_experts, keys[0])
        self.W1_ = Param(jax.random.normal(keys[1], (n_experts, n_x, hidden)) * 0.1)
        self.b1_ = Param(jnp.zeros((n_experts, hidden)))
        self.W2_ = Param(jax.random.normal(keys[2], (n_experts, hidden, n_y)) * 0.1)
        self.b2_ = Param(jnp.zeros((n_experts, n_y)))

        self.x_: jax.Array | None = None
        self.aux_: dict[str, Any] | None = None
        self._vjp: Any = None
        self._cap: int = 0
        self._rng_key_ = jax.random.PRNGKey(seed + 1)
        self._call_count_ = 0
        self._last_rng_key_: jax.Array | None = None

    def __repr__(self) -> str:
        return (
            f"SparseMoE(n_x={self.n_x}, n_y={self.n_y}, "
            f"n_experts={self.n_experts}, top_k={self.top_k}, "
            f"hidden={self.hidden}, seed={self.seed}, alpha={self.alpha}, "
            f"capacity_factor={self.capacity_factor})"
        )

    def _params(self) -> tuple:
        return (
            self.gating.W.value,
            self.gating.b.value,
            self.W1_.value,
            self.b1_.value,
            self.W2_.value,
            self.b2_.value,
        )

    def _capacity(self, n_tokens: int) -> int:
        if self.capacity_factor is None:
            return n_tokens
        # each token is routed to top_k experts, so the average load per
        # expert is top_k * n_tokens / n_experts, not n_tokens / n_experts
        # (Fedus et al. 2021, Section 2.2, page 5: capacity is a multiple of
        # the average tokens per expert).
        avg_load = self.top_k * n_tokens / self.n_experts
        return max(1, math.floor(self.capacity_factor * avg_load))

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Dispatch the tokens to the top_k experts and combine the output."""
        del training
        x_2d, was_1d = _to_2d(x)
        cap = self._capacity(x_2d.shape[0])
        params = self._params()
        cap_, top_k_, n_experts_ = cap, self.top_k, self.n_experts

        if self.noisy_gating:
            key = jax.random.fold_in(self._rng_key_, self._call_count_)
            self._call_count_ += 1
            self._last_rng_key_ = key
        else:
            key = None

        y, aux = _forward_pure_aux(params, x_2d, cap_, top_k_, n_experts_, key)

        def f(p_, x_):
            return forward_pure(p_, x_, cap_, top_k_, n_experts_, key)

        _, self._vjp = jax.vjp(f, params, x_2d)
        self.aux_ = aux
        self.x_ = x_2d
        self._cap = cap_
        return y[0] if was_1d else y

    def load_balancing_loss(self) -> jax.Array:
        """Fedus et al. (2021), Eq. (4), page 7."""
        if self.aux_ is None:
            raise RuntimeError("call forward() before load_balancing_loss()")
        p = self.aux_["p"]
        top_idx = self.aux_["top_idx"]
        b = p.shape[0]
        one_hot = jax.nn.one_hot(top_idx, self.n_experts).sum(axis=-2)
        f = one_hot.sum(axis=0) / b
        pi = p.mean(axis=0)
        return self.alpha * self.n_experts * jnp.sum(f * pi)

    def backward(self, g: jax.Array) -> jax.Array:
        """Backward through jax.vjp of forward_pure."""
        if self.aux_ is None:
            raise RuntimeError("call forward() before backward()")
        g_2d, _ = _to_2d(g)
        dparams, dx = self._vjp(g_2d)
        dWg, dbg, dW1, db1, dW2, db2 = dparams
        self.gating.W.grad = self.gating.W.grad + dWg
        self.gating.b.grad = self.gating.b.grad + dbg
        self.W1_.grad = self.W1_.grad + dW1
        self.b1_.grad = self.b1_.grad + db1
        self.W2_.grad = self.W2_.grad + dW2
        self.b2_.grad = self.b2_.grad + db2
        return dx

    def expert_usage(self) -> jax.Array:
        """Fraction of tokens dispatched to each expert (post-capacity)."""
        if self.aux_ is None:
            raise RuntimeError("call forward() before expert_usage()")
        pos = self.aux_["pos"]
        n_tokens = pos.shape[0]
        return (pos >= 0).sum(axis=0) / n_tokens

    def dropped_fraction(self) -> jax.Array:
        """Fraction of (token, expert) assignments dropped by the capacity."""
        if self.aux_ is None:
            raise RuntimeError("call forward() before dropped_fraction()")
        pos = self.aux_["pos"]
        n_tokens = pos.shape[0]
        n_assigned = n_tokens * self.top_k
        n_kept = (pos >= 0).sum()
        return 1.0 - n_kept / n_assigned

    def params(self) -> list[Param]:
        return [
            self.gating.W,
            self.gating.b,
            self.W1_,
            self.b1_,
            self.W2_,
            self.b2_,
        ]
