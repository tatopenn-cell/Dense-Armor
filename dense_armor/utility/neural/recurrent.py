"""Recurrent layers: RNN, GRU, LSTM, with backpropagation through time.

Each layer maps a sequence ``(T, n_in)`` or ``(B, T, n_in)`` to the
sequence of the outputs of the step, ``(T, n_hidden)`` or
``(B, T, n_hidden)``. ``step(x_t, state)`` returns one sample and the
next state, and ``forward`` from a zero state gives the same numbers as
``step`` called T times.

GRU: Eqs.(5) to (8), page 3 of Cho et al. 2014. LSTM: the vanilla block
of Greff et al. 2017, pages 1 and 2, forward and backpropagation through
time, with the peephole weights optionally removed.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from dense_armor.utility.neural.init import xavier_normal, zeros_bias
from dense_armor.utility.neural.params import Param


def _split_input(seq: jax.Array) -> tuple[jax.Array, bool]:
    """Return ``(B, T, n_in)`` and a flag that says if the batch was added."""
    seq = jnp.asarray(seq)
    squeeze = seq.ndim == 2
    if squeeze:
        seq = seq[None]
    return seq, squeeze


def _sigmoid(x: jax.Array) -> jax.Array:
    return jax.nn.sigmoid(x)


class RNN:
    """Elman recurrent layer ``h = tanh(W x + U h_prev + b)``.

    The activation can be ``"tanh"`` or ``"relu"``; anything else raises
    ``ValueError``. The sources hold no paper for this layer, so the
    definition is textbook.

    Args:
        n_in: input features.
        n_hidden: hidden state size.
        key: a ``jax.random`` key.
        activation: ``"tanh"`` (default) or ``"relu"``.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from dense_armor.utility.neural.recurrent import RNN
        >>> r = RNN(2, 3, jax.random.PRNGKey(0))
        >>> r.forward(jnp.ones((4, 2))).shape
        (4, 3)
    """

    def __init__(
        self,
        n_in: int,
        n_hidden: int,
        key: Any,
        activation: str = "tanh",
    ) -> None:
        if activation not in ("tanh", "relu"):
            raise ValueError(f"activation must be 'tanh' or 'relu', got {activation!r}")
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.activation = activation
        k1, k2 = jax.random.split(key)
        self.W = Param(xavier_normal(k1, (n_in, n_hidden)))
        self.U = Param(xavier_normal(k2, (n_hidden, n_hidden)))
        self.b = Param(zeros_bias(n_hidden))
        self._cache: list[tuple[jax.Array, jax.Array, jax.Array]] = []

    def __repr__(self) -> str:
        return f"RNN({self.n_in}, {self.n_hidden}, activation={self.activation!r})"

    def params(self) -> list[Param]:
        """The three parameters, ``W``, ``U``, ``b``."""
        return [self.W, self.U, self.b]

    def _act(self, x: jax.Array) -> jax.Array:
        return jnp.tanh(x) if self.activation == "tanh" else jax.nn.relu(x)

    def init_state(self, batch: int | None = None) -> jax.Array:
        """Zeros of shape ``(n_hidden,)`` or ``(batch, n_hidden)``."""
        if batch is None:
            return jnp.zeros(self.n_hidden)
        return jnp.zeros((batch, self.n_hidden))

    def step(self, x_t: jax.Array, state: jax.Array) -> tuple[jax.Array, jax.Array]:
        """One step: ``(y_t, h_t)`` from ``(x_t, h_prev)``."""
        x_t = jnp.asarray(x_t)
        h = self._act(x_t @ self.W.value + state @ self.U.value + self.b.value)
        return h, h

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Run the layer over a sequence, from a zero state."""
        del training
        seq, squeeze = _split_input(x)
        B, T, _ = seq.shape
        h = jnp.zeros((B, self.n_hidden), dtype=seq.dtype)
        ys = []
        self._cache = []
        for t in range(T):
            x_t = seq[:, t, :]
            h_prev = h
            h = self._act(x_t @ self.W.value + h_prev @ self.U.value + self.b.value)
            ys.append(h)
            self._cache.append((x_t, h_prev, h))
        y = jnp.stack(ys, axis=1)
        return y[0] if squeeze else y

    def backward(self, g: jax.Array) -> jax.Array:
        """Backpropagation through time; returns ``dx`` and adds to grads."""
        if not self._cache:
            raise RuntimeError("call forward() before backward()")
        g = jnp.asarray(g)
        squeeze = g.ndim == 2
        if squeeze:
            g = g[None]
        B, T, _ = g.shape
        dW = jnp.zeros_like(self.W.value)
        dU = jnp.zeros_like(self.U.value)
        db = jnp.zeros_like(self.b.value)
        dh_next = jnp.zeros((B, self.n_hidden), dtype=g.dtype)
        dxs = [None] * T
        for t in range(T - 1, -1, -1):
            x_t, h_prev, h_t = self._cache[t]
            dh = g[:, t, :] + dh_next
            if self.activation == "tanh":
                da = dh * (1.0 - h_t**2)
            else:
                da = dh * (h_t > 0).astype(dh.dtype)
            dW = dW + x_t.T @ da
            dU = dU + h_prev.T @ da
            db = db + da.sum(axis=0)
            dh_next = da @ self.U.value.T
            dxs[t] = da @ self.W.value.T
        self.W.grad = self.W.grad + dW
        self.U.grad = self.U.grad + dU
        self.b.grad = self.b.grad + db
        dx = jnp.stack(dxs, axis=1)
        return dx[0] if squeeze else dx


class GRU:
    """GRU layer, Eqs.(5) to (8), page 3 of Cho et al. 2014.

    ``r = sigmoid(W_r x + U_r h_prev)``; ``z = sigmoid(W_z x + U_z h_prev)``;
    ``h_tilde = tanh(W x + U (r * h_prev))``; ``h = z * h_prev + (1 - z) * h_tilde``.
    The paper writes no bias; a bias vector added to each of the three
    pre-activations is our choice, starting at zeros.

    Args:
        n_in: input features.
        n_hidden: hidden state size.
        key: a ``jax.random`` key.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from dense_armor.utility.neural.recurrent import GRU
        >>> r = GRU(2, 3, jax.random.PRNGKey(0))
        >>> r.forward(jnp.ones((4, 2))).shape
        (4, 3)
    """

    def __init__(self, n_in: int, n_hidden: int, key: Any) -> None:
        self.n_in = n_in
        self.n_hidden = n_hidden
        keys = jax.random.split(key, 9)
        self.W_r = Param(xavier_normal(keys[0], (n_in, n_hidden)))
        self.W_z = Param(xavier_normal(keys[1], (n_in, n_hidden)))
        self.W = Param(xavier_normal(keys[2], (n_in, n_hidden)))
        self.U_r = Param(xavier_normal(keys[3], (n_hidden, n_hidden)))
        self.U_z = Param(xavier_normal(keys[4], (n_hidden, n_hidden)))
        self.U = Param(xavier_normal(keys[5], (n_hidden, n_hidden)))
        self.b_r = Param(zeros_bias(n_hidden))
        self.b_z = Param(zeros_bias(n_hidden))
        self.b = Param(zeros_bias(n_hidden))
        self._cache: list[dict[str, jax.Array]] = []

    def __repr__(self) -> str:
        return f"GRU({self.n_in}, {self.n_hidden})"

    def params(self) -> list[Param]:
        """The nine parameters, in a fixed order."""
        return [
            self.W_r,
            self.W_z,
            self.W,
            self.U_r,
            self.U_z,
            self.U,
            self.b_r,
            self.b_z,
            self.b,
        ]

    def init_state(self, batch: int | None = None) -> jax.Array:
        """Zeros of shape ``(n_hidden,)`` or ``(batch, n_hidden)``."""
        if batch is None:
            return jnp.zeros(self.n_hidden)
        return jnp.zeros((batch, self.n_hidden))

    def _step(
        self, x_t: jax.Array, h_prev: jax.Array
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        r_pre = x_t @ self.W_r.value + h_prev @ self.U_r.value + self.b_r.value
        z_pre = x_t @ self.W_z.value + h_prev @ self.U_z.value + self.b_z.value
        r = _sigmoid(r_pre)
        z = _sigmoid(z_pre)
        rh = r * h_prev
        h_tilde_pre = x_t @ self.W.value + rh @ self.U.value + self.b.value
        h_tilde = jnp.tanh(h_tilde_pre)
        h = z * h_prev + (1.0 - z) * h_tilde
        cache = dict(
            r=r,
            z=z,
            h_prev=h_prev,
            rh=rh,
            h_tilde=h_tilde,
            r_pre=r_pre,
            z_pre=z_pre,
            h_tilde_pre=h_tilde_pre,
            x_t=x_t,
        )
        return h, cache

    def step(self, x_t: jax.Array, state: jax.Array) -> tuple[jax.Array, jax.Array]:
        """One step: ``(y_t, h_t)`` from ``(x_t, h_prev)``."""
        h, _ = self._step(jnp.asarray(x_t), state)
        return h, h

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Run the layer over a sequence, from a zero state."""
        del training
        seq, squeeze = _split_input(x)
        B, T, _ = seq.shape
        h = jnp.zeros((B, self.n_hidden), dtype=seq.dtype)
        ys = []
        self._cache = []
        for t in range(T):
            h, cache = self._step(seq[:, t, :], h)
            ys.append(h)
            self._cache.append(cache)
        y = jnp.stack(ys, axis=1)
        return y[0] if squeeze else y

    def backward(self, g: jax.Array) -> jax.Array:
        """Backpropagation through time; returns ``dx`` and adds to grads."""
        if not self._cache:
            raise RuntimeError("call forward() before backward()")
        g = jnp.asarray(g)
        squeeze = g.ndim == 2
        if squeeze:
            g = g[None]
        B, T, _ = g.shape
        grads = {
            name: jnp.zeros_like(p.value)
            for name, p in [
                ("W_r", self.W_r),
                ("W_z", self.W_z),
                ("W", self.W),
                ("U_r", self.U_r),
                ("U_z", self.U_z),
                ("U", self.U),
                ("b_r", self.b_r),
                ("b_z", self.b_z),
                ("b", self.b),
            ]
        }
        dh_next = jnp.zeros((B, self.n_hidden), dtype=g.dtype)
        dxs = [None] * T
        for t in range(T - 1, -1, -1):
            c = self._cache[t]
            h_prev = c["h_prev"]
            r = c["r"]
            z = c["z"]
            rh = c["rh"]
            h_tilde = c["h_tilde"]
            x_t = c["x_t"]
            dh = g[:, t, :] + dh_next
            dz = dh * (h_prev - h_tilde)
            dh_tilde = dh * (1.0 - z)
            dh_prev_direct = dh * z
            dzt = dz * z * (1.0 - z)
            dht_pre = dh_tilde * (1.0 - h_tilde**2)
            drh = dht_pre @ self.U.value.T
            dr = drh * h_prev
            dh_prev_from_r = drh * r
            drt = dr * r * (1.0 - r)
            grads["W"] = grads["W"] + x_t.T @ dht_pre
            grads["U"] = grads["U"] + rh.T @ dht_pre
            grads["b"] = grads["b"] + dht_pre.sum(axis=0)
            grads["W_z"] = grads["W_z"] + x_t.T @ dzt
            grads["U_z"] = grads["U_z"] + h_prev.T @ dzt
            grads["b_z"] = grads["b_z"] + dzt.sum(axis=0)
            grads["W_r"] = grads["W_r"] + x_t.T @ drt
            grads["U_r"] = grads["U_r"] + h_prev.T @ drt
            grads["b_r"] = grads["b_r"] + drt.sum(axis=0)
            dh_next = (
                dh_prev_direct
                + dh_prev_from_r
                + dzt @ self.U_z.value.T
                + drt @ self.U_r.value.T
            )
            dxs[t] = (
                dht_pre @ self.W.value.T
                + dzt @ self.W_z.value.T
                + drt @ self.W_r.value.T
            )
        self.W_r.grad = self.W_r.grad + grads["W_r"]
        self.W_z.grad = self.W_z.grad + grads["W_z"]
        self.W.grad = self.W.grad + grads["W"]
        self.U_r.grad = self.U_r.grad + grads["U_r"]
        self.U_z.grad = self.U_z.grad + grads["U_z"]
        self.U.grad = self.U.grad + grads["U"]
        self.b_r.grad = self.b_r.grad + grads["b_r"]
        self.b_z.grad = self.b_z.grad + grads["b_z"]
        self.b.grad = self.b.grad + grads["b"]
        dx = jnp.stack(dxs, axis=1)
        return dx[0] if squeeze else dx


class LSTM:
    """Vanilla LSTM with peepholes, Greff et al. 2017 pages 1-2.

    Forward: ``z = tanh(W_z x + R_z y_prev + b_z)``, ``i = sigmoid(W_i x +
    R_i y_prev + p_i * c_prev + b_i)``, ``f = sigmoid(W_f x + R_f y_prev +
    p_f * c_prev + b_f)``, ``c = z * i + c_prev * f``, ``o = sigmoid(W_o x
    + R_o y_prev + p_o * c + b_o)``, ``y = tanh(c) * o``. ``peepholes=False``
    removes the three vectors.

    Args:
        n_in: input features.
        n_hidden: hidden state size.
        key: a ``jax.random`` key.
        peepholes: include the three peephole vectors, default True.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from dense_armor.utility.neural.recurrent import LSTM
        >>> r = LSTM(2, 3, jax.random.PRNGKey(0))
        >>> r.forward(jnp.ones((4, 2))).shape
        (4, 3)
    """

    def __init__(
        self,
        n_in: int,
        n_hidden: int,
        key: Any,
        peepholes: bool = True,
    ) -> None:
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.peepholes = peepholes
        keys = jax.random.split(key, 8)
        self.W_z = Param(xavier_normal(keys[0], (n_in, n_hidden)))
        self.W_i = Param(xavier_normal(keys[1], (n_in, n_hidden)))
        self.W_f = Param(xavier_normal(keys[2], (n_in, n_hidden)))
        self.W_o = Param(xavier_normal(keys[3], (n_in, n_hidden)))
        self.R_z = Param(xavier_normal(keys[4], (n_hidden, n_hidden)))
        self.R_i = Param(xavier_normal(keys[5], (n_hidden, n_hidden)))
        self.R_f = Param(xavier_normal(keys[6], (n_hidden, n_hidden)))
        self.R_o = Param(xavier_normal(keys[7], (n_hidden, n_hidden)))
        self.b_z = Param(zeros_bias(n_hidden))
        self.b_i = Param(zeros_bias(n_hidden))
        self.b_f = Param(zeros_bias(n_hidden))
        self.b_o = Param(zeros_bias(n_hidden))
        if peepholes:
            self.p_i = Param(zeros_bias(n_hidden))
            self.p_f = Param(zeros_bias(n_hidden))
            self.p_o = Param(zeros_bias(n_hidden))
        self._cache: list[dict[str, jax.Array]] = []

    def __repr__(self) -> str:
        return f"LSTM({self.n_in}, {self.n_hidden}, peepholes={self.peepholes})"

    def params(self) -> list[Param]:
        """Every parameter, in a fixed order."""
        out = [
            self.W_z,
            self.W_i,
            self.W_f,
            self.W_o,
            self.R_z,
            self.R_i,
            self.R_f,
            self.R_o,
            self.b_z,
            self.b_i,
            self.b_f,
            self.b_o,
        ]
        if self.peepholes:
            out += [self.p_i, self.p_f, self.p_o]
        return out

    def init_state(self, batch: int | None = None) -> tuple[jax.Array, jax.Array]:
        """The pair ``(y, c)``, both zero."""
        if batch is None:
            return jnp.zeros(self.n_hidden), jnp.zeros(self.n_hidden)
        return jnp.zeros((batch, self.n_hidden)), jnp.zeros((batch, self.n_hidden))

    def _step(
        self, x_t: jax.Array, state: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array], dict[str, jax.Array]]:
        y_prev, c_prev = state
        if self.peepholes:
            pi, pf, po = self.p_i.value, self.p_f.value, self.p_o.value
        else:
            pi = pf = po = 0.0
        z_pre = x_t @ self.W_z.value + y_prev @ self.R_z.value + self.b_z.value
        i_pre = (
            x_t @ self.W_i.value
            + y_prev @ self.R_i.value
            + pi * c_prev
            + self.b_i.value
        )
        f_pre = (
            x_t @ self.W_f.value
            + y_prev @ self.R_f.value
            + pf * c_prev
            + self.b_f.value
        )
        z = jnp.tanh(z_pre)
        i = _sigmoid(i_pre)
        f = _sigmoid(f_pre)
        c = z * i + c_prev * f
        o_pre = x_t @ self.W_o.value + y_prev @ self.R_o.value + po * c + self.b_o.value
        o = _sigmoid(o_pre)
        y = jnp.tanh(c) * o
        cache = dict(
            z=z,
            i=i,
            f=f,
            o=o,
            y_prev=y_prev,
            c_prev=c_prev,
            c=c,
            tc=jnp.tanh(c),
            x_t=x_t,
        )
        return y, (y, c), cache

    def step(
        self, x_t: jax.Array, state: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        """One step: ``(y_t, (y_t, c_t))`` from ``(x_t, (y_prev, c_prev))``."""
        y, new_state, _ = self._step(jnp.asarray(x_t), state)
        return y, new_state

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Run the layer over a sequence, from a zero state."""
        del training
        seq, squeeze = _split_input(x)
        B, T, _ = seq.shape
        y = jnp.zeros((B, self.n_hidden), dtype=seq.dtype)
        c = jnp.zeros((B, self.n_hidden), dtype=seq.dtype)
        ys = []
        self._cache = []
        for t in range(T):
            y, (y, c), cache = self._step(seq[:, t, :], (y, c))
            ys.append(y)
            self._cache.append(cache)
        out = jnp.stack(ys, axis=1)
        return out[0] if squeeze else out

    def backward(self, g: jax.Array) -> jax.Array:
        """Backpropagation through time; returns ``dx`` and adds to grads."""
        if not self._cache:
            raise RuntimeError("call forward() before backward()")
        g = jnp.asarray(g)
        squeeze = g.ndim == 2
        if squeeze:
            g = g[None]
        B, T, _ = g.shape
        names = [
            "W_z",
            "W_i",
            "W_f",
            "W_o",
            "R_z",
            "R_i",
            "R_f",
            "R_o",
            "b_z",
            "b_i",
            "b_f",
            "b_o",
        ]
        if self.peepholes:
            names += ["p_i", "p_f", "p_o"]
        grads = {n: jnp.zeros_like(getattr(self, n).value) for n in names}
        dy_next = jnp.zeros((B, self.n_hidden), dtype=g.dtype)
        dc_next = jnp.zeros((B, self.n_hidden), dtype=g.dtype)
        dxs = [None] * T
        pi = self.p_i.value if self.peepholes else 0.0
        pf = self.p_f.value if self.peepholes else 0.0
        po = self.p_o.value if self.peepholes else 0.0
        for t in range(T - 1, -1, -1):
            c = self._cache[t]
            z, i, f, o = c["z"], c["i"], c["f"], c["o"]
            y_prev, c_prev, cc, tc, x_t = (
                c["y_prev"],
                c["c_prev"],
                c["c"],
                c["tc"],
                c["x_t"],
            )
            dy = g[:, t, :] + dy_next
            do_pre = dy * tc * o * (1.0 - o)
            dc = dy * o * (1.0 - tc**2) + dc_next
            if self.peepholes:
                dc = dc + po * do_pre
            df_pre = dc * c_prev * f * (1.0 - f)
            di_pre = dc * z * i * (1.0 - i)
            dz_pre = dc * i * (1.0 - z**2)
            grads["W_z"] = grads["W_z"] + x_t.T @ dz_pre
            grads["W_i"] = grads["W_i"] + x_t.T @ di_pre
            grads["W_f"] = grads["W_f"] + x_t.T @ df_pre
            grads["W_o"] = grads["W_o"] + x_t.T @ do_pre
            grads["R_z"] = grads["R_z"] + y_prev.T @ dz_pre
            grads["R_i"] = grads["R_i"] + y_prev.T @ di_pre
            grads["R_f"] = grads["R_f"] + y_prev.T @ df_pre
            grads["R_o"] = grads["R_o"] + y_prev.T @ do_pre
            grads["b_z"] = grads["b_z"] + dz_pre.sum(axis=0)
            grads["b_i"] = grads["b_i"] + di_pre.sum(axis=0)
            grads["b_f"] = grads["b_f"] + df_pre.sum(axis=0)
            grads["b_o"] = grads["b_o"] + do_pre.sum(axis=0)
            if self.peepholes:
                grads["p_i"] = grads["p_i"] + (di_pre * c_prev).sum(axis=0)
                grads["p_f"] = grads["p_f"] + (df_pre * c_prev).sum(axis=0)
                grads["p_o"] = grads["p_o"] + (do_pre * cc).sum(axis=0)
                dc = dc + po * do_pre
            dy_next = (
                dz_pre @ self.R_z.value.T
                + di_pre @ self.R_i.value.T
                + df_pre @ self.R_f.value.T
                + do_pre @ self.R_o.value.T
            )
            dc_next = dc * f + pi * di_pre + pf * df_pre
            if self.peepholes:
                dc_next = dc * f + pi * di_pre + pf * df_pre
            dxs[t] = (
                dz_pre @ self.W_z.value.T
                + di_pre @ self.W_i.value.T
                + df_pre @ self.W_f.value.T
                + do_pre @ self.W_o.value.T
            )
        for n in names:
            p = getattr(self, n)
            p.grad = p.grad + grads[n]
        dx = jnp.stack(dxs, axis=1)
        return dx[0] if squeeze else dx
