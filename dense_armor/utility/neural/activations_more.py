"""More activations: GELU, SiLU, ELU, LeakyReLU, Softplus.

Each activation has two pure functions, ``<name>_forward(x)`` and
``<name>_backward(x, y, g)``, that work under ``jax.jit``, and a class
whose ``forward(x)`` caches ``x`` and ``y`` and whose ``backward(g)``
returns the gradient of the input, as in
:mod:`dense_armor.utility.neural.activations`.

References:
    Hendrycks, D., Gimpel, K. (2016). Gaussian Error Linear Units
        (GELUs). arXiv:1606.08415. Page 1: ``GELU(x) = x Phi(x)``, with
        ``Phi`` the standard Gaussian CDF. Page 2: the approximation
        ``0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x**3)))``.

    Elfwing, S., Uchibe, E., Doya, K. (2017). Sigmoid-weighted linear
        units for neural network function approximation in
        reinforcement learning. arXiv:1702.03118. Pages 1 and 2: the
        SiLU, ``f(x) = x * sigmoid(x)``.

    Clevert, D.-A., Unterthiner, T., Hochreiter, S. (2016). Fast and
        accurate deep network learning by exponential linear units
        (ELUs). arXiv:1511.07289. Eq.(15), page 5: the ELU,
        ``f(x) = x`` if ``x > 0``, ``alpha (exp(x) - 1)`` otherwise.

    He, K., Zhang, X., Ren, S., Sun, J. (2015). Delving deep into
        rectifiers. arXiv:1502.01852. Page 2: the leaky ReLU with a
        small fixed ``a``.

    The softplus is a textbook definition; the source cited in
    Dugas et al. (2001) is not in this folder and is cited as
    ``bibliographic only``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erf

_SQRT_2_OVER_PI = float(np.sqrt(2.0 / np.pi))
_SQRT_2_PI = float(np.sqrt(2.0 * np.pi))


def gelu_forward(x: jax.Array) -> jax.Array:
    """``x * Phi(x)`` (Eq. on page 1 of Hendrycks and Gimpel 2016)."""
    phi = 0.5 * (1.0 + erf(x / jnp.sqrt(2.0)))
    return x * phi


def gelu_backward(x: jax.Array, y: jax.Array, g: jax.Array) -> jax.Array:
    """Derivative of the exact GELU: ``Phi(x) + x phi(x)``."""
    del y
    phi_cdf = 0.5 * (1.0 + erf(x / jnp.sqrt(2.0)))
    phi_pdf = jnp.exp(-0.5 * x * x) / _SQRT_2_PI
    return g * (phi_cdf + x * phi_pdf)


def gelu_approx_forward(x: jax.Array) -> jax.Array:
    """``0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x**3)))``."""
    inner = _SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1.0 + jnp.tanh(inner))


def gelu_approx_backward(x: jax.Array, y: jax.Array, g: jax.Array) -> jax.Array:
    """Derivative of the tanh approximation of the GELU."""
    del y
    inner = _SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)
    t = jnp.tanh(inner)
    d_inner = _SQRT_2_OVER_PI * (1.0 + 3.0 * 0.044715 * x * x)
    return g * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * d_inner)


def silu_forward(x: jax.Array) -> jax.Array:
    """``x * sigmoid(x)`` (SiLU, pages 1 and 2 of Elfwing et al. 2017)."""
    return x * jax.nn.sigmoid(x)


def silu_backward(x: jax.Array, y: jax.Array, g: jax.Array) -> jax.Array:
    """Derivative of SiLU: ``s + x s (1 - s)`` with ``s = sigmoid(x)``."""
    del y
    s = jax.nn.sigmoid(x)
    return g * (s + x * s * (1.0 - s))


def elu_forward(x: jax.Array, alpha: float = 1.0) -> jax.Array:
    """``x`` if ``x > 0``, ``alpha (exp(x) - 1)`` otherwise (Eq.(15))."""
    return jnp.where(x > 0.0, x, alpha * jnp.expm1(x))


def elu_backward(
    x: jax.Array, y: jax.Array, g: jax.Array, alpha: float = 1.0
) -> jax.Array:
    """Derivative of ELU: ``1`` if ``x > 0``, ``alpha exp(x)`` otherwise."""
    del y
    return jnp.where(x > 0.0, g, g * alpha * jnp.exp(x))


def leaky_relu_forward(x: jax.Array, slope: float = 0.01) -> jax.Array:
    """``x`` if ``x > 0``, ``slope * x`` otherwise."""
    return jnp.where(x > 0.0, x, slope * x)


def leaky_relu_backward(
    x: jax.Array, y: jax.Array, g: jax.Array, slope: float = 0.01
) -> jax.Array:
    """Derivative of LeakyReLU: ``1`` if ``x > 0``, ``slope`` otherwise."""
    del y
    return jnp.where(x > 0.0, g, slope * g)


def softplus_forward(x: jax.Array, beta: float = 1.0) -> jax.Array:
    """``log(1 + exp(beta x)) / beta``, stable with ``jax.nn.softplus``."""
    return jax.nn.softplus(beta * x) / beta


def softplus_backward(
    x: jax.Array, y: jax.Array, g: jax.Array, beta: float = 1.0
) -> jax.Array:
    """Derivative of the softplus: ``sigmoid(beta x)``."""
    del y
    return g * jax.nn.sigmoid(beta * x)


class GELU:
    """GELU (exact or approximate) with a forward and a backward."""

    def __init__(self, approximate: bool = False) -> None:
        self.approximate = bool(approximate)
        self.x: jax.Array | None = None
        self.y: jax.Array | None = None

    def forward(self, x: jax.Array) -> jax.Array:
        if self.approximate:
            y = gelu_approx_forward(x)
        else:
            y = gelu_forward(x)
        self.x = x
        self.y = y
        return y

    def backward(self, g: jax.Array) -> jax.Array:
        if self.x is None or self.y is None:
            raise RuntimeError("call forward() before backward()")
        if self.approximate:
            return gelu_approx_backward(self.x, self.y, g)
        return gelu_backward(self.x, self.y, g)


class SiLU:
    """SiLU: ``x * sigmoid(x)``."""

    def __init__(self) -> None:
        self.x: jax.Array | None = None
        self.y: jax.Array | None = None

    def forward(self, x: jax.Array) -> jax.Array:
        y = silu_forward(x)
        self.x = x
        self.y = y
        return y

    def backward(self, g: jax.Array) -> jax.Array:
        if self.x is None or self.y is None:
            raise RuntimeError("call forward() before backward()")
        return silu_backward(self.x, self.y, g)


class ELU:
    """ELU with a scale ``alpha`` (Eq.(15), page 5 of Clevert et al.)."""

    def __init__(self, alpha: float = 1.0) -> None:
        if alpha <= 0.0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        self.alpha = float(alpha)
        self.x: jax.Array | None = None
        self.y: jax.Array | None = None

    def forward(self, x: jax.Array) -> jax.Array:
        y = elu_forward(x, self.alpha)
        self.x = x
        self.y = y
        return y

    def backward(self, g: jax.Array) -> jax.Array:
        if self.x is None or self.y is None:
            raise RuntimeError("call forward() before backward()")
        return elu_backward(self.x, self.y, g, self.alpha)


class LeakyReLU:
    """Leaky ReLU with a fixed slope (no Param)."""

    def __init__(self, slope: float = 0.01) -> None:
        if slope < 0.0:
            raise ValueError(f"slope must be >= 0, got {slope}")
        self.slope = float(slope)
        self.x: jax.Array | None = None
        self.y: jax.Array | None = None

    def forward(self, x: jax.Array) -> jax.Array:
        y = leaky_relu_forward(x, self.slope)
        self.x = x
        self.y = y
        return y

    def backward(self, g: jax.Array) -> jax.Array:
        if self.x is None or self.y is None:
            raise RuntimeError("call forward() before backward()")
        return leaky_relu_backward(self.x, self.y, g, self.slope)


class Softplus:
    """Softplus with a scale ``beta``."""

    def __init__(self, beta: float = 1.0) -> None:
        if beta <= 0.0:
            raise ValueError(f"beta must be > 0, got {beta}")
        self.beta = float(beta)
        self.x: jax.Array | None = None
        self.y: jax.Array | None = None

    def forward(self, x: jax.Array) -> jax.Array:
        y = softplus_forward(x, self.beta)
        self.x = x
        self.y = y
        return y

    def backward(self, g: jax.Array) -> jax.Array:
        if self.x is None or self.y is None:
            raise RuntimeError("call forward() before backward()")
        return softplus_backward(self.x, self.y, g, self.beta)
