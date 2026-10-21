"""BatchNorm and LayerNorm, with a forward and a backward.

References:
    Ioffe, S., Szegedy, C. (2015). Batch Normalization: Accelerating
        Deep Network Training by Reducing Internal Covariate Shift.
        arXiv:1502.03167. Page 3, Algorithm 1: the batch mean and
        variance, ``xhat = (x - mu) / sqrt(var + eps)``, ``y = gamma *
        xhat + beta``. Page 4: the unbiased estimate ``m/(m-1)``, the
        moving averages, Algorithm 2 step 11 for inference.
    Ba, J. L., Kiros, J. R., Hinton, G. E. (2016). Layer Normalization.
        arXiv:1607.06450. Page 1: the gain and the bias come after the
        normalisation. Page 2, Eq.(3): ``mu`` and ``sigma`` over the
        features of one sample.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from dense_armor.utility.neural.params import Param


class BatchNorm:
    """Batch normalisation of a 2-D batch ``(m, n_features)``.

    In training mode the mean and the variance are computed over the
    batch axis, ``y = gamma * (x - mu) / sqrt(var + eps) + beta``
    (Algorithm 1, page 3). The running statistics follow the paper's
    moving average with the unbiased variance ``m/(m-1) * var`` (page
    4). In evaluation mode the running statistics are used, also for a
    single sample (Algorithm 2 step 11, page 4).

    Args:
        n_features: number of features in the last axis.
        eps: constant added to the variance for numerical stability.
            The paper (page 3) does not give a value; ``1e-5`` is our
            choice.
        momentum: weight of the current batch in the running average.
            The paper (page 4) does not give a value; ``0.1`` is our
            choice.

    Raises:
        ValueError: on ``forward`` in training mode with a batch of one
            sample, because the batch variance is not defined.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from dense_armor.utility.neural.norm import BatchNorm
        >>> bn = BatchNorm(3)
        >>> y = bn.forward(jnp.ones((4, 3)))
        >>> y.shape
        (4, 3)
    """

    def __init__(
        self,
        n_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ) -> None:
        self.n_features = int(n_features)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.gamma = Param(jnp.ones(n_features))
        self.beta = Param(jnp.zeros(n_features))
        self.running_mean_: np.ndarray = np.zeros(n_features, dtype=np.float32)
        self.running_var_: np.ndarray = np.ones(n_features, dtype=np.float32)
        self.xhat_: jax.Array | None = None
        self.std_: jax.Array | None = None
        self.m_: int = 0
        self.mode_: str = "train"

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Normalise the batch or apply the running statistics."""
        x = jnp.asarray(x)
        if not training:
            mu = jnp.asarray(self.running_mean_)
            var = jnp.asarray(self.running_var_)
            xhat = (x - mu) / jnp.sqrt(var + self.eps)
            self.xhat_ = xhat
            self.std_ = jnp.sqrt(var + self.eps)
            self.mode_ = "eval"
            return self.gamma.value * xhat + self.beta.value
        if x.ndim == 1:
            raise ValueError(
                "BatchNorm needs a 2-D batch of shape (m, n_features), "
                f"got 1-D of shape {x.shape}"
            )
        m = x.shape[0]
        if m < 2:
            raise ValueError(f"BatchNorm training needs at least two samples, got {m}")
        mu = jnp.mean(x, axis=0)
        var = jnp.mean((x - mu) ** 2, axis=0)
        std = jnp.sqrt(var + self.eps)
        xhat = (x - mu) / std
        self.xhat_ = xhat
        self.std_ = std
        self.m_ = int(m)
        self.mode_ = "train"
        var_unbiased = var * (m / (m - 1))
        mo = self.momentum
        self.running_mean_ = (1.0 - mo) * self.running_mean_ + mo * np.asarray(mu)
        self.running_var_ = (1.0 - mo) * self.running_var_ + mo * np.asarray(
            var_unbiased
        )
        return self.gamma.value * xhat + self.beta.value

    def backward(self, g: jax.Array) -> jax.Array:
        """Return ``dx`` and add to ``gamma.grad`` and ``beta.grad``."""
        if self.xhat_ is None or self.std_ is None:
            raise RuntimeError("call forward() before backward()")
        if self.mode_ == "eval":
            raise RuntimeError(
                "backward after an evaluation forward has no training "
                "gradient; call forward(training=True) before backward()"
            )
        xhat = self.xhat_
        std = self.std_
        self.gamma.grad = self.gamma.grad + jnp.sum(g * xhat, axis=0)
        self.beta.grad = self.beta.grad + jnp.sum(g, axis=0)
        m = self.m_
        dxhat = g * self.gamma.value
        dx = (
            m * dxhat - jnp.sum(dxhat, axis=0) - xhat * jnp.sum(dxhat * xhat, axis=0)
        ) / (m * std)
        return dx

    def params(self) -> list[Param]:
        """The two ``Param`` objects, ``gamma`` then ``beta``."""
        return [self.gamma, self.beta]


class LayerNorm:
    """Layer normalisation over the features of each sample.

    For each sample ``mu`` and ``sigma`` are the mean and the standard
    deviation over the features (Eq.(3), page 2), and ``y = gain *
    (x - mu) / sqrt(sigma**2 + eps) + bias``, with the gain and the
    bias after the normalisation (page 1). ``eps`` inside the root is
    our choice: the paper writes no constant.

    Args:
        n_features: number of features in the last axis.
        eps: constant added inside the root. Default ``1e-5``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from dense_armor.utility.neural.norm import LayerNorm
        >>> ln = LayerNorm(3)
        >>> ln.forward(jnp.ones((4, 3))).shape
        (4, 3)
    """

    def __init__(self, n_features: int, eps: float = 1e-5) -> None:
        self.n_features = int(n_features)
        self.eps = float(eps)
        self.gain = Param(jnp.ones(n_features))
        self.bias = Param(jnp.zeros(n_features))
        self.xhat_: jax.Array | None = None
        self.std_: jax.Array | None = None

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Return the normalised input; ``training`` is ignored."""
        del training
        x = jnp.asarray(x)
        mu = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean((x - mu) ** 2, axis=-1, keepdims=True)
        std = jnp.sqrt(var + self.eps)
        xhat = (x - mu) / std
        self.xhat_ = xhat
        self.std_ = std
        return self.gain.value * xhat + self.bias.value

    def backward(self, g: jax.Array) -> jax.Array:
        """Return ``dx`` and add to ``gain.grad`` and ``bias.grad``."""
        if self.xhat_ is None or self.std_ is None:
            raise RuntimeError("call forward() before backward()")
        xhat = self.xhat_
        std = self.std_
        sum_axes = tuple(range(g.ndim - 1))
        self.gain.grad = self.gain.grad + jnp.sum(g * xhat, axis=sum_axes)
        self.bias.grad = self.bias.grad + jnp.sum(g, axis=sum_axes)
        n = self.n_features
        dxhat = g * self.gain.value
        dx = (
            n * dxhat
            - jnp.sum(dxhat, axis=-1, keepdims=True)
            - xhat * jnp.sum(dxhat * xhat, axis=-1, keepdims=True)
        ) / (n * std)
        return dx

    def params(self) -> list[Param]:
        """The two ``Param`` objects, ``gain`` then ``bias``."""
        return [self.gain, self.bias]
