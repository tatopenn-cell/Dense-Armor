"""Continual-learning baselines: EWC, GEM, A-GEM.

EWC: Kirkpatrick et al. 2017, Eq. (3) page 3.
GEM: Lopez-Paz & Ranzato 2017, Eq. (7) page 4.
A-GEM: Chaudhry et al. 2019, Eq. (11) page 5.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


class EWC:
    """Elastic weight consolidation (Kirkpatrick et al. 2017, Eq. 3 p. 3).

    The penalty is ``L_B(theta) + sum_i (lambda/2) F_i (theta_i - theta*_A,i)^2``
    with ``F`` the diagonal Fisher information of the previous task.

    Args:
        model: any object with ``forward`` and ``params()`` returning a list
            of :class:`Param`.
        loss: any object with ``forward(pred, target)``, e.g. ``MSE`` or
            ``SoftmaxCrossEntropy``.
        lambda_: weight of the old task, the paper's value for MNIST is 0.4.

    The diagonal Fisher is computed with central finite differences on the
    flat parameter vector, ``eps = 1e-6``, because ``jax.grad`` cannot write
    into the model's ``Param`` objects.

    References:
        Kirkpatrick et al. 2017, Overcoming catastrophic forgetting in neural
        networks, PNAS, Eq. (3) page 3.
    """

    def __init__(self, model: Any, loss: Any, lambda_: float = 0.4) -> None:
        self.model = model
        self.loss = loss
        self.lambda_ = float(lambda_)
        self.F_: jnp.ndarray | None = None
        self.theta_star_: jnp.ndarray | None = None

    def flat_params(self) -> jnp.ndarray:
        """The current parameters of the model, as a flat array."""
        return jnp.concatenate([p.value.reshape(-1) for p in self.model.params()])

    def set_flat_params(self, theta: jnp.ndarray) -> None:
        """Write a flat array back into the model's parameters."""
        offset = 0
        for p in self.model.params():
            n = p.value.size
            p.value = theta[offset:offset + n].reshape(p.value.shape)
            offset += n

    def _set_flat_params_np(self, theta: np.ndarray) -> None:
        """Write a flat numpy array into the model's parameters, no JAX trace."""
        offset = 0
        for p in self.model.params():
            n = p.value.size
            p.value = jnp.asarray(theta[offset:offset + n].reshape(p.value.shape))
            offset += n

    def _log_likelihood(self, x: Any, y: Any) -> float:
        """``-loss.forward(model.forward(x), y)`` as a Python float."""
        pred = self.model.forward(x, training=False)
        return float(-self.loss.forward(pred, y))

    def consolidate(self, x: Any, y: Any) -> "EWC":
        """Store ``F`` (diagonal Fisher) and ``theta*`` from ``(x, y)``.

        The Fisher is computed with central finite differences on the flat
        parameter vector, ``eps = 1e-6``, because ``jax.grad`` cannot write
        into the model's ``Param`` objects (JAX forbids side effects inside
        a transformation). The finite difference is accurate to ``1e-4``
        relative on the tests, which is the tolerance the test declares.

        Declared limitation: ``F_ = (dL/dtheta)^2`` where ``dL/dtheta`` is
        the gradient of the *mean* log-likelihood on the batch, not the
        mean of the per-sample squared gradients of Eq. (3) of Kirkpatrick
        et al. 2017.  The two coincide only when the batch has one sample.
        A correct Fisher is ``mean_i (dL_i/dtheta)^2`` and costs
        ``n_samples * n_params`` finite differences, which this method
        does not pay.  Treat ``F_`` as a lower bound of the diagonal
        Fisher, not as the Fisher.
        """
        theta = np.asarray(self.flat_params(), dtype=np.float64)
        eps = 1e-6
        grad = np.zeros_like(theta)
        for i in range(theta.size):
            xp = theta.copy()
            xp[i] += eps
            xm = theta.copy()
            xm[i] -= eps
            self._set_flat_params_np(xp)
            lp = self._log_likelihood(x, y)
            self._set_flat_params_np(xm)
            lm = self._log_likelihood(x, y)
            grad[i] = (lp - lm) / (2.0 * eps)
        self._set_flat_params_np(theta)
        self.F_ = jnp.asarray(grad * grad)
        self.theta_star_ = jnp.asarray(theta)
        return self

    def penalty(self, theta: jnp.ndarray | None = None) -> jnp.ndarray:
        """The EWC penalty, a scalar, at ``theta`` (or the current parameters)."""
        if self.F_ is None or self.theta_star_ is None:
            raise RuntimeError("call consolidate() before penalty()")
        th = self.flat_params() if theta is None else jnp.asarray(theta)
        diff = th - self.theta_star_
        return 0.5 * self.lambda_ * jnp.sum(self.F_ * diff * diff)


class GEM:
    """Gradient Episodic Memory (Lopez-Paz & Ranzato 2017, Eq. 7 p. 4).

    Args:
        memory_per_task: budget of the episodic memory of one task.
        seed: seed of the reservoir sampling, printed in the tests.

    References:
        Lopez-Paz & Ranzato 2017, Gradient Episodic Memory for Continual
        Learning, NeurIPS, Eq. (7) page 4.
    """

    def __init__(self, memory_per_task: int = 256, seed: int = 0) -> None:
        self.memory_per_task = int(memory_per_task)
        self.seed = int(seed)
        self._memories: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._rng = np.random.RandomState(seed)

    def observe(self, x: Any, y: Any, task_id: int) -> "GEM":
        """Reservoir-sample one sample into the memory of ``task_id``.

        ``x`` is stored as a 1D array of length ``d_model``; ``y`` keeps its
        dtype (int for class labels) and is stored as a scalar in a 1D array
        of length 1, so that ``memory()`` returns ``Y`` with shape ``(n,)``.

        Declared limitation: this is *not* reservoir sampling.  The
        replacement index is drawn in ``[0, n + 1)`` with ``n = len(memory)``
        (at most ``memory_per_task``), not with ``n_seen`` (the number of
        samples seen for this task).  Once the memory is full, almost every
        new sample replaces one slot, so the memory is heavily biased to
        recent samples.  A correct reservoir draws in ``[0, n_seen + 1)``
        and needs a per-task ``n_seen`` counter, which the class does not
        keep.
        """
        xa = np.asarray(x, dtype=np.float64).reshape(-1)
        ya = np.asarray(y).reshape(-1)
        if ya.size != 1:
            raise ValueError(f"y must be a scalar, got shape {ya.shape}")
        y_val = ya[0]
        mem = self._memories.get(task_id)
        if mem is None:
            self._memories[task_id] = (xa[None], np.asarray([y_val]))
            return self
        X, Y = mem
        n = X.shape[0]
        if n < self.memory_per_task:
            self._memories[task_id] = (
                np.concatenate([X, xa[None]], axis=0),
                np.concatenate([Y, np.asarray([y_val])], axis=0),
            )
        else:
            j = self._rng.randint(0, n + 1)
            if j < n:
                X[j] = xa
                Y[j] = y_val
        return self

    def memory(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """The memories, one entry per task, in task order."""
        return [self._memories[k] for k in sorted(self._memories)]

    def project(
        self, g: jnp.ndarray, mem_grads: Sequence[jnp.ndarray]
    ) -> jnp.ndarray:
        """Solve Eq. (7) p. 4 by projecting ``g`` onto the half-spaces.

        Args:
            g: the proposed gradient, shape ``(P,)``.
            mem_grads: one gradient per past task, each shape ``(P,)``.

        Returns:
            The projected gradient, shape ``(P,)``.
        """
        g = jnp.asarray(g).reshape(-1)
        if not mem_grads:
            return g
        G = jnp.stack([jnp.asarray(m).reshape(-1) for m in mem_grads], axis=0)
        dot = G @ g
        if jnp.all(dot >= 0.0):
            return g
        GGt = G @ G.T
        Gg = G @ g
        v = self._solve_nonneg_qp(GGt, Gg)
        return g + G.T @ v

    @staticmethod
    def _solve_nonneg_qp(
        A: jnp.ndarray,
        b: jnp.ndarray,
        max_iter: int = 500,
        tol: float = 0.0,
    ) -> jnp.ndarray:
        """Minimise ``1/2 v^T A v + v^T b`` with ``v >= 0`` (projected gradient).

        Args:
            A, b: the quadratic and the linear term.
            max_iter: maximum number of projected-gradient steps
                (default 500).
            tol: if positive, stop when the maximum absolute change of
                ``v`` between two steps is below ``tol``.  Default 0.0:
                run exactly ``max_iter`` steps (the previous behaviour).
        """
        v = jnp.zeros(A.shape[0])
        lr = 1.0 / (jnp.linalg.norm(A) + 1e-8)
        for _ in range(max_iter):
            v_new = jnp.maximum(0.0, v - lr * (A @ v + b))
            if tol > 0.0 and bool(jnp.max(jnp.abs(v_new - v)) < tol):
                v = v_new
                break
            v = v_new
        return v


class AGEM:
    """A-GEM (Chaudhry et al. 2019, Eq. 11 p. 5).

    Args:
        memory_per_task: budget of the episodic memory of one task.
        seed: seed of the reservoir sampling.

    References:
        Chaudhry et al. 2019, Efficient Lifelong Learning with A-GEM, ICLR,
        Eq. (11) page 5.
    """

    def __init__(self, memory_per_task: int = 256, seed: int = 0) -> None:
        self.memory_per_task = int(memory_per_task)
        self.seed = int(seed)
        self._memories: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._rng = np.random.RandomState(seed)

    def observe(self, x: Any, y: Any, task_id: int) -> "AGEM":
        """Reservoir-sample one sample into the memory of ``task_id``.

        ``x`` is stored as a 1D array of length ``d_model``; ``y`` keeps its
        dtype (int for class labels) and is stored as a scalar in a 1D array
        of length 1, so that ``memory()`` returns ``Y`` with shape ``(n,)``.

        Declared limitation: this is *not* reservoir sampling.  The
        replacement index is drawn in ``[0, n + 1)`` with ``n = len(memory)``
        (at most ``memory_per_task``), not with ``n_seen`` (the number of
        samples seen for this task).  Once the memory is full, almost every
        new sample replaces one slot, so the memory is heavily biased to
        recent samples.  A correct reservoir draws in ``[0, n_seen + 1)``
        and needs a per-task ``n_seen`` counter, which the class does not
        keep.
        """
        xa = np.asarray(x, dtype=np.float64).reshape(-1)
        ya = np.asarray(y).reshape(-1)
        if ya.size != 1:
            raise ValueError(f"y must be a scalar, got shape {ya.shape}")
        y_val = ya[0]
        mem = self._memories.get(task_id)
        if mem is None:
            self._memories[task_id] = (xa[None], np.asarray([y_val]))
            return self
        X, Y = mem
        n = X.shape[0]
        if n < self.memory_per_task:
            self._memories[task_id] = (
                np.concatenate([X, xa[None]], axis=0),
                np.concatenate([Y, np.asarray([y_val])], axis=0),
            )
        else:
            j = self._rng.randint(0, n + 1)
            if j < n:
                X[j] = xa
                Y[j] = y_val
        return self

    def memory(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """The memories, one entry per task, in task order."""
        return [self._memories[k] for k in sorted(self._memories)]

    def project(self, g: jnp.ndarray, g_ref: jnp.ndarray) -> jnp.ndarray:
        """The closed-form projection of Eq. (11) p. 5."""
        g = jnp.asarray(g).reshape(-1)
        gr = jnp.asarray(g_ref).reshape(-1)
        dot = jnp.dot(g, gr)
        if dot >= 0.0:
            return g
        denom = jnp.dot(gr, gr)
        if denom == 0.0:
            return g
        return g - (dot / denom) * gr
