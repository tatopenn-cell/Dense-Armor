"""Continual-learning baselines: EWC, GEM, A-GEM.

EWC: Kirkpatrick et al. 2017, Eq. (3) page 3.
GEM: Lopez-Paz & Ranzato 2017, Eq. (8) QP and Eq. (11) dual, page 4.
A-GEM: Chaudhry et al. 2019, Eq. (11) page 5.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np

from dense_armor._dtype import as_default_array


class EWC_OLD:
    """Elastic weight consolidation (Kirkpatrick et al. 2017, Eq. 3 p. 3).

    The penalty is ``L_B(theta) + sum_i (lambda/2) F_i (theta_i - theta*_A,i)^2``
    with ``F`` the diagonal Fisher information of the previous task.

    Args:
        model: any object with ``forward`` and ``params()`` returning a list
            of :class:`Param`.
        loss: any object with ``forward(pred, target)``, e.g. ``MSE`` or
            ``SoftmaxCrossEntropy``.
        lambda_: weight of the old task. 0.4 is our choice: Kirkpatrick et al.
            2017 (PNAS 114(13), p. 5) tuned lambda by hyperparameter search on
            Atari and state no single value; 0.4 is not in the paper.

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

    def consolidate(self, x: Any, y: Any) -> EWC:
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
        theta = as_default_array(self.flat_params())
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


class GEM_OLD:
    """Gradient Episodic Memory (Lopez-Paz & Ranzato 2017, Eq. 8 QP and Eq. 11 dual, p. 4).

    Args:
        memory_per_task: budget of the episodic memory of one task.
        seed: seed of the reservoir sampling, printed in the tests.

    References:
        Lopez-Paz & Ranzato 2017, Gradient Episodic Memory for Continual
        Learning, NeurIPS, Eq. (8) QP and Eq. (11) dual, page 4.
    """

    def __init__(self, memory_per_task: int = 256, seed: int = 0) -> None:
        self.memory_per_task = int(memory_per_task)
        self.seed = int(seed)
        self._memories: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._rng = np.random.RandomState(seed)

    def observe(self, x: Any, y: Any, task_id: int) -> GEM:
        """Store one sample into the memory of ``task_id``.

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
        xa = as_default_array(x).reshape(-1)
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
        """Solve Eq. (8) p. 4 (QP) by projecting ``g`` onto the half-spaces.

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


class AGEM_OLD:
    """A-GEM (Chaudhry et al. 2019, Eq. 11 p. 5).

    Args:
        memory_per_task: budget of the episodic memory of one task.
        seed: seed of the random state used to pick the replacement index.

    References:
        Chaudhry et al. 2019, Efficient Lifelong Learning with A-GEM, ICLR,
        Eq. (11) page 5.
    """

    def __init__(self, memory_per_task: int = 256, seed: int = 0) -> None:
        self.memory_per_task = int(memory_per_task)
        self.seed = int(seed)
        self._memories: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._rng = np.random.RandomState(seed)

    def observe(self, x: Any, y: Any, task_id: int) -> AGEM:
        """Store one sample into the memory of ``task_id``.

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
        xa = as_default_array(x).reshape(-1)
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


# ============================================================
# NEW WORKING IMPLEMENTATIONS (replacing buggy old versions)
# ============================================================
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
from dense_armor._dtype import as_default_array, get_default_dtype

def get_flat_params(model):
    flat, treedef = jax.tree_util.tree_flatten([p.value for p in model.params()])
    flat_vals = [jnp.asarray(p).reshape(-1) for p in flat]
    return jnp.concatenate(flat_vals), treedef

def set_flat_params(model, theta, treedef):
    flat, _ = jax.tree_util.tree_flatten([p.value for p in model.params()])
    offset = 0
    new_flat = []
    for p in flat:
        n = p.size
        new_flat.append(theta[offset:offset + n].reshape(p.shape))
        offset += n
    new_params = treedef.unflatten(new_flat)
    for p, new_val in zip(model.params(), new_flat):
        p.value = new_val

@dataclass
class Reservoir:
    capacity: int
    seed: int = 0
    def __post_init__(self):
        import numpy as np
        self._rng = np.random.RandomState(self.seed)
        self._buffer = []
        self._n_seen = 0
    def add(self, item):
        self._n_seen += 1
        if len(self._buffer) < self.capacity:
            self._buffer.append(item)
        else:
            import numpy as np
            j = np.random.randint(0, self._n_seen)
            if j < self.capacity:
                self._buffer[j] = item
    def items(self):
        return self._buffer.copy()
    def __len__(self):
        return len(self._buffer)

@dataclass
class TaskMemory:
    X: list
    y: list
    reservoir: Reservoir
    def __init__(self, capacity, seed=0):
        self.X = []
        self.y = []
        self.reservoir = Reservoir(capacity, seed)
    def add(self, x, y):
        x_flat = as_default_array(x).reshape(-1)
        y_val = float(np.asarray(y).reshape(-1)[0])
        self.reservoir.add((x_flat, y_val))
        self.X = [it[0] for it in self.reservoir.items()]
        self.y = [it[1] for it in self.reservoir.items()]
    def __len__(self):
        return len(self.reservoir)

class EWC:
    def __init__(self, model, loss, lambda_=0.4, gamma=1.0):
        self.model = model
        self.loss = loss
        self.lambda_ = float(lambda_)
        self.gamma = float(gamma)
        self.F_ = None
        self.theta_star_ = None
        self._t = 0
    @property
    def n_params(self):
        return sum(p.value.size for p in self.model.params())
    def step(self, x, y):
        theta, treedef = get_flat_params(self.model)
        pred = self.model.forward(x, training=True)
        task_loss = self.loss.forward(pred, y)
        ewc_penalty = 0.0
        if self.F_ is not None and self.theta_star_ is not None:
            diff = theta - self.theta_star_
            ewc_penalty = 0.5 * self.lambda_ * jnp.sum(self.F_ * diff * diff)
        return float(task_loss + ewc_penalty)
    def after_step(self, grads):
        self._t += 1
        g2 = grads * grads
        if self.F_ is None:
            self.F_ = g2
            self.theta_star_, _ = get_flat_params(self.model)
        else:
            alpha = self.gamma * (1.0 - 1.0 / self._t)
            self.F_ = alpha * self.F_ + (1.0 / self._t) * g2
            self.theta_star_, _ = get_flat_params(self.model)
    def consolidate(self, x, y):
        theta, treedef = get_flat_params(self.model)
        n = len(x) if hasattr(x, '__len__') else 1
        F_accum = jnp.zeros(self.n_params, dtype=get_default_dtype())
        for i in range(n):
            xi = x[i] if n > 1 else x
            yi = y[i] if n > 1 else y
            pred = self.model.forward(xi, training=False)
            grads = jax.grad(lambda th: -self.loss.forward(self.model.forward(xi, training=False), yi))(theta)
            F_accum += grads * grads
        self.F_ = F_accum / n
        self.theta_star_, _ = get_flat_params(self.model)
        self._t = n
        return self
    def penalty(self, theta=None):
        if self.F_ is None or self.theta_star_ is None:
            raise RuntimeError('Call consolidate() or run at least one step first')
        th, _ = get_flat_params(self.model) if theta is None else (jnp.asarray(theta), None)
        diff = th - self.theta_star_
        return 0.5 * self.lambda_ * jnp.sum(self.F_ * diff * diff)

class GEM:
    def __init__(self, memory_per_task=256, seed=0):
        self.memory_per_task = int(memory_per_task)
        self.seed = int(seed)
        self._memories = {}
        self._rng = np.random.RandomState(seed)
    def _get_memory(self, task_id):
        if task_id not in self._memories:
            self._memories[task_id] = TaskMemory(self.memory_per_task, self.seed)
        return self._memories[task_id]
    def observe(self, x, y, task_id):
        mem = self._get_memory(task_id)
        mem.add(as_default_array(x).reshape(-1), y)
        return self
    def memory(self):
        return [(np.stack(mem.X), np.array(mem.y)) for _, mem in sorted(self._memories.items())]
    def memory_gradients(self, model, loss):
        mem_grads = []
        for task_id, mem in sorted(self._memories.items()):
            if len(mem) == 0:
                continue
            theta, treedef = get_flat_params(model)
            grads_sum = jnp.zeros(sum(p.value.size for p in model.params()))
            for x_mem, y_mem in zip(mem.X, mem.y):
                def loss_fn(th):
                    set_flat_params(model, th, treedef)
                    pred = model.forward(x_mem, training=False)
                    return loss.forward(pred, y_mem)
                grads = jax.grad(loss_fn)(theta)
                grads_sum += grads
            mem_grads.append(grads_sum / len(mem))
        return mem_grads
    def project(self, g, mem_grads):
        g = jnp.asarray(g).reshape(-1)
        if not mem_grads:
            return g
        G = jnp.stack([jnp.asarray(m).reshape(-1) for m in mem_grads], axis=0)
        dot = G @ g
        if jnp.all(dot >= 0.0):
            return g
        GGt = G @ G.T
        Gg = G @ g
        v = self._solve_dual_qp(GGt, Gg)
        return g + G.T @ v
    @staticmethod
    def _solve_dual_qp(A, b, max_iter=200, tol=1e-8):
        T = A.shape[0]
        v = jnp.zeros(T)
        eigvals = jnp.linalg.eigvalsh(A)
        lr = 1.0 / (jnp.max(eigvals) + 1e-10)
        for _ in range(max_iter):
            grad = A @ v + b
            v_new = jnp.maximum(0.0, v - lr * grad)
            if jnp.max(jnp.abs(v_new - v)) < tol:
                return v_new
            v = v_new
        return v

class AGEM:
    def __init__(self, memory_per_task=256, seed=0):
        self.memory_per_task = int(memory_per_task)
        self.seed = int(seed)
        self._memories = {}
        self._rng = np.random.RandomState(seed)
    def _get_memory(self, task_id):
        if task_id not in self._memories:
            self._memories[task_id] = TaskMemory(self.memory_per_task, self.seed)
        return self._memories[task_id]
    def observe(self, x, y, task_id):
        mem = self._get_memory(task_id)
        mem.add(as_default_array(x).reshape(-1), y)
        return self
    def memory(self):
        return [(np.stack(mem.X), np.array(mem.y)) for _, mem in sorted(self._memories.items())]
    def reference_gradient(self, model, loss):
        theta, treedef = get_flat_params(model)
        def get_grad(x_mem, y_mem):
            def loss_fn(th):
                set_flat_params(model, th, treedef)
                pred = model.forward(x_mem, training=False)
                return loss.forward(pred, y_mem)
            return jax.grad(loss_fn)(theta)
        grads_sum = jnp.zeros(sum(p.value.size for p in model.params()))
        total = 0
        for mem in self._memories.values():
            for x_mem, y_mem in zip(mem.X, mem.y):
                grads = get_grad(x_mem, y_mem)
                grads_sum += grads
                total += 1
        if total == 0:
            return jnp.zeros_like(grads_sum)
        return grads_sum / total
    def project(self, g, g_ref):
        g = jnp.asarray(g).reshape(-1)
        gr = jnp.asarray(g_ref).reshape(-1)
        dot = jnp.dot(g, gr)
        if dot >= 0.0:
            return g
        denom = jnp.dot(gr, gr)
        if denom == 0.0:
            return g
        return g - (dot / denom) * gr
