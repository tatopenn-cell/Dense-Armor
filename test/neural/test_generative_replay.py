"""Tests for DeepGenerativeReplay (item D of task N09)."""

import jax.numpy as jnp
import numpy as np

from dense_armor.utility.neural.generative_replay import DeepGenerativeReplay


def _blob_a(n: int = 100, seed: int = 0):
    rng = np.random.RandomState(seed)
    x = rng.randn(n, 2) * 0.5 + np.array([1.0, 1.0])
    y = np.zeros(n, dtype=np.int32)
    return jnp.asarray(x, dtype=jnp.float32), jnp.asarray(y)


def _blob_b(n: int = 100, seed: int = 1):
    rng = np.random.RandomState(seed)
    x = rng.randn(n, 2) * 0.5 + np.array([-1.0, -1.0])
    y = np.ones(n, dtype=np.int32)
    return jnp.asarray(x, dtype=jnp.float32), jnp.asarray(y)


def test_gradient_reaches_solver():
    dgr = DeepGenerativeReplay(n_x=2, n_y=2, dim_z=4, hidden=8, n_fake=16, seed=0)
    xa, ya = _blob_a()
    dgr.train_task(xa, ya, n_epochs=2)
    before = [np.asarray(p.value).copy() for p in dgr.solver.params()]
    xb, yb = _blob_b()
    dgr.train_task(xb, yb, n_epochs=2)
    after = [np.asarray(p.value).copy() for p in dgr.solver.params()]
    diff = sum(float(np.sum(np.abs(a - b))) for a, b in zip(after, before))
    assert diff > 1e-6, f"solver unchanged after fake batch, diff={diff}"


def test_n_fake_per_task():
    dgr = DeepGenerativeReplay(n_x=2, n_y=2, dim_z=4, hidden=8, n_fake=16, seed=0)
    xa, ya = _blob_a()
    dgr.train_task(xa, ya, n_epochs=1)
    assert len(dgr.generators) == 1
    assert dgr.n_tasks_ == 1
    xb, yb = _blob_b()
    dgr.train_task(xb, yb, n_epochs=1)
    assert len(dgr.generators) == 2
    assert dgr.n_tasks_ == 2
    xf, yf = dgr.sample(dgr.generators[0], 32)
    assert xf.shape == (32, 2)
    assert yf.shape == (32,)


def test_generator_state_grows_linearly():
    dgr = DeepGenerativeReplay(n_x=2, n_y=2, dim_z=4, hidden=8, n_fake=16, seed=0)
    xa, ya = _blob_a()
    dgr.train_task(xa, ya, n_epochs=1)
    size_one = sum(int(np.asarray(p.value).size) for p in dgr.generators[0].params())
    xb, yb = _blob_b()
    dgr.train_task(xb, yb, n_epochs=1)
    size_two = sum(
        int(np.asarray(p.value).size) for g in dgr.generators for p in g.params()
    )
    assert size_two == 2 * size_one


def test_two_task_retention():
    dgr = DeepGenerativeReplay(
        n_x=2, n_y=2, dim_z=4, hidden=16, n_fake=32, seed=0, lr=0.05
    )
    xa, ya = _blob_a(n=100)
    xb, yb = _blob_b(n=100)
    dgr.train_task(xa, ya, n_epochs=20, batch_size=32)
    pred_a_before = np.argmax(np.asarray(dgr.predict(xa)), axis=-1)
    acc_a_before = float((pred_a_before == np.asarray(ya)).mean())
    dgr.train_task(xb, yb, n_epochs=20, batch_size=32)
    pred_a_after = np.argmax(np.asarray(dgr.predict(xa)), axis=-1)
    acc_a_after = float((pred_a_after == np.asarray(ya)).mean())
    assert acc_a_after >= acc_a_before - 0.3, (
        f"replay failed: acc_a_before={acc_a_before:.2f} "
        f"acc_a_after={acc_a_after:.2f}"
    )
