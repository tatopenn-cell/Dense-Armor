"""Tests for matryoshka.py."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.losses import softmax_cross_entropy_forward
from dense_armor.utility.neural.matryoshka import MatryoshkaLoss
from dense_armor.utility.optim import Adam


def _hand(weights=None):
    ml = MatryoshkaLoss([1, 2], 2, 2, jax.random.PRNGKey(0), weights=weights)
    ml.heads_[0].value = jnp.array([[1.0], [-1.0]])
    ml.heads_[1].value = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    for h in ml.heads_:
        h.grad = jnp.zeros_like(h.value)
    return ml


def test_hand_case():
    ml = _hand()
    z = jnp.array([[1.0, 0.0]])
    y = jnp.array([0])
    loss = ml.forward(z, y)
    assert abs(float(loss) - 0.44019) < 1e-4

    assert np.allclose(np.asarray(ml.logits(z, 1)), np.array([[1.0, -1.0]]))
    assert np.allclose(np.asarray(ml.logits(z, 2)), np.array([[1.0, 0.0]]))

    dz = np.asarray(ml.backward())
    assert np.allclose(dz, np.array([[-0.50735, 0.26894]]), atol=1e-4)
    assert np.allclose(
        np.asarray(ml.heads_[0].grad), np.array([[-0.11920], [0.11920]]), atol=1e-4
    )
    assert np.allclose(
        np.asarray(ml.heads_[1].grad),
        np.array([[-0.26894, 0.0], [0.26894, 0.0]]),
        atol=1e-4,
    )


def test_hand_case_weights():
    ml = _hand(weights=[2.0, 1.0])
    z = jnp.array([[1.0, 0.0]])
    y = jnp.array([0])
    loss = ml.forward(z, y)
    assert abs(float(loss) - 0.56712) < 1e-4
    dz = np.asarray(ml.backward())
    assert np.allclose(dz, np.array([[-0.74575, 0.26894]]), atol=1e-4)


def test_prefix_invariance():
    ml = _hand()
    z1 = jnp.array([[1.0, 0.0]])
    z2 = jnp.array([[1.0, 3.0]])
    y = jnp.array([0])
    ml.forward(z1, y)
    l1_m1 = ml.logits(z1, 1)
    ml.forward(z2, y)
    l2_m1 = ml.logits(z2, 1)
    assert np.allclose(np.asarray(l1_m1), np.asarray(l2_m1))


def test_gradient_against_jax_grad():
    ml = _hand()
    z = jnp.array([[1.0, 0.5]])
    y = jnp.array([0])

    def objective(z_in):
        total = 0.0
        for i, m in enumerate(ml.dims):
            logits = z_in[:, :m] @ ml.heads_[i].value.T
            total = total + ml.weights[i] * softmax_cross_entropy_forward(logits, y)
        return total

    dz_auto = jax.grad(objective)(z)
    ml.forward(z, y)
    dz_man = ml.backward()
    assert np.allclose(np.asarray(dz_auto), np.asarray(dz_man), atol=1e-6)


def test_gradient_heads_against_jax_grad():
    ml = _hand()
    z = jnp.array([[1.0, 0.5]])
    y = jnp.array([0])

    ml.forward(z, y)
    ml.backward()

    for i, m in enumerate(ml.dims):
        W0 = ml.heads_[i].value

        def objective(W, i=i):
            total = 0.0
            for j, mj in enumerate(ml.dims):
                Wj = W if j == i else ml.heads_[j].value
                logits = z[:, :mj] @ Wj.T
                total = total + ml.weights[j] * softmax_cross_entropy_forward(logits, y)
            return total

        g = jax.grad(objective)(W0)
        assert np.allclose(np.asarray(g), np.asarray(ml.heads_[i].grad), atol=1e-6), (
            f"head {m}"
        )


def test_one_sample_and_batch_give_finite_losses():
    ml = _hand()
    z_b = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    y_b = jnp.array([0, 1])

    l_b = ml.forward(z_b, y_b)
    dz_b = ml.backward()
    assert dz_b.shape == (2, 2)
    assert np.isfinite(float(l_b))

    for h in ml.heads_:
        h.grad = jnp.zeros_like(h.value)

    z_1 = jnp.array([1.0, 0.0])
    y_1 = jnp.array(0)
    l_1 = ml.forward(z_1, y_1)
    dz_1 = ml.backward()
    assert dz_1.shape == (2,)
    assert np.isfinite(float(l_1))


@pytest.mark.training
def test_learning_each_size_has_accuracy():
    rng = np.random.default_rng(0)
    n = 60
    means = np.array(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=np.float32
    )
    y = rng.integers(0, 4, n)
    X = np.zeros((n, 8), dtype=np.float32)
    X[:, :2] = means[y] + rng.normal(0.0, 0.2, (n, 2)).astype(np.float32)
    X[:, 2:] = rng.normal(0.0, 0.2, (n, 6)).astype(np.float32)

    ml = MatryoshkaLoss([2, 4, 8], 8, 4, jax.random.PRNGKey(0))
    opts = {id(h): Adam(0.02) for h in ml.params()}
    for _ in range(80):
        for h in ml.params():
            h.zero_grad()
        for i in range(n):
            ml.forward(X[i], y[i])
            ml.backward()
        for h in ml.params():
            h.update(opts[id(h)])

    accs = []
    for m in [2, 4, 8]:
        ok = 0
        for i in range(n):
            logits = ml.logits(X[i], m)
            pred = int(jnp.argmax(logits))
            if pred == int(y[i]):
                ok += 1
        accs.append(ok / n)
    print(f"accuracies: {[round(a, 3) for a in accs]}")
    assert accs[-1] >= accs[0] - 0.05


def test_value_errors_and_runtime_error():
    with pytest.raises(ValueError, match="dims must have at least one"):
        MatryoshkaLoss([], 2, 2, jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="strictly increasing"):
        MatryoshkaLoss([2, 1], 2, 2, jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="exceeds"):
        MatryoshkaLoss([1, 3], 2, 2, jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="n_classes must be at least 2"):
        MatryoshkaLoss([1], 2, 1, jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="weights has"):
        MatryoshkaLoss([1, 2], 2, 2, jax.random.PRNGKey(0), weights=[1.0])
    with pytest.raises(ValueError, match="weights must be non-negative"):
        MatryoshkaLoss([1, 2], 2, 2, jax.random.PRNGKey(0), weights=[1.0, -1.0])

    ml = MatryoshkaLoss([1, 2], 2, 2, jax.random.PRNGKey(0))
    with pytest.raises(RuntimeError, match="call forward"):
        ml.backward()


def test_repr():
    ml = _hand()
    r = repr(ml)
    assert "MatryoshkaLoss" in r
    assert "dims=[1, 2]" in r
