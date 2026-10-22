"""Tests for the small layers and the three architectures."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.gradcheck import gradcheck
from dense_armor.utility.neural.losses import (
    bce_with_logits_backward,
    bce_with_logits_forward,
)
from dense_armor.utility.neural.models import (
    AddPositions,
    LastStep,
    MeanOverTime,
    cnn,
    recurrent_net,
    transformer_encoder,
)
from dense_armor.utility.optim.adam import Adam


def test_laststep_forward_and_backward_hand():
    ls = LastStep()
    x = jnp.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    y = ls.forward(x)
    assert np.allclose(np.asarray(y), np.array([[5.0, 6.0]]))
    dx = ls.backward(jnp.array([[10.0, 20.0]]))
    assert np.allclose(
        np.asarray(dx).reshape(3, 2), np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 20.0]])
    )


def test_mean_overtime_forward_and_backward_hand():
    mt = MeanOverTime()
    x = jnp.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    y = mt.forward(x)
    assert np.allclose(np.asarray(y), np.array([[3.0, 4.0]]))
    dx = mt.backward(jnp.array([[3.0, 6.0]]))
    assert np.allclose(
        np.asarray(dx).reshape(3, 2), np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    )


def test_add_positions_hand():
    ap = AddPositions(4)
    x = jnp.zeros((1, 2, 4))
    y = np.asarray(ap.forward(x)).reshape(2, 4)
    assert np.allclose(y[0], np.array([0.0, 1.0, 0.0, 1.0]))
    assert np.allclose(
        y[1], np.array([np.sin(1), np.cos(1), np.sin(0.01), np.cos(0.01)]), atol=1e-6
    )


def test_add_positions_too_long():
    ap = AddPositions(2)
    with pytest.raises(ValueError, match="longer than max_len"):
        ap.forward(jnp.zeros((2, 3, 4)))


def test_laststep_mean_overtime_addpositions_gradchecks():
    for layer, x in [
        (LastStep(), jax.random.normal(jax.random.PRNGKey(0), (2, 4, 3))),
        (MeanOverTime(), jax.random.normal(jax.random.PRNGKey(0), (2, 4, 3))),
        (AddPositions(8), jax.random.normal(jax.random.PRNGKey(0), (2, 4, 8))),
    ]:

        def f(xx, ly=layer):
            ly.x_ = None
            return ly.forward(xx)

        def b(xx, y, g, ly=layer):
            ly.x_ = None
            ly.forward(xx)
            return ly.backward(g)

        assert gradcheck(f, b, x) < 1e-4


def test_cnn_param_count():
    m = cnn((8, 8, 1), [2], 3, jax.random.PRNGKey(0))
    assert len(m.params()) == 4


def test_cnn_no_batch_and_batch():
    m = cnn((8, 8, 1), [2], 2, jax.random.PRNGKey(0))
    y1 = m.forward(jnp.ones((8, 8, 1)))
    yb = m.forward(jnp.ones((4, 8, 8, 1)))
    assert y1.shape == (2,)
    assert yb.shape == (4, 2)
    assert np.allclose(np.asarray(y1), np.asarray(yb[0]))


def test_cnn_x_gradcheck():
    m = cnn((6, 6, 1), [2], 2, jax.random.PRNGKey(0), kernel=3)
    x = jax.random.normal(jax.random.PRNGKey(1), (6, 6, 1))

    def f(xx):
        return m.forward(xx)

    def b(xx, y, g):
        return m.backward(g)

    assert gradcheck(f, b, x) < 1e-3


def test_cnn_pooling_too_small():
    with pytest.raises(ValueError, match="pooling reduces the side below 1"):
        cnn((2, 2, 1), [2, 2], 3, jax.random.PRNGKey(0))


def test_recurrent_net_param_counts():
    for cell, expected in [("lstm", 17), ("gru", 11), ("rnn", 5)]:
        m = recurrent_net(3, 4, 2, jax.random.PRNGKey(0), cell=cell)
        assert len(m.params()) == expected, cell


def test_recurrent_net_bad_cell_raises():
    with pytest.raises(ValueError, match="cell must be"):
        recurrent_net(3, 4, 2, jax.random.PRNGKey(0), cell="foo")


def test_recurrent_net_bad_layers_raises():
    with pytest.raises(ValueError, match="n_layers must be"):
        recurrent_net(3, 4, 2, jax.random.PRNGKey(0), n_layers=0)


def test_recurrent_net_x_gradcheck():
    m = recurrent_net(3, 4, 2, jax.random.PRNGKey(0), cell="gru")
    x = jax.random.normal(jax.random.PRNGKey(1), (4, 3))

    def f(xx):
        return m.forward(xx)

    def b(xx, y, g):
        return m.backward(g)

    assert gradcheck(f, b, x) < 1e-3


def test_transformer_encoder_shapes():
    m = transformer_encoder(4, 8, 2, 16, 2, 3, jax.random.PRNGKey(0), 16)
    y = m.forward(jnp.ones((5, 4)))
    assert y.shape == (3,)


def test_transformer_encoder_bad_depth_raises():
    with pytest.raises(ValueError, match="depth must be at least 1"):
        transformer_encoder(4, 8, 2, 16, 0, 3, jax.random.PRNGKey(0), 16)


def test_transformer_encoder_x_gradcheck():
    m = transformer_encoder(2, 4, 2, 8, 1, 2, jax.random.PRNGKey(0), 16)
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 2))

    def f(xx):
        return m.forward(xx)

    def b(xx, y, g):
        return m.backward(g)

    assert gradcheck(f, b, x) < 1e-3


def _train_classifier(model, X, y, steps, lr, seed=0, batch_size=8):
    opts = {id(p): Adam(lr=lr) for p in model.params()}
    rng = np.random.default_rng(seed)
    losses = []
    n_batches = max(1, len(X) // batch_size)
    for _ in range(steps):
        model.zero_grad()
        perm = rng.permutation(len(X))
        total = 0.0
        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            Xb = jnp.asarray(X[idx])
            yb = jnp.asarray(y[idx]).reshape(-1, 1)
            logits = model.forward(Xb).reshape(-1, 1)
            total += float(bce_with_logits_forward(logits, yb))
            g = bce_with_logits_backward(logits, yb) / len(idx)
            model.backward(g)
        for p in model.params():
            p.update(opts[id(p)])
        losses.append(total / n_batches)
    return losses


def _accuracy_classifier(model, X, y):
    ok = 0
    for xi, yi in zip(X, y):
        logit = float(model.forward(jnp.asarray(xi))[0])
        pred = 1.0 if logit > 0.0 else 0.0
        if pred == float(yi):
            ok += 1
    return ok / len(X)


def _bar_images(n, rng, noise=0.1):
    X = np.zeros((n, 8, 8, 1), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)
    for i in range(n):
        X[i] += rng.normal(0.0, noise, (8, 8, 1))
        col = rng.integers(0, 8)
        row = rng.integers(0, 8)
        if i % 2 == 0:
            X[i, :, col, 0] += 1.0
            y[i] = 0.0
        else:
            X[i, row, :, 0] += 1.0
            y[i] = 1.0
    return X, y


@pytest.mark.slow
def test_cnn_separates_vertical_from_horizontal():
    rng = np.random.default_rng(0)
    X, y = _bar_images(80, rng)
    accs = []
    for seed in range(5):
        m = cnn((8, 8, 1), [4], 1, jax.random.PRNGKey(seed))
        _train_classifier(m, X, y, 60, 0.01, seed=seed)
        accs.append(_accuracy_classifier(m, X, y))
    print(f"cnn accuracies: {accs}")
    n_ok = sum(1 for a in accs if a >= 0.95)
    assert n_ok >= 4, f"only {n_ok} of 5 seeds reached 0.95: {accs}"


def _first_sign_sequences(n, rng, t=10):
    X = rng.normal(0.0, 0.3, (n, t, 1)).astype(np.float32)
    y = np.zeros(n, dtype=np.float32)
    for i in range(n):
        s = 1.0 if i % 2 == 0 else -1.0
        X[i, 0, 0] = s
        y[i] = 0.0 if s > 0 else 1.0
    return X, y


@pytest.mark.slow
def test_recurrent_net_lstm_and_gru():
    rng = np.random.default_rng(1)
    X, y = _first_sign_sequences(80, rng)
    for cell in ("lstm", "gru"):
        accs = []
        for seed in range(5):
            m = recurrent_net(1, 4, 1, jax.random.PRNGKey(seed), cell=cell)
            _train_classifier(m, X, y, 60, 0.01, seed=seed)
            accs.append(_accuracy_classifier(m, X, y))
        print(f"{cell} accuracies: {accs}")
        n_ok = sum(1 for a in accs if a >= 0.9)
        assert n_ok >= 4, f"{cell}: only {n_ok} of 5 reached 0.9: {accs}"


@pytest.mark.slow
def test_transformer_encoder_learns_first_sign():
    rng = np.random.default_rng(2)
    X, y = _first_sign_sequences(80, rng)
    accs = []
    for seed in range(5):
        m = transformer_encoder(
            1, 4, 2, 8, 1, 1, jax.random.PRNGKey(seed), 16, causal=False
        )
        _train_classifier(m, X, y, 60, 0.01, seed=seed)
        accs.append(_accuracy_classifier(m, X, y))
    print(f"transformer accuracies: {accs}")
    n_ok = sum(1 for a in accs if a >= 0.9)
    assert n_ok >= 4, f"only {n_ok} of 5 reached 0.9: {accs}"


def test_small_layers_repr_and_empty_params():
    assert repr(LastStep()) == "LastStep()"
    assert repr(MeanOverTime()) == "MeanOverTime()"
    assert repr(AddPositions(8)) == "AddPositions(8)"
    assert LastStep().params() == []
    assert MeanOverTime().params() == []
    assert AddPositions(8).params() == []


def test_small_layers_backward_before_forward_raises():
    with pytest.raises(RuntimeError, match="call forward"):
        LastStep().backward(jnp.ones((2, 3)))
    with pytest.raises(RuntimeError, match="call forward"):
        MeanOverTime().backward(jnp.ones((2, 3)))
    with pytest.raises(RuntimeError, match="call forward"):
        AddPositions(8).backward(jnp.ones((2, 4, 8)))
