"""Tests for TestTimeCompute."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.activations import Relu
from dense_armor.utility.neural.chain import Sequential
from dense_armor.utility.neural.layers import Dense
from dense_armor.utility.neural.test_time import TestTimeCompute


def _model():
    key = jax.random.PRNGKey(0)
    return Sequential([Dense(2, 4, key), Relu(), Dense(4, 2, key)])


def test_forward_shape():
    ttc = TestTimeCompute(_model(), n_candidates=4)
    y = ttc.forward(jnp.ones((5, 2)))
    assert y.shape == (5, 2)


def test_n_candidates_1_equals_model():
    model = _model()
    ttc = TestTimeCompute(model, n_candidates=1, noise=0.0)
    x = jnp.ones((3, 2))
    y_ttc = ttc.forward(x)
    y_model = model.forward(x, training=True)
    assert np.allclose(np.asarray(y_ttc), np.asarray(y_model), atol=1e-5)


def test_backward_before_forward_raises():
    ttc = TestTimeCompute(_model(), n_candidates=2)
    with pytest.raises(RuntimeError, match="forward"):
        ttc.backward(jnp.ones((2,)))


def test_params_equals_model():
    model = _model()
    ttc = TestTimeCompute(model, n_candidates=2)
    assert len(ttc.params()) == len(model.params())


def test_invalid_n_candidates():
    with pytest.raises(ValueError, match="n_candidates"):
        TestTimeCompute(_model(), n_candidates=0)


def test_invalid_noise():
    with pytest.raises(ValueError, match="noise"):
        TestTimeCompute(_model(), noise=-0.1)

def _make_base():
    return Sequential([
        Dense(2, 4, jax.random.PRNGKey(0)),
        Relu(),
        Dense(4, 2, jax.random.PRNGKey(1)),
    ])


def test_repr_contains_params():
    ttc = TestTimeCompute(_make_base(), n_candidates=4, noise=0.01, seed=0)
    r = repr(ttc)
    assert "n_candidates=4" in r
    assert "noise=0.01" in r
    assert "seed=0" in r
    assert "verifier='confidence'" in r


def test_margin_verifier():
    ttc = TestTimeCompute(_make_base(), n_candidates=4, noise=0.1, seed=0, verifier="margin")
    y = ttc.forward(jnp.ones((2, 2)))
    assert y.shape == (2, 2)
    assert ttc.verifier == "margin"


def test_entropy_verifier():
    ttc = TestTimeCompute(_make_base(), n_candidates=4, noise=0.1, seed=0, verifier="entropy")
    y = ttc.forward(jnp.ones((2, 2)))
    assert y.shape == (2, 2)
    assert ttc.verifier == "entropy"


def test_ensemble_verifier():
    ttc = TestTimeCompute(_make_base(), n_candidates=4, noise=0.1, seed=0, verifier="ensemble")
    y = ttc.forward(jnp.ones((2, 2)))
    assert y.shape == (2, 2)
    assert ttc._selected is None


def test_backward_ensemble_raises():
    ttc = TestTimeCompute(_make_base(), n_candidates=2, noise=0.1, seed=0, verifier="ensemble")
    ttc.forward(jnp.ones((2, 2)))
    with pytest.raises(RuntimeError, match="ensemble"):
        ttc.backward(jnp.ones((2, 2)))


def test_backward_after_forward():
    ttc = TestTimeCompute(_make_base(), n_candidates=2, noise=0.1, seed=0)
    y = ttc.forward(jnp.ones((2, 2)))
    dx = ttc.backward(jnp.ones_like(y))
    assert dx.shape == y.shape


def test_invalid_verifier():
    with pytest.raises(ValueError, match="verifier"):
        TestTimeCompute(_make_base(), verifier="bogus")


