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
