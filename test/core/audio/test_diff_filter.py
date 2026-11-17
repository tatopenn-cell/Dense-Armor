import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.core.audio import (
    AllPole,
    Biquad,
    JaxBiquad,
    jax_allpole_time_varying,
    jax_biquad,
    jax_lowpass,
    lowpass,
)
from dense_armor.core.audio.allpole import reflection_to_lpc
from dense_armor.core.audio.diff_filter import _allpole_scan
from dense_armor.core.deep.neural.chain import Sequential
from dense_armor.core.deep.neural.conv import Conv1D
from dense_armor.core.deep.neural.losses import mse_backward, mse_forward


@pytest.fixture
def f32():
    """JAX in its default 32 bits: other test modules switch x64 on for the whole session."""
    old = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", old)


def _noise(n, seed=0):
    return np.random.default_rng(seed).standard_normal(n)


def _central(f, v, eps):
    """Central finite differences of ``f`` (a scalar) with respect to each entry of ``v``."""
    v = np.asarray(v, dtype=float)
    out = np.empty(v.shape)
    for idx in np.ndindex(*v.shape):
        p, m = v.copy(), v.copy()
        p[idx] += eps
        m[idx] -= eps
        out[idx] = (float(f(jnp.asarray(p))) - float(f(jnp.asarray(m)))) / (2.0 * eps)
    return out


COMBOS = [
    (1000.0, 8000.0, 0.707),
    (200.0, 44100.0, 2.0),
    (3000.0, 8000.0, 0.5),
    (50.0, 1000.0, 5.0),
    (7000.0, 16000.0, 1.0),
]


@pytest.mark.parametrize("f0, sr, q", COMBOS)
def test_jax_lowpass_equals_the_numpy_lowpass_in_float64(x64, f0, sr, q):
    b, a = jax_lowpass(f0, sr, q)
    bn, an = lowpass(f0, sr, q)
    assert b.shape == (3,) and a.shape == (3,)
    assert np.max(np.abs(np.asarray(b) - bn)) < 1e-10
    assert np.max(np.abs(np.asarray(a) - an)) < 1e-10
    assert float(a[0]) == 1.0


def test_jax_lowpass_in_float32_is_within_float32_rounding_of_numpy(f32):
    b, a = jax_lowpass(1000.0, 8000.0, 0.707)
    bn, an = lowpass(1000.0, 8000.0, 0.707)
    assert b.dtype == jnp.float32
    assert np.max(np.abs(np.asarray(b) - bn)) < 1e-6
    assert np.max(np.abs(np.asarray(a) - an)) < 1e-6


@pytest.mark.parametrize("f0, sr, q", COMBOS)
def test_the_gradient_of_the_lowpass_wrt_f0_is_the_finite_difference(x64, f0, sr, q):
    jac = jax.jacobian(lambda f: jnp.concatenate(jax_lowpass(f, sr, q)))(f0)
    eps = 1e-5
    fd = (
        jnp.concatenate(jax_lowpass(f0 + eps, sr, q))
        - jnp.concatenate(jax_lowpass(f0 - eps, sr, q))
    ) / (2 * eps)
    # a[0] is the constant 1: its exact derivative is 0 and the difference is rounding noise.
    assert np.allclose(np.asarray(jac), np.asarray(fd), rtol=1e-4, atol=1e-9)


def test_the_gradient_of_the_lowpass_reaches_sr_and_q(x64):
    for argnum in (1, 2):
        jac = jax.jacobian(lambda *p: jnp.concatenate(jax_lowpass(*p)), argnum)(
            1000.0, 8000.0, 0.707
        )
        assert np.all(np.isfinite(np.asarray(jac)))
        assert np.max(np.abs(np.asarray(jac))) > 0.0


def test_the_lowpass_refuses_a_cutoff_outside_the_band_and_a_bad_q():
    with pytest.raises(ValueError, match="f0 must be in"):
        jax_lowpass(4000.0, 8000.0)
    with pytest.raises(ValueError, match="f0 must be in"):
        jax_lowpass(0.0, 8000.0)
    with pytest.raises(ValueError, match="q must be positive"):
        jax_lowpass(1000.0, 8000.0, 0.0)


def test_the_lowpass_runs_under_jit_and_grad():
    f = jax.jit(lambda f0: jax_lowpass(f0, 8000.0, 0.707)[0][0])
    assert np.isfinite(float(f(1000.0)))
    assert np.isfinite(float(jax.grad(f)(1000.0)))


def test_jax_biquad_equals_the_numpy_biquad_in_float64(x64):
    x = _noise(1000)
    for f0, sr, q in COMBOS:
        b, a = lowpass(f0, sr, q)
        y = np.asarray(JaxBiquad(b, a).forward(x))
        assert np.max(np.abs(y - Biquad(b, a).update_many(x))) < 1e-10


def test_jax_biquad_in_float32_is_within_float32_rounding_of_numpy(f32):
    x = _noise(1000)
    b, a = lowpass(1000.0, 8000.0, 0.707)
    y = np.asarray(JaxBiquad(b, a).forward(x))
    assert y.dtype == np.float32
    assert np.max(np.abs(y - Biquad(b, a).update_many(x))) < 1e-5


def test_the_gradient_of_the_biquad_wrt_b_and_a_is_the_finite_difference(x64):
    x, g = _noise(300), _noise(300, 1)
    b, a = lowpass(1000.0, 8000.0, 0.707)

    def loss(b_, a_):
        return jnp.sum(g * jax_biquad(b_, a_, x))

    gb, ga = jax.grad(loss, (0, 1))(jnp.asarray(b), jnp.asarray(a))
    fd_b = _central(lambda v: loss(v, a), b, 1e-6)
    fd_a = _central(lambda v: loss(b, v), a, 1e-6)
    assert np.max(np.abs(np.asarray(gb) - fd_b) / (np.abs(fd_b) + 1e-12)) < 1e-4
    # a[0] is divided out, so the finite difference of a[0] is the gradient through that division.
    assert np.max(np.abs(np.asarray(ga) - fd_a) / (np.abs(fd_a) + 1e-12)) < 1e-4


def test_the_gradient_of_the_biquad_is_finite_over_1000_samples(x64):
    x = _noise(1000)
    b, a = lowpass(1000.0, 8000.0, 0.707)
    f = JaxBiquad(b, a)
    y = f.forward(x)
    dx = f.backward(jnp.ones_like(y))
    for v in (y, dx, f.b.grad, f.a.grad):
        assert bool(jnp.all(jnp.isfinite(v)))
    assert float(jnp.max(jnp.abs(f.b.grad))) > 0.0


def test_the_biquad_layer_accumulates_the_gradient_and_zero_grad_clears_it():
    b, a = lowpass(1000.0, 8000.0, 0.707)
    f = JaxBiquad(b, a)
    x = jnp.asarray(_noise(50))
    f.forward(x)
    f.backward(jnp.ones(50))
    once = np.asarray(f.b.grad).copy()
    f.forward(x)
    f.backward(jnp.ones(50))
    assert np.allclose(np.asarray(f.b.grad), 2 * once, rtol=1e-5)
    f.b.zero_grad()
    assert float(jnp.max(jnp.abs(f.b.grad))) == 0.0
    assert [p is q for p, q in zip(f.params(), (f.b, f.a))] == [True, True]
    assert repr(f) == "JaxBiquad(b, a)"


def test_the_biquad_filters_each_channel_and_each_batch_item_alone(x64):
    b, a = lowpass(1000.0, 8000.0, 0.707)
    x = _noise(200 * 6).reshape(2, 200, 3)
    y = np.asarray(JaxBiquad(b, a).forward(x))
    assert y.shape == x.shape
    for i in range(2):
        for c in range(3):
            assert (
                np.max(np.abs(y[i, :, c] - Biquad(b, a).update_many(x[i, :, c])))
                < 1e-10
            )
    y2 = np.asarray(JaxBiquad(b, a).forward(x[0]))
    assert np.allclose(y2, y[0])


def test_the_biquad_divides_by_a0_like_the_numpy_one(x64):
    b, a = lowpass(1000.0, 8000.0, 0.707)
    x = _noise(100)
    y = np.asarray(jax_biquad(2 * b, 2 * a, x))
    assert np.max(np.abs(y - Biquad(b, a).update_many(x))) < 1e-10


def test_the_biquad_refuses_wrong_coefficients_and_backward_before_forward():
    with pytest.raises(ValueError, match="three coefficients"):
        JaxBiquad([1.0, 0.0], [1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="a\\[0\\] must not be zero"):
        JaxBiquad([1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    with pytest.raises(RuntimeError, match="forward"):
        JaxBiquad([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]).backward(jnp.ones(3))


def test_the_allpole_with_a_constant_a_equals_the_numpy_allpole(x64):
    x = _noise(500)
    for a in ([-0.5], [-0.5, 0.2, -0.1], [0.3, -0.2, 0.1, -0.05]):
        y = np.asarray(jax_allpole_time_varying(a, x))
        assert np.max(np.abs(y - AllPole(a).update_many(x))) < 1e-10
        full = np.tile(np.asarray(a), (500, 1))
        assert np.array_equal(np.asarray(jax_allpole_time_varying(full, x)), y)


def test_the_allpole_with_a_varying_a_equals_the_numpy_allpole_with_coeffs(x64):
    x = _noise(300)
    base = np.array([-0.5, 0.2, -0.1])
    a = base[None, :] + 0.05 * np.sin(np.linspace(0, 6, 300))[:, None]
    y = np.asarray(jax_allpole_time_varying(a, x))
    assert np.max(np.abs(y - AllPole(base).update_many(x, a))) < 1e-10


def test_the_gradient_of_the_allpole_wrt_a_is_the_finite_difference(x64):
    x, g = _noise(120), _noise(120, 1)
    base = np.array([-0.5, 0.2, -0.1])
    a = base[None, :] + 0.05 * np.sin(np.linspace(0, 6, 120))[:, None]

    def loss(a_):
        return jnp.sum(g * jax_allpole_time_varying(a_, x))

    ga = np.asarray(jax.grad(loss)(jnp.asarray(a)))
    fd = _central(loss, a, 1e-6)
    assert np.max(np.abs(ga - fd) / (np.abs(fd) + 1e-9)) < 1e-4


def test_the_gradient_of_the_allpole_wrt_x_is_the_finite_difference(x64):
    x, g = _noise(120), _noise(120, 1)
    a = np.array([-0.5, 0.2, -0.1])

    def loss(x_):
        return jnp.sum(g * jax_allpole_time_varying(a, x_))

    gx = np.asarray(jax.grad(loss)(jnp.asarray(x)))
    fd = _central(loss, x, 1e-6)
    assert np.max(np.abs(gx - fd) / (np.abs(fd) + 1e-9)) < 1e-4


def test_the_backward_of_the_paper_equals_the_gradient_through_the_scan(x64):
    x, g = _noise(200), _noise(200, 1)
    a = (
        np.array([-0.5, 0.2, -0.1])[None, :]
        + 0.05 * np.sin(np.linspace(0, 6, 200))[:, None]
    )
    ours = jax.grad(
        lambda a_, x_: jnp.sum(g * jax_allpole_time_varying(a_, x_)), (0, 1)
    )(jnp.asarray(a), jnp.asarray(x))
    plain = jax.grad(lambda a_, x_: jnp.sum(g * _allpole_scan(a_, x_)), (0, 1))(
        jnp.asarray(a), jnp.asarray(x)
    )
    for p, q in zip(ours, plain):
        assert float(jnp.max(jnp.abs(p - q))) < 1e-10


def test_the_allpole_stays_bounded_with_poles_inside_the_unit_circle_for_five_trajectories(
    x64,
):
    n = 1000
    t = np.linspace(0.0, 1.0, n)
    ks = [
        np.stack(
            [0.5 * np.sin(2 * np.pi * 3 * t), 0.4 * np.cos(2 * np.pi * 2 * t)], axis=1
        ),
        np.stack(
            [0.9 * np.sin(2 * np.pi * t), 0.9 * np.sin(2 * np.pi * 0.5 * t)], axis=1
        ),
        np.stack([0.8 * np.ones(n), -0.7 * np.ones(n)], axis=1),
        np.stack([0.95 * np.cos(2 * np.pi * 5 * t), 0.3 * t], axis=1),
        np.stack([-0.6 * t, 0.85 * np.sin(2 * np.pi * 8 * t)], axis=1),
    ]
    x = _noise(n)
    for k in ks:
        a = np.stack([reflection_to_lpc(row) for row in k])
        for row in a:
            assert np.all(np.abs(np.roots(np.r_[1.0, row])) < 1.0)
        y = jax_allpole_time_varying(a, x)
        assert bool(jnp.all(jnp.isfinite(y)))
        assert float(jnp.max(jnp.abs(y))) < 1e3
        ga = jax.grad(lambda a_: jnp.sum(jax_allpole_time_varying(a_, x) ** 2))(
            jnp.asarray(a)
        )
        assert bool(jnp.all(jnp.isfinite(ga)))


def test_the_allpole_refuses_a_wrong_shape():
    with pytest.raises(ValueError, match="x must be 1D"):
        jax_allpole_time_varying([-0.5], jnp.ones((3, 2)))
    with pytest.raises(ValueError, match="a must be"):
        jax_allpole_time_varying(jnp.ones((4, 2)), jnp.ones(5))
    with pytest.raises(ValueError, match="a must be"):
        jax_allpole_time_varying(jnp.ones(0), jnp.ones(5))


def _model():
    b, a = lowpass(1000.0, 8000.0, 0.707)
    return Sequential(
        [Conv1D(1, 1, 3, jax.random.PRNGKey(0), causal=True), JaxBiquad(b, a)]
    )


def test_the_gradient_reaches_the_biquad_coefficients_through_a_neural_model():
    model = _model()
    x = jnp.asarray(_noise(200), dtype=jnp.float32)[:, None]
    target = jnp.asarray(_noise(200, 1), dtype=jnp.float32)[:, None]
    pred = model.forward(x)
    model.backward(mse_backward(pred, target))
    conv, filt = model.layers
    assert float(jnp.max(jnp.abs(filt.b.grad))) > 1e-6
    assert float(jnp.max(jnp.abs(filt.a.grad))) > 1e-6
    assert float(jnp.max(jnp.abs(conv.W.grad))) > 1e-6
    assert len(model.params()) == 4


def test_the_biquad_gradient_through_the_model_is_the_finite_difference(x64):
    model = _model()
    x = jnp.asarray(_noise(100))[:, None]
    target = jnp.asarray(_noise(100, 1))[:, None]
    filt = model.layers[1]
    model.backward(mse_backward(model.forward(x), target))
    grad = np.asarray(filt.b.grad)

    def loss(v):
        filt.b.value = v
        return mse_forward(model.forward(x), target)

    fd = _central(loss, np.asarray(filt.b.value), 1e-6)
    assert np.max(np.abs(grad - fd) / (np.abs(fd) + 1e-12)) < 1e-4


def test_a_few_steps_of_gradient_descent_bring_the_biquad_towards_its_target(x64):
    x = jnp.asarray(_noise(400))
    tb, ta = lowpass(1000.0, 8000.0, 0.707)
    target = jax_biquad(tb, ta, x)
    b0, a0 = lowpass(2000.0, 8000.0, 0.707)
    f = JaxBiquad(b0, a0)
    first = float(mse_forward(f.forward(x), target))
    for _ in range(200):
        f.b.zero_grad()
        f.a.zero_grad()
        f.backward(mse_backward(f.forward(x), target))
        f.b.value = f.b.value - 0.05 * f.b.grad
        f.a.value = f.a.value - 0.05 * f.a.grad
    last = float(mse_forward(f.forward(x), target))
    assert last < 0.1 * first
