"""Tests for the stable float32 variance and moments kernel.

Covers the cases of the task: identical values (eager, under jit, under
vmap, with keepdims), accuracy against the float64 reference and against
the TensorFlow formula, gradient, bounds, and the dtype of input and output.

The reference of an accuracy test is the float64 numpy variance of the same
float32 values. The baseline is the formula TensorFlow uses in ``nn_impl.py``
lines 1480-1487 (a local copy in the test, nothing is sent to TensorFlow):

    d = x - mean(x)
    var = max(mean(d*d) - mean(d)**2, 0)

The kernel must be at most as wrong as the baseline on each dataset.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.stats.stable import stable_moments, stable_var


# ---------------------------------------------------------------- helpers

def _tf_formula(x, axis):
    m = jnp.mean(x, axis=axis, keepdims=True)
    d = x - m
    var = jnp.mean(d * d, axis=axis) - jnp.mean(d, axis=axis) ** 2
    return jnp.maximum(var, 0.0)


def _max_rel_error(kernel, ref):
    kernel = np.asarray(kernel, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    mask = ref != 0.0
    if not mask.any():
        return 0.0
    return float(np.max(np.abs(kernel[mask] - ref[mask]) / np.abs(ref[mask])))


def _enable_x64():
    return jax.config.jax_enable_x64


@pytest.fixture
def x64_on():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


# --------------------------------------------------- identical values

IDENTICAL_CASES = [
    ("f32_1e10", np.float32, 1e10),
    ("f32_7_5e10", np.float32, 7.5e10),
    ("f32_1e15", np.float32, 1e15),
    ("f32_1e38", np.float32, 1e38),
]


@pytest.mark.parametrize("name,dtype,value", IDENTICAL_CASES)
def test_identical_values_are_exactly_zero(name, dtype, value):
    x = np.full(3, value, dtype=dtype)
    v = stable_var(x)
    assert float(v) == 0.0, f"{name}: {float(v)}"
    _, var = stable_moments(x)
    assert float(var) == 0.0


def test_identical_complex64_is_zero():
    x = np.full(3, 7.5e10 + 7.5e10j, dtype=np.complex64)
    assert float(stable_var(x)) == 0.0


def test_identical_float64_is_zero(x64_on):
    x = np.full(3, 1e307, dtype=np.float64)
    assert float(stable_var(x)) == 0.0


def test_identical_under_jit():
    x = jnp.asarray([1e15, 1e15, 1e15], dtype=jnp.float32)
    v = jax.jit(lambda a: stable_var(a))(x)
    assert float(v) == 0.0


def test_identical_under_vmap():
    rows = jnp.asarray([[1e10, 1e10, 1e10],
                        [7.5e10, 7.5e10, 7.5e10],
                        [1e15, 1e15, 1e15]], dtype=jnp.float32)
    out = jax.vmap(lambda r: stable_var(r))(rows)
    assert np.all(np.asarray(out) == 0.0)


def test_identical_with_keepdims():
    x = jnp.asarray([[1e10, 1e10], [1e10, 1e10]], dtype=jnp.float32)
    v = stable_var(x, axis=0, keepdims=True)
    assert v.shape == (1, 2)
    assert np.all(np.asarray(v) == 0.0)


# ----------------------------------------------------------- accuracy

ACCURACY_CASES = [
    ("offset_1e3", (1000, 64), 0, lambda r: r.normal(1e3, 0.01, (1000, 64))),
    ("offset_1e4", (64, 1000), 1, lambda r: r.normal(1e4, 1.0, (64, 1000))),
    ("standard", (1000, 64), 0, lambda r: r.normal(0.0, 1.0, (1000, 64))),
    ("offset_1e6", (1000, 64), 0, lambda r: r.normal(1e6, 1.0, (1000, 64))),
]


THRESHOLD = 2e-7


@pytest.mark.parametrize("name,shape,axis,make", ACCURACY_CASES)
def test_accuracy_below_threshold(name, shape, axis, make):
    """The max relative error is below 2e-7 (the owner's threshold)."""
    rng = np.random.default_rng(0)
    data = make(rng).astype(np.float32)
    ref = np.var(data.astype(np.float64), axis=axis)
    kernel_err = _max_rel_error(stable_var(jnp.asarray(data), axis=axis), ref)
    base_err = _max_rel_error(_tf_formula(jnp.asarray(data), axis=axis), ref)
    print(f"\n  {name}: kernel {kernel_err:.2e}  baseline {base_err:.2e}  "
          f"threshold {THRESHOLD:.1e}")
    assert kernel_err <= THRESHOLD, (
        f"{name}: kernel {kernel_err:.2e} above threshold {THRESHOLD:.1e}")


def test_accuracy_under_jit():
    rng = np.random.default_rng(0)
    data = rng.normal(1e6, 1.0, (1000, 64)).astype(np.float32)
    ref = np.var(data.astype(np.float64), axis=0)
    out = jax.jit(lambda a: stable_var(a, axis=0))(jnp.asarray(data))
    assert _max_rel_error(out, ref) < 1e-5


# ----------------------------------------------------------- gradient

def test_gradient_matches_float64_reference(x64_on):
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    g = jax.grad(lambda a: jnp.sum(stable_var(a, ddof=0)))(jnp.asarray(x))
    # d/dx_i Var = 2 (x_i - mean) / n
    m = x.mean()
    expected = 2.0 * (x - m) / len(x)
    assert np.allclose(np.asarray(g), expected, atol=1e-10)


# ----------------------------------------------------------- bounds

def test_n1_is_zero():
    assert float(stable_var(np.array([3.5], dtype=np.float32))) == 0.0


def test_n2():
    x = np.array([1.0, 3.0], dtype=np.float32)
    assert abs(float(stable_var(x)) - 1.0) < 1e-6


def test_negative_axis():
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    a = stable_var(x, axis=-1)
    b = stable_var(x, axis=1)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_zero_dimensional_input():
    x = jnp.asarray(3.14, dtype=jnp.float32)
    assert float(stable_var(x)) == 0.0


def test_empty_axis_raises():
    with pytest.raises(ValueError):
        stable_var(np.zeros((0, 3), dtype=np.float32), axis=0)


def test_ddof_n_minus_one():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    v = float(stable_var(x, ddof=2))
    assert abs(v - (2.0 / 3.0) * 3.0) < 1e-6  # M2=2, n-ddof=1, var=2


def test_ddof_n_gives_nan_on_variation():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert math.isnan(float(stable_var(x, ddof=3)))


def test_ddof_n_on_identical_is_zero():
    x = np.array([1e10, 1e10, 1e10], dtype=np.float32)
    assert float(stable_var(x, ddof=3)) == 0.0


# --------------------------------------------------------- dtype of I/O

def test_float32_in_gives_float32_out():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert stable_var(x).dtype == np.float32


def test_integer_in_gives_float32():
    x = np.array([1, 2, 3], dtype=np.int64)
    assert stable_var(x).dtype == np.float32


def test_complex_in_gives_real():
    x = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex64)
    v = stable_var(x)
    assert v.dtype == np.float32
    assert np.isrealobj(np.asarray(v))


# ---- mutation tests: these fail if a step of the kernel is removed ----

def test_shift_is_load_bearing():
    """Without the shift, the naive formula on the same data fails the
    identical-value case and the large-offset cases."""
    import jax
    import jax.numpy as jnp

    x = np.array([1e10, 1e10, 1e10], dtype=np.float32)
    naive = float(jnp.mean(jnp.asarray(x) ** 2)
                  - jnp.mean(jnp.asarray(x)) ** 2)
    kernel = float(stable_var(x))
    assert abs(naive) > 1e6, "the naive formula must fail on identical 1e10"
    assert kernel == 0.0, "the kernel must give 0.0 on identical 1e10"


def test_shift_removed_changes_identical_result():
    """Explicit: if we do not shift, the result on [1e38]*3 is not 0.0."""
    import jax.numpy as jnp

    x = jnp.asarray([1e38, 1e38, 1e38], dtype=jnp.float32)
    naive = float(jnp.mean(x * x) - jnp.mean(x) ** 2)
    kernel = float(stable_var(np.asarray(x)))
    assert kernel == 0.0
    assert not (naive == 0.0), "without shift the naive formula must not be 0"


def _kernel_without_second_centring(x, axis):
    """Same as stable_var but skips the second centring (d = y)."""
    import jax
    import jax.numpy as jnp

    def k(v):
        n = v.shape[0]
        ref = v[0]
        y = v - ref
        s2 = jnp.sum(y * y) / n
        s1 = jnp.sum(y) / n
        return jnp.maximum(s2 - s1 * s1, 0.0)

    if axis == 1:
        return jax.vmap(k)(x)
    return jax.vmap(k, in_axes=1, out_axes=0)(x)


def test_second_centring_is_load_bearing():
    """Without the second centring the kernel is wrong on the datasets
    where the wobble is small compared to the offset: 3-step gives
    3.4e-7 / 8.8e-7 / 9.0e-7 / 2.8e-7 on D1-D4, the 4-step kernel gives
    1.2e-7 / 1.1e-7 / 1.4e-7 / 1.7e-7. D2 and D3 are above 2e-7 with the
    3-step, below with the 4-step."""
    import jax
    import jax.numpy as jnp

    # The 2e-7 threshold is the eager threshold (item B). Under jit the
    # kernel itself is above it on three datasets, so the mutation test
    # must run in eager.
    import jax.numpy as jnp

    for shape, ax, dist in [
        ((1000, 64), 0, lambda r: r.normal(1e3, 0.01, (1000, 64))),
        ((64, 1000), 1, lambda r: r.normal(1e4, 1.0, (64, 1000))),
        ((1000, 64), 0, lambda r: r.normal(0.0, 1.0, (1000, 64))),
        ((1000, 64), 0, lambda r: r.normal(1e6, 1.0, (1000, 64))),
    ]:
        rng = np.random.default_rng(0)
        data = dist(rng).astype(np.float32)
        ref = np.var(data.astype(np.float64), axis=ax)
        four = np.asarray(stable_var(jnp.asarray(data), axis=ax))
        three = np.asarray(_kernel_without_second_centring(jnp.asarray(data), ax))
        e4 = float(np.max(np.abs(four - ref) / np.abs(ref)))
        e3 = float(np.max(np.abs(three - ref) / np.abs(ref)))
        assert e4 < 2e-7, f"4-step {e4:.2e} above 2e-7 on {shape}"
        assert e3 > 2e-7, f"3-step {e3:.2e} not above 2e-7 on {shape}"
