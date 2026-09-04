# -*- coding: utf-8 -*-
"""
test/test_curvature.py
=========================
Tests for utility/curvature.py's curvature(), including the `scale`
parameter added after a real finding on SO-101 joint data (Dense-Evolution-
Discovery, elbow_flex): without a scale tied to real physical units, the
function saturates to ~1.0 within ~5 raw units of any reference, making it
a near-binary near/far indicator rather than a graded proximity signal in
degrees (correlation with raw distance 0.235 at scale=1.0, vs 0.932 at a
physically meaningful scale=15.0 -- see the function's own docstring).
"""
import jax.numpy as jnp
import numpy as np

from dense_armor.utility.curvature import curvature


def test_zero_distance_returns_zero():
    x = jnp.array([0.0, 0.0])
    assert float(curvature(x, x)) < 1e-2


def test_default_scale_matches_original_unscaled_behavior():
    # Locks in backward compatibility for Orca's existing call site
    # (orca.py calls curvature(fh, c_chunk) with no scale/delta), which
    # must keep computing sin(atan(2*|x-ref|)) exactly as before.
    a = jnp.array([5.0])
    b = jnp.array([0.0])
    d = 2.0 * float(jnp.abs(a - b)[0])
    expected = np.sqrt(d ** 2) / np.sqrt(1.0 + d ** 2)
    assert abs(float(curvature(a, b)) - expected) < 1e-4


def test_is_bounded_in_zero_one():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = jnp.array(rng.normal(scale=50.0, size=6))
        b = jnp.array(rng.normal(scale=50.0, size=6))
        v = float(curvature(a, b))
        assert 0.0 <= v < 1.0


def test_monotonic_in_raw_distance_regardless_of_scale():
    ref = jnp.array([0.0])
    dists = [0.0, 1.0, 5.0, 20.0, 80.0]
    for scale in (1.0, 15.0, 50.0):
        vals = [float(curvature(jnp.array([d]), ref, scale)) for d in dists]
        assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))


def test_small_scale_saturates_faster_than_large_scale():
    # A small scale (narrow warning zone) reaches near-saturation at a
    # shorter raw distance than a large scale (wide warning zone).
    d = jnp.array([10.0])
    ref = jnp.array([0.0])
    v_small_scale = float(curvature(d, ref, 1.0))
    v_large_scale = float(curvature(d, ref, 50.0))
    assert v_small_scale > v_large_scale


def test_scaled_curvature_correlates_better_with_real_elbow_flex_distance():
    # Reproduces the real finding (Dense-Evolution-Discovery,
    # curvature_joint_limits_check.py) on synthetic data with the same
    # shape: a distance-to-limit trajectory spanning ~0 to ~67 (degrees),
    # matching real SO-101 elbow_flex episode 0. A physically meaningful
    # scale must correlate with raw distance better than the unscaled
    # default over this real-shaped range.
    rng = np.random.default_rng(1)
    d = np.abs(rng.normal(loc=25.0, scale=20.0, size=300))
    d = np.clip(d, 0.0, 67.0)
    ref = jnp.zeros(1)
    c_default = np.array([float(curvature(jnp.array([x]), ref)) for x in d])
    c_scaled = np.array([float(curvature(jnp.array([x]), ref, 15.0)) for x in d])
    corr_default = np.corrcoef(c_default, d)[0, 1]
    corr_scaled = np.corrcoef(c_scaled, d)[0, 1]
    assert corr_scaled > corr_default
