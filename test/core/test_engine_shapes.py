"""AdaptiveSignalStabilizer: every tensor geometry, empty inputs, the mask and the 1D filter."""

import numpy as np
import pytest

from dense_armor.utility.engine import AdaptiveSignalStabilizer


def _batch(shape, seed=0):
    x = np.random.default_rng(seed).normal(size=shape)
    x.flat[7] = 25.0
    return x


@pytest.fixture(scope="module")
def stab():
    return AdaptiveSignalStabilizer()


@pytest.mark.parametrize("method", ["filter_batch_scenarios", "filter_batch_scenarios_independent"])
@pytest.mark.parametrize("shape", [(3, 16), (3, 4, 6), (2, 3, 4, 5)])
def test_every_geometry_comes_back_in_the_same_shape(stab, method, shape):
    x = _batch(shape)
    y = getattr(stab, method)(x)
    assert y.shape == shape
    assert np.all(np.isfinite(y))


@pytest.mark.parametrize("method", ["filter_batch_scenarios", "filter_batch_scenarios_independent"])
def test_an_unsupported_geometry_is_rejected(stab, method):
    with pytest.raises(ValueError, match="tensor geometry not supported"):
        getattr(stab, method)(_batch((2, 2, 2, 2, 2)))


@pytest.mark.parametrize("method", ["filter_batch_scenarios", "filter_batch_scenarios_independent"])
def test_an_empty_batch_gives_an_empty_batch(stab, method):
    out = getattr(stab, method)(np.zeros((0, 8)))
    assert out.shape == (0, 8)


@pytest.mark.parametrize("method", ["filter_batch_scenarios", "filter_batch_scenarios_independent"])
def test_a_hard_clamp_mask_changes_the_flagged_elements(stab, method):
    x = _batch((2, 3, 8))
    mask = np.zeros(x.shape, dtype=bool)
    mask[:, :, 3:5] = True
    plain = getattr(stab, method)(x)
    clamped = getattr(stab, method)(x, hard_clamp_mask=mask)
    assert clamped.shape == plain.shape
    assert not np.array_equal(clamped, plain)


def test_the_default_mask_is_the_all_false_mask(stab):
    x = _batch((2, 12))
    explicit = stab.filter_batch_scenarios_independent(x, hard_clamp_mask=np.zeros(x.shape, dtype=bool))
    assert np.array_equal(stab.filter_batch_scenarios_independent(x), explicit)


def test_calibration_of_an_empty_batch_changes_nothing():
    s = AdaptiveSignalStabilizer()
    before = (s.dyn_thr, s.dyn_dmp, s.dyn_alp, s.noise_scalar)
    s.calibrate_macro_context(np.zeros((0, 4)))
    assert (s.dyn_thr, s.dyn_dmp, s.dyn_alp, s.noise_scalar) == before


def test_calibration_of_a_single_sample_sets_only_the_noise_scalar():
    s = AdaptiveSignalStabilizer()
    before = (s.dyn_thr, s.dyn_dmp, s.dyn_alp)
    s.calibrate_macro_context(np.array([[2.0]]))
    assert s.noise_scalar == pytest.approx(1.0 / (1.0 + 0.0))
    assert (s.dyn_thr, s.dyn_dmp, s.dyn_alp) == before


def test_the_legacy_stream_filter_follows_a_series_and_damps_a_spike():
    s = AdaptiveSignalStabilizer()
    x = np.sin(np.linspace(0.0, 6.0, 80))
    x[40] = 30.0
    y = s.filter_data_stream(x)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))
    assert abs(y[40]) < abs(x[40])


def test_the_legacy_stream_filter_returns_an_empty_series_for_an_empty_series():
    assert AdaptiveSignalStabilizer().filter_data_stream(np.zeros(0)).shape == (0,)
