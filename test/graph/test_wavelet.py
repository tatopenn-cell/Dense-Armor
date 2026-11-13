"""Tests for the Chebyshev wavelet terms and SpectralWavelet."""

import numpy as np
import pytest

from dense_armor.core.graph import Graph, SpectralWavelet, chebyshev_wavelet_terms


def _ring(n=8):
    return Graph(n, [(i, (i + 1) % n, 1.0) for i in range(n)])


def test_even_terms_vanish_and_odd_terms_are_one_at_zero():
    even, odd = chebyshev_wavelet_terms([0.0], 3)
    assert np.allclose(even, 0.0) and np.allclose(odd, 1.0)


def test_terms_by_hand_for_orders_one_and_two():
    lam = np.linspace(0.0, 2.0, 9)
    even, odd = chebyshev_wavelet_terms(lam, 1)
    assert np.allclose(odd[0], (2.0 - lam) / 2.0)
    assert np.allclose(even[0], lam * (2.0 - lam))


def test_terms_stay_in_zero_one():
    lam = np.linspace(0.0, 2.0, 41)
    even, odd = chebyshev_wavelet_terms(lam, 4)
    assert even.min() >= -1e-12 and even.max() <= 1 + 1e-12
    assert odd.min() >= -1e-12 and odd.max() <= 1 + 1e-12


def test_responses_of_the_first_terms_on_a_ring():
    g = _ring()
    w = SpectralWavelet(g, [1.0], [1.0], [1.0])
    lam = np.sort(np.linalg.eigvalsh(g.normalized_laplacian.toarray()))
    assert np.allclose(w.scaling_response, (2.0 - lam) / 2.0, atol=1e-10)
    assert np.allclose(w.wavelet_responses[0], lam * (2.0 - lam), atol=1e-10)


def test_a_wavelet_has_no_dc_component():
    g = _ring()
    w = SpectralWavelet(g, [1.0, 0.5], [1.0, 0.3], [1.0])
    assert w.wavelet_responses[0][0] == pytest.approx(0.0, abs=1e-10)


def test_a_larger_scale_squeezes_the_wavelet():
    g = _ring()
    w = SpectralWavelet(g, [1.0], [1.0], [1.0, 2.0])
    lam = np.sort(np.linalg.eigvalsh(g.normalized_laplacian.toarray()))
    assert (w.wavelet_responses[1][lam > 1.0 + 1e-9] == 0.0).all()


def test_operators_apply_the_responses():
    g = _ring()
    w = SpectralWavelet(g, [1.0], [1.0], [1.0, 1.5])
    x = np.arange(8.0)
    assert w.scaling(x).shape == (8,)
    assert w.wavelets(x).shape == (2, 8)


def test_the_frame_sum_matches_the_responses():
    g = _ring()
    w = SpectralWavelet(g, [1.0], [1.0], [1.0])
    assert np.allclose(
        w.frame_sum(), w.scaling_response**2 + w.wavelet_responses[0] ** 2
    )


def test_bad_arguments_are_refused():
    g = _ring()
    with pytest.raises(ValueError):
        SpectralWavelet(g, [1.0], [1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        SpectralWavelet(g, [1.0], [1.0], [0.0])
    with pytest.raises(ValueError):
        chebyshev_wavelet_terms([0.0], 0)
