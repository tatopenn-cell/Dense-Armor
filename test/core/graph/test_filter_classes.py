"""Tests for GraphFourier, SpectralFilter and ChebyshevFilter."""

import numpy as np
import pytest
from scipy.linalg import expm

from dense_armor.core.graph import (
    ChebyshevFilter,
    Graph,
    GraphFourier,
    SpectralFilter,
)


def _path(n=10):
    return Graph(n, [(i, i + 1, 1.0) for i in range(n - 1)])


def _x():
    x = np.zeros(10)
    x[4] = 5.0
    return x


def test_eigenvalues_of_the_path_in_closed_form():
    f = GraphFourier(_path())
    k = np.arange(10)
    assert np.allclose(f.eigenvalues, 2 - 2 * np.cos(np.pi * k / 10), atol=1e-10)


def test_transform_keeps_the_energy_and_inverts():
    f = GraphFourier(_path())
    x = _x()
    xh = f.transform(x)
    assert xh @ xh == pytest.approx(x @ x)
    assert np.allclose(f.inverse(xh), x)


def test_normalized_eigenvalues_lie_in_zero_two():
    f = GraphFourier(_path(), normalized=True)
    assert f.eigenvalues.min() > -1e-10 and f.eigenvalues.max() < 2 + 1e-10


def test_heat_filter_is_the_matrix_exponential():
    g = _path()
    y = SpectralFilter.heat(g, 1.0).apply(_x())
    assert np.allclose(y, expm(-g.laplacian.toarray()) @ _x(), atol=1e-10)


def test_low_pass_keeps_only_the_low_frequencies():
    g = _path()
    f = GraphFourier(g)
    low = f.basis[:, 0] + f.basis[:, 1]
    high = f.basis[:, 9]
    y = SpectralFilter.low_pass(g, 0.5).apply(low + high)
    assert np.allclose(y, low, atol=1e-10)


def test_response_with_the_wrong_shape_is_refused():
    with pytest.raises(ValueError):
        SpectralFilter(_path(), lambda w: np.ones(3))


def test_chebyshev_filter_matches_the_exact_filter():
    g = _path()
    exact = expm(-g.laplacian.toarray()) @ _x()
    y = ChebyshevFilter(g, lambda w: np.exp(-w), lmax=4.0, order=20).apply(_x())
    assert np.abs(y - exact).max() < 1e-10


def test_chebyshev_error_falls_with_the_order():
    g = _path()
    exact = expm(-g.laplacian.toarray()) @ _x()
    errs = [
        np.abs(
            ChebyshevFilter(g, lambda w: np.exp(-w), lmax=4.0, order=k).apply(_x())
            - exact
        ).max()
        for k in (2, 4, 8)
    ]
    assert errs[0] > errs[1] > errs[2]
