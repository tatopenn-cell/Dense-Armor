"""Tests for heat_kernel, low_pass, apply_filter, chebyshev."""

import numpy as np
import pytest
import scipy.linalg as sla

from dense_armor.core.graph import (
    Graph,
    apply_filter,
    chebyshev_coefficients,
    chebyshev_filter,
    eigendecomposition,
    heat_kernel,
    low_pass,
)


def _path(n):
    return Graph(n, [(i, i + 1, 1.0) for i in range(n - 1)])


def test_heat_kernel_equals_expm():
    g = _path(10)
    tau = 1.0
    H = heat_kernel(g.laplacian, tau)
    assert np.allclose(H, sla.expm(-tau * g.laplacian.toarray()), atol=1e-10)


def test_heat_kernel_matches_pygsp():
    pytest.importorskip("pygsp")
    import pygsp

    g = _path(10)
    G_pg = pygsp.graphs.Graph(g.adjacency, lap_type="combinatorial")
    G_pg.compute_fourier_basis()
    lmax = G_pg.lmax
    tau = 1.0
    # PyGSP Heat usa scale, e il kernel è exp(-scale * lambda / lmax)
    # noi vogliamo exp(-tau * lambda) -> scale = tau * lmax
    f = pygsp.filters.Heat(G_pg, scale=tau * lmax)
    H_pg = f.filter(np.eye(10), method="exact")
    H = heat_kernel(g.laplacian, tau)
    assert np.allclose(H, H_pg, atol=1e-10)


def test_low_pass_removes_high():
    g = _path(10)
    _, U = eigendecomposition(g.laplacian)
    s = U[:, 1]
    e = U[:, -1]
    x = s + e
    L = g.laplacian
    cutoff = 0.5 * (
        np.linalg.eigvalsh(L.toarray())[1] + np.linalg.eigvalsh(L.toarray())[-1]
    )
    H = low_pass(L, cutoff)
    y = H @ x
    assert np.allclose(y, s, atol=1e-10)


def test_apply_filter_linear():
    g = _path(10)
    L = g.laplacian
    a, b = 1.3, -0.7
    x = np.random.default_rng(0).standard_normal(10)
    y = np.random.default_rng(1).standard_normal(10)

    def fn(lam):
        return np.exp(-0.5 * lam)

    lhs = apply_filter(a * x + b * y, L, fn)
    rhs = a * apply_filter(x, L, fn) + b * apply_filter(y, L, fn)
    assert np.allclose(lhs, rhs, atol=1e-10)


def test_heat_kernel_tau_zero():
    g = _path(10)
    H = heat_kernel(g.laplacian, 0.0)
    assert np.allclose(H, np.eye(10), atol=1e-12)


def test_chebyshev_coefficients_and_filter():
    g = _path(10)
    L = g.laplacian
    w = np.linalg.eigvalsh(L.toarray())
    lmax = float(w.max())

    def g_fn(lam):
        return np.exp(-lam)

    coeffs = chebyshev_coefficients(g_fn, lmax, order=20)
    x = np.random.default_rng(0).standard_normal(10)
    y_ch = chebyshev_filter(x, L, coeffs, lmax)
    # riferimento spettrale
    _, U = eigendecomposition(L)
    y_ref = U @ (np.exp(-w) * (U.T @ x))
    assert np.max(np.abs(y_ch - y_ref)) < 1e-10


def test_chebyshev_error_decreases():
    g = _path(10)
    L = g.laplacian
    w = np.linalg.eigvalsh(L.toarray())
    lmax = float(w.max())

    def g_fn(lam):
        return np.exp(-lam)

    x = np.random.default_rng(0).standard_normal(10)
    _, U = eigendecomposition(L)
    y_ref = U @ (np.exp(-w) * (U.T @ x))
    errs = []
    for K in [2, 4, 6, 8, 10]:
        coeffs = chebyshev_coefficients(g_fn, lmax, order=K)
        y = chebyshev_filter(x, L, coeffs, lmax)
        errs.append(np.max(np.abs(y - y_ref)))
    # errore decrescente finché sopra il rumore di macchina
    for i in range(1, len(errs)):
        if errs[i] > 1e-12:
            assert errs[i] < errs[i - 1] + 1e-15 or errs[i] < 1e-12


def test_chebyshev_bad_lmax():
    g = _path(10)
    x = np.zeros(10)
    with pytest.raises(ValueError):
        chebyshev_coefficients(lambda lam: lam, -1.0, 5)
    with pytest.raises(ValueError):
        chebyshev_filter(x, g.laplacian, np.array([1.0]), lmax=-1.0)
    with pytest.raises(ValueError):
        chebyshev_filter(x, g.laplacian, np.array([]), lmax=1.0)
