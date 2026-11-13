"""Tests for gft and igft."""

import numpy as np
import pytest

from dense_armor.core.graph import Graph, eigendecomposition, gft, igft


def _path(n):
    return Graph(n, [(i, i + 1, 1.0) for i in range(n - 1)])


def test_parseval():
    g = _path(10)
    _, U = eigendecomposition(g.laplacian)
    rng = np.random.default_rng(0)
    signals = [
        np.ones(10),
        np.eye(10)[0],
        np.sin(2 * np.pi * np.arange(10) / 10),
        rng.standard_normal(10),
        np.linspace(-1, 1, 10),
    ]
    for x in signals:
        x_hat = gft(x, U)
        assert abs(np.sum(x**2) - np.sum(x_hat**2)) < 1e-10


def test_igft_gft_roundtrip():
    g = _path(10)
    _, U = eigendecomposition(g.laplacian)
    x = np.random.default_rng(0).standard_normal(10)
    assert np.allclose(igft(gft(x, U), U), x, atol=1e-10)


def test_gft_matches_pygsp():
    # Gli autovettori sono definiti a meno del segno: si confronta il
    # modulo, non i valori con segno. Il gauge fisso (fix_sign) rende
    # confrontabili i segni, ma PyGSP usa un gauge suo, quindi il test
    # confronta |gft| (pattern 17 del kit).
    pytest.importorskip("pygsp")
    import pygsp

    g = _path(10)
    G_pg = pygsp.graphs.Graph(g.adjacency, lap_type="combinatorial")
    G_pg.compute_fourier_basis()
    _, U = eigendecomposition(g.laplacian)
    x = np.random.default_rng(0).standard_normal(10)
    assert np.allclose(np.abs(gft(x, U)), np.abs(G_pg.gft(x)), atol=1e-10)


def test_constant_has_only_dc():
    g = _path(10)
    _, U = eigendecomposition(g.laplacian)
    x = np.ones(10)
    x_hat = gft(x, U)
    assert abs(x_hat[0]) > 0
    assert np.allclose(x_hat[1:], 0.0, atol=1e-10)
