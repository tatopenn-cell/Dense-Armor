"""Tests for recover_graph_signal and its proximity operators."""

import numpy as np
import pytest

from dense_armor.core.graph import Graph, recover_graph_signal
from dense_armor.core.graph.recovery import (
    project_l1_ball,
    project_l2_ball,
    prox_l1,
)


def test_l1_projection_by_hand():
    assert project_l1_ball([3.0, 1.0], 2.0).tolist() == [2.0, 0.0]
    assert np.allclose(project_l1_ball([1.0, 1.0, 1.0], 1.5), [0.5, 0.5, 0.5])
    assert np.allclose(project_l1_ball([-3.0, 1.0], 2.0), [-2.0, 0.0])


def test_l1_projection_inside_the_ball_and_radius_zero():
    z = np.array([0.2, -0.1])
    assert np.allclose(project_l1_ball(z, 1.0), z)
    assert not project_l1_ball([1.0, 2.0], 0.0).any()


def test_l1_projection_lands_on_the_sphere_and_is_the_nearest_point():
    rng = np.random.default_rng(0)
    for _ in range(20):
        z = rng.standard_normal(8)
        p = project_l1_ball(z, 1.0)
        assert np.abs(p).sum() == pytest.approx(1.0)
        for _ in range(20):
            q = rng.standard_normal(8)
            q = q / np.abs(q).sum() * rng.uniform(0, 1)
            assert np.linalg.norm(z - p) <= np.linalg.norm(z - q) + 1e-12


def test_soft_threshold_by_hand():
    assert prox_l1([3.0, -0.5, -2.0], 1.0).tolist() == [2.0, 0.0, -1.0]


def test_l2_projection_by_hand():
    assert np.allclose(project_l2_ball([3.0, 4.0], [0.0, 0.0], 1.0), [0.6, 0.8])
    assert np.allclose(project_l2_ball([0.1, 0.1], [0.0, 0.0], 1.0), [0.1, 0.1])
    assert np.allclose(project_l2_ball([4.0, 1.0], [1.0, 1.0], 1.0), [2.0, 1.0])


def _path_laplacian(n):
    return Graph(n, [(i, i + 1, 1.0) for i in range(n - 1)])


def _truth(n=15, p=20):
    i, k = np.arange(n)[:, None], np.arange(p)[None, :]
    return np.sin(0.3 * i) * np.cos(0.1 * k)


def test_denoising_beats_the_noisy_signal():
    n, p, sigma = 15, 20, 0.1
    X0 = _truth(n, p)
    Y = X0 + sigma * np.random.default_rng(1).standard_normal((n, p))
    eps = 0.9 * sigma * np.sqrt(n * p)
    Xh = recover_graph_signal(
        Y, _path_laplacian(n), lam=0.05, eps=eps, temporal="l2", max_iter=3000
    )
    rmse = lambda A: np.sqrt(np.mean((A - X0) ** 2))
    assert rmse(Xh) < rmse(Y)


def test_the_fidelity_constraint_is_met():
    n, p, sigma = 15, 20, 0.1
    Y = _truth(n, p) + sigma * np.random.default_rng(2).standard_normal((n, p))
    eps = 0.9 * sigma * np.sqrt(n * p)
    Xh = recover_graph_signal(Y, _path_laplacian(n), lam=0.05, eps=eps, max_iter=3000)
    assert np.linalg.norm(Y - Xh) <= 1.1 * eps


def test_missing_values_are_filled_from_the_neighbours():
    n, p, sigma = 15, 20, 0.05
    X0 = _truth(n, p)
    rng = np.random.default_rng(3)
    mask = (rng.random((n, p)) > 0.1).astype(float)
    Y = mask * (X0 + sigma * rng.standard_normal((n, p)))
    eps = 0.9 * sigma * np.sqrt(mask.sum())
    Xh = recover_graph_signal(
        Y, _path_laplacian(n), mask=mask, lam=0.05, eps=eps, temporal="l2", max_iter=4000
    )
    missing = mask == 0
    assert np.sqrt(np.mean((Xh[missing] - X0[missing]) ** 2)) < 0.5


def test_outliers_are_returned_and_bounded():
    n, p = 10, 12
    Y = _truth(n, p)
    Y[3, 4] += 5.0
    X, S = recover_graph_signal(
        Y, _path_laplacian(n), lam=0.05, eps=0.1, eta=4.0, max_iter=3000, return_outliers=True
    )
    assert X.shape == S.shape == (n, p)
    assert np.abs(S).sum() <= 4.0 * 1.001


def test_a_list_of_laplacians_equals_the_static_one():
    n, p = 8, 6
    Y = _truth(n, p)
    L = _path_laplacian(n)
    a = recover_graph_signal(Y, L, lam=0.1, eps=0.2, max_iter=200)
    b = recover_graph_signal(Y, [L] * p, lam=0.1, eps=0.2, max_iter=200)
    assert np.allclose(a, b)


def test_both_smoothness_terms_run():
    n, p = 8, 6
    Y = _truth(n, p)
    for smoothness in ("signal", "difference"):
        X = recover_graph_signal(
            Y, _path_laplacian(n), lam=0.1, eps=0.2, smoothness=smoothness, max_iter=100
        )
        assert X.shape == (n, p) and np.isfinite(X).all()


def test_bad_arguments_are_refused():
    Y, L = _truth(5, 4), _path_laplacian(5)
    with pytest.raises(ValueError):
        recover_graph_signal(Y[:, :1], L)
    with pytest.raises(ValueError):
        recover_graph_signal(Y, L, lam=-1.0)
    with pytest.raises(ValueError):
        recover_graph_signal(Y, L, temporal="l3")
    with pytest.raises(ValueError):
        recover_graph_signal(Y, L, smoothness="x")
    with pytest.raises(ValueError):
        recover_graph_signal(Y, [L] * 3)
    with pytest.raises(ValueError):
        recover_graph_signal(Y, L, mask=np.ones((2, 2)))
