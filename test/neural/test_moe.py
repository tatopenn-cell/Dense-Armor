"""Tests for SparseMoE."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.gradcheck import gradcheck
from dense_armor.utility.neural.moe import SparseMoE, forward_pure


def _model(n_experts=3, top_k=2, n_x=2, n_y=2, hidden=4, seed=0, cf=1.25):
    return SparseMoE(
        n_x=n_x,
        n_y=n_y,
        n_experts=n_experts,
        top_k=top_k,
        hidden=hidden,
        seed=seed,
        capacity_factor=cf,
    )


def test_forward_shape():
    m = _model()
    x = jnp.ones((5, 2))
    assert m.forward(x).shape == (5, 2)


def test_forward_single_sample():
    m = _model()
    assert m.forward(jnp.ones(2)).shape == (2,)


def test_top_k_sparse():
    m = _model(n_experts=4, top_k=2)
    m.forward(jnp.ones((3, 2)))
    mask = np.asarray(jax.nn.one_hot(m.aux_["top_idx"], 4).sum(axis=-2))
    assert mask.shape == (3, 4)
    assert np.all(mask.sum(axis=1) == 2)


def test_dispatch_structural_no_drop():
    """Con capacity_factor=None, nessun token viene droppato: ogni
    token ha esattamente top_k posizioni non negative in pos. Il
    dispatch e' strutturale, non wall-time."""
    n_tokens, n_experts, top_k = 20, 4, 2
    m = _model(n_experts=n_experts, top_k=top_k, cf=None)
    m.forward(jnp.ones((n_tokens, 2)))
    assert m._cap == n_tokens
    pos = np.asarray(m.aux_["pos"])
    per_token = (pos >= 0).sum(axis=1)
    assert np.all(per_token == top_k), per_token.tolist()
    per_expert = (pos >= 0).sum(axis=0)
    assert per_expert.sum() == n_tokens * top_k


def test_jit_compiles():
    m = _model(n_experts=4, top_k=2, hidden=4)
    m.forward(jnp.ones((10, 2)))
    params = m._params()
    cap = m._cap
    f = jax.jit(lambda p, x: forward_pure(p, x, cap, 2, 4))
    y_jit = f(params, jnp.ones((10, 2)))
    assert y_jit.shape == (10, 2)


def test_capacity_drops():
    """capacity = cf * top_k * n_tokens / n_experts = 1.0 * 2 * 100 / 4 = 50.
    Con input randomici i token si distribuiscono sui 4 esperti, e nessun
    esperto supera la capacita'. Con input identici (jnp.ones) tutti i token
    finiscono sui soli due esperti top-2, e la capacita' viene comunque
    rispettata: il test usa input random per non essere degenere."""
    m = _model(n_experts=4, top_k=2, cf=1.0)
    x = jax.random.normal(jax.random.PRNGKey(0), (100, 2))
    m.forward(x)
    pos = np.asarray(m.aux_["pos"])
    per_expert = (pos >= 0).sum(axis=0)
    assert np.all(per_expert <= 50), per_expert.tolist()


def test_dispatch_independent_of_n_experts():
    """Con top_k fisso, la shape del sotto-batch per esperto dipende
    da capacity = cf * top_k * n_tokens / n_experts, non da n_experts
    da solo. Il test e' strutturale: la scelta di eseguire solo top_k
    esperti si vede da pos e da y_disp, non dal wall-time."""
    n_tokens, top_k, cf = 200, 2, 1.25
    for n_experts in (2, 4, 8):
        m = _model(n_experts=n_experts, top_k=top_k, cf=cf)
        x = jax.random.normal(jax.random.PRNGKey(n_experts), (n_tokens, 2))
        m.forward(x)
        cap = max(1, int(np.floor(cf * top_k * n_tokens / n_experts)))
        assert m._cap == cap
        y_disp_shape = m.aux_["y_disp"].shape
        assert y_disp_shape == (n_experts, cap, 2), y_disp_shape
        pos = np.asarray(m.aux_["pos"])
        per_expert = (pos >= 0).sum(axis=0)
        assert np.all(per_expert <= cap), (n_experts, per_expert.tolist())


def test_load_balancing_loss_positive():
    m = _model()
    m.forward(jnp.ones((10, 2)))
    loss = float(m.load_balancing_loss())
    assert np.isfinite(loss) and loss >= 0.0


def test_backward_before_forward_raises():
    m = _model()
    with pytest.raises(RuntimeError, match="forward"):
        m.backward(jnp.ones((2,)))


def test_backward_produces_grad():
    m = _model(n_experts=3, top_k=2)
    y = m.forward(jnp.ones((5, 2)))
    m.backward(jnp.ones_like(y))
    norms = [float(jnp.linalg.norm(p.grad)) for p in m.params()]
    assert sum(norms) > 0.0


def test_gradcheck_expert_continuous():
    """Il top-k e' discreto: gradcheck sul forward intero sarebbe
    discontinuo. Il test isola un esperto, che e' smooth."""
    m = _model(n_x=2, n_y=2, n_experts=1, top_k=1, hidden=3)

    def f(xx):
        return m.forward(xx, training=True)

    def b(xx, yy, gg):
        return m.backward(gg)

    x = jax.random.normal(jax.random.PRNGKey(0), (2, 2))
    err = gradcheck(f, b, x)
    assert err < 1e-3, f"gradcheck failed, error {err:.2e}"


def test_params_count():
    m = _model(n_experts=3, top_k=2, n_x=2, n_y=2, hidden=4)
    assert len(m.params()) == 2 + 4


def test_invalid_n_experts():
    with pytest.raises(ValueError, match="n_experts"):
        SparseMoE(n_x=2, n_y=2, n_experts=0)


def test_invalid_top_k():
    with pytest.raises(ValueError, match="top_k"):
        SparseMoE(n_x=2, n_y=2, n_experts=2, top_k=0)
    with pytest.raises(ValueError, match="top_k"):
        SparseMoE(n_x=2, n_y=2, n_experts=2, top_k=3)


def test_invalid_alpha():
    with pytest.raises(ValueError, match="alpha"):
        SparseMoE(n_x=2, n_y=2, alpha=-0.1)


def test_invalid_capacity_factor():
    with pytest.raises(ValueError, match="capacity_factor"):
        SparseMoE(n_x=2, n_y=2, capacity_factor=-0.5)

def test_top_k_one():
    """Con top_k=1 ogni token ha esattamente 1 posizione >= 0 in pos."""
    m = _model(n_experts=4, top_k=1, cf=None)
    m.forward(jnp.ones((20, 2)))
    pos = np.asarray(m.aux_["pos"])
    assert np.all((pos >= 0).sum(axis=1) == 1)


def test_tiny_capacity_drops_most():
    """Con capacity_factor=0.01 quasi tutte le assegnazioni sono droppate."""
    m = _model(n_experts=4, top_k=2, cf=0.01)
    m.forward(jnp.ones((100, 2)))
    drop = float(m.dropped_fraction())
    print(f"dropped_fraction={drop:.3f}")
    assert drop > 0.9, drop


def test_many_experts():
    """Con 8 esperti e top_k=2 il forward funziona e la shape e' corretta."""
    m = _model(n_experts=8, top_k=2, cf=1.25)
    y = m.forward(jnp.ones((40, 2)))
    assert y.shape == (40, 2)
    assert m.aux_["y_disp"].shape[0] == 8


def test_batch_one():
    """Batch di 1 campione: nessun crash, shape (2,)."""
    m = _model(n_experts=4, top_k=2, cf=None)
    y = m.forward(jnp.ones((1, 2)))
    assert y.shape == (1, 2)


def test_expert_usage_sums_to_top_k():
    """expert_usage() somma a top_k quando non c'e' dropping."""
    m = _model(n_experts=4, top_k=2, cf=None)
    m.forward(jnp.ones((40, 2)))
    usage = m.expert_usage()
    assert abs(float(usage.sum()) - 2.0) < 1e-5, float(usage.sum())


def test_noisy_gating_changes_output():
    """Con noisy_gating=True e stessi input, due forward danno output diversi."""
    m = SparseMoE(
        n_x=2, n_y=2, n_experts=4, top_k=2, hidden=7, seed=0,
        capacity_factor=None, noisy_gating=True,
    )
    x = jnp.ones((8, 2))
    y1 = m.forward(x)
    y2 = m.forward(x)
    assert not bool(jnp.allclose(y1, y2, atol=1e-6)), "noise non applicato"


def test_noisy_gating_backward():
    """Con noisy_gating=True il backward produce gradienti non nulli."""
    m = SparseMoE(
        n_x=2, n_y=2, n_experts=4, top_k=2, hidden=7, seed=0,
        capacity_factor=None, noisy_gating=True,
    )
    y = m.forward(jnp.ones((8, 2)))
    m.backward(jnp.ones_like(y))
    norms = [float(jnp.linalg.norm(p.grad)) for p in m.params()]
    assert sum(norms) > 0.0
def test_load_balancing_loss_before_forward_raises():
    m = _model()
    with pytest.raises(RuntimeError, match="forward"):
        m.load_balancing_loss()


def test_expert_usage_before_forward_raises():
    m = _model()
    with pytest.raises(RuntimeError, match="forward"):
        m.expert_usage()


def test_dropped_fraction_before_forward_raises():
    m = _model()
    with pytest.raises(RuntimeError, match="forward"):
        m.dropped_fraction()


def test_repr_contains_hyperparameters():
    m = _model(n_experts=3, top_k=2, hidden=4, cf=1.25)
    r = repr(m)
    assert "SparseMoE(" in r
    assert "n_experts=3" in r
    assert "top_k=2" in r
    assert "hidden=4" in r
    assert "capacity_factor=1.25" in r

