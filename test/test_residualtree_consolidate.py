"""Regression tests for ResidualTree.consolidate with d_model >= 5."""
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.core.deep.neural.consolidation import ResidualTree


def _build(d_model, n_codes, branching, max_depth, n_events=40, seed=0):
    rng = np.random.RandomState(seed)
    t = ResidualTree(d_model=d_model, n_codes=n_codes, max_depth=max_depth,
                     branching=branching, kappa=0.0, seed=seed)
    data = [rng.randn(d_model).astype(np.float32) for _ in range(n_events)]
    for x in data:
        t.insert(jnp.asarray(x, dtype=jnp.float32))
    return t, data


@pytest.mark.parametrize("d_model,n_codes,branching,max_depth", [
    (2, 8, 2, 2),
    (4, 16, 4, 2),
    (5, 8, 2, 1),
    (8, 32, 4, 2),
])
def test_consolidate_runs(d_model, n_codes, branching, max_depth):
    """consolidate() must actually consolidate at least one group.

    On master, d_model >= 5 raises TypeError before reaching the assert.
    On the fixed code, removed > 0 proves the consolidation ran.
    """
    t, _ = _build(d_model, n_codes, branching, max_depth)
    removed = t.consolidate()
    assert removed > 0, f"consolidate did nothing on d={d_model}"
    assert t.n_nodes() >= 1


def test_consolidate_does_not_degrade_read():
    """read() on the inserted events must not degrade after consolidate.

    Uses the inserted events, not new random data. Tolerance +0.05
    (observed deltas are negative on d=5 and d=8).
    """
    d_model, n_codes, branching, max_depth = 5, 8, 2, 1
    t, data = _build(d_model, n_codes, branching, max_depth)

    def mean_rel_err():
        errs = []
        for x in data:
            r = t.read(jnp.asarray(x, dtype=jnp.float32))
            rec = np.asarray(r[0] if isinstance(r, tuple) else r)
            errs.append(float(np.linalg.norm(rec - x) / (np.linalg.norm(x) + 1e-9)))
        return float(np.mean(errs))

    before = mean_rel_err()
    t.consolidate()
    after = mean_rel_err()
    assert after <= before + 0.05, f"read degraded: {before:.4f} -> {after:.4f}"


def test_second_consolidation_keeps_parent_and_reseeds(monkeypatch):
    """A second consolidate() adds the parent codebook and uses a new seed.

    The first call has two fresh children (``2 * n_codes`` points). The
    second call has two fresh children plus the already reduced parent
    codebook (``2 * n_codes + n_codes // branching`` points). The seeds
    passed to k-means must differ between the two calls.
    """
    import dense_armor.core.deep.neural.consolidation.tree as tree_mod

    calls = []
    real = tree_mod.kmeans2

    def spy(X, k, **kw):
        calls.append((X.shape[0], kw["seed"]))
        return real(X, k, **kw)

    monkeypatch.setattr(tree_mod, "kmeans2", spy)
    d_model, n_codes, branching, max_depth = 5, 8, 2, 1
    t, _ = _build(d_model, n_codes, branching, max_depth)
    assert t.consolidate() > 0
    rng = np.random.RandomState(1)
    for _ in range(40):
        t.insert(jnp.asarray(rng.randn(d_model), dtype=jnp.float32))
    assert t.consolidate() > 0
    assert calls == [
        (branching * n_codes, 0),
        (branching * n_codes + n_codes // branching, 1),
    ]
