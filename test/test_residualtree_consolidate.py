"""Regression tests for ResidualTree.consolidate with d_model >= 5."""
import numpy as np
import jax.numpy as jnp
import pytest
from dense_armor.utility.deep.neural.consolidation import ResidualTree


def _build(d_model, n_codes, branching, max_depth, n_events=40, seed=0):
    rng = np.random.RandomState(seed)
    t = ResidualTree(d_model=d_model, n_codes=n_codes, max_depth=max_depth,
                     branching=branching, kappa=0.0, seed=seed)
    for _ in range(n_events):
        t.insert(jnp.asarray(rng.randn(d_model), dtype=jnp.float32))
    return t


@pytest.mark.parametrize("d_model,n_codes,branching,max_depth", [
    (2, 8, 2, 2),
    (4, 16, 4, 2),
    (5, 8, 2, 1),
    (8, 32, 4, 2),
])
def test_consolidate_runs(d_model, n_codes, branching, max_depth):
    """consolidate() must not raise for any d_model (was TypeError for d>=5)."""
    t = _build(d_model, n_codes, branching, max_depth)
    removed = t.consolidate()
    assert removed >= 0
    assert t.n_nodes() >= 1


def test_consolidate_does_not_degrade_read():
    """read() after consolidate must be no worse than before (small tolerance)."""
    d_model, n_codes, branching, max_depth = 5, 8, 2, 1
    t = _build(d_model, n_codes, branching, max_depth)
    rng = np.random.RandomState(1)
    data = [rng.randn(d_model).astype(np.float32) for _ in range(10)]

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
    assert after <= before + 0.1, f"read degraded: {before:.4f} -> {after:.4f}"
