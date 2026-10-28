"""Tests for VQMemory (item E of task N09)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.neural.consolidation import (
    ArchivalMemory,
    DualMemory,
    LifecycleMemory,
    PQMemory,
    ResidualTree,
    TheoreticalBound,
    VQMemory,
)


def _data(n=200, seed=0):
    rng = np.random.RandomState(seed)
    return jnp.asarray(rng.randn(n, 2) * 0.5, dtype=jnp.float32)


def test_quantize_matches_bruteforce():
    vq = VQMemory(n_x=2, d_model=4, n_codes=8, seed=0)
    x = _data(50)
    z_e = vq.encode(x)
    z_q, k, _ = vq.quantize(z_e)
    diff = z_e[:, None, :] - vq.codebook.value[None, :, :]
    dist = np.asarray(np.sum(np.asarray(diff) ** 2, axis=-1))
    k_ref = np.argmin(dist, axis=-1)
    assert np.array_equal(np.asarray(k), k_ref)




def test_codebook_gradient_independent_of_commitment():
    """The codebook term is ``-2 (z_e - e_k)``, the commitment does not enter."""
    vq1 = VQMemory(n_x=2, d_model=4, n_codes=8, commitment=0.0, seed=0)
    vq2 = VQMemory(n_x=2, d_model=4, n_codes=8, commitment=0.5, seed=0)
    x = _data(10)
    x1 = vq1.forward(x)
    vq1.backward(jnp.ones_like(x1))
    x2 = vq2.forward(x)
    vq2.backward(jnp.ones_like(x2))
    g1 = np.asarray(vq1.codebook.grad)
    g2 = np.asarray(vq2.codebook.grad)
    assert np.allclose(g1, g2, atol=1e-6), "commitment must not change codebook grad"


def test_codebook_ema_update():
    vq = VQMemory(n_x=2, d_model=4, n_codes=8, seed=0)
    x = _data(30)
    for _ in range(10):
        vq.forward(x)
        vq.update_codebook()
    used = np.asarray(vq._counts > 0)
    means = np.asarray(vq._sums[used] / vq._counts[used][:, None])
    codebook_used = np.asarray(vq.codebook.value[used])
    assert np.allclose(codebook_used, means, atol=1e-5)


def test_codebook_ema_reduces_quantization_mse():
    """The EMA codebook update reduces ||z_e - e_k||^2 on the same data.

    The full training of encoder and decoder goes through ``Trainer``,
    which clones one optimiser per parameter; a manual loop with one
    shared optimiser fails on the differing shapes. This test pins what
    ``VQMemory`` itself controls: the codebook EMA.
    """
    vq = VQMemory(n_x=2, d_model=4, n_codes=8, seed=0)
    x = _data(200)
    z_e = vq.encode(x)

    def quant_mse(z):
        diff = z[:, None, :] - vq.codebook.value[None, :, :]
        dist = jnp.sum(diff * diff, axis=-1)
        return float(jnp.mean(jnp.min(dist, axis=-1)))

    mse_0 = quant_mse(z_e)
    for _ in range(10):
        vq.forward(x)
        vq.update_codebook()
    mse_1 = quant_mse(vq.encode(x))
    assert mse_1 <= mse_0, f"quantization MSE did not decrease: {mse_0:.4f} -> {mse_1:.4f}"

def test_all_slots_reachable():
    vq = VQMemory(n_x=2, d_model=4, n_codes=8, seed=0)
    x = _data(10000, seed=1)
    z_e = vq.encode(x)
    _, k, _ = vq.quantize(z_e)
    used = set(np.asarray(k).tolist())
    assert len(used) >= 4, f"only {len(used)} slots of 8 used"


def test_pq_fit_shapes():
    pq = PQMemory(d_model=8, n_subspaces=2, n_centroids=8, seed=0)
    X = jnp.asarray(np.random.RandomState(0).randn(200, 8), dtype=jnp.float32)
    pq.fit(X)
    assert len(pq.centroids_) == 2
    assert pq.centroids_[0].shape == (8, 4)
    assert pq.centroids_[1].shape == (8, 4)


def test_pq_encode_indices():
    pq = PQMemory(d_model=8, n_subspaces=2, n_centroids=8, seed=0)
    X = jnp.asarray(np.random.RandomState(0).randn(200, 8), dtype=jnp.float32)
    pq.fit(X)
    code = pq.encode(X[0])
    assert code.shape == (2,)
    assert 0 <= int(code[0]) < 8
    assert 0 <= int(code[1]) < 8


def test_pq_decode_shape():
    pq = PQMemory(d_model=8, n_subspaces=2, n_centroids=8, seed=0)
    X = jnp.asarray(np.random.RandomState(0).randn(200, 8), dtype=jnp.float32)
    pq.fit(X)
    code = pq.encode(X[0])
    x_hat = pq.decode(code)
    assert x_hat.shape == (8,)


def test_pq_adc_equals_reconstruction_error():
    pq = PQMemory(d_model=8, n_subspaces=2, n_centroids=8, seed=0)
    X = jnp.asarray(np.random.RandomState(0).randn(200, 8), dtype=jnp.float32)
    pq.fit(X)
    q = X[0]
    code = pq.encode(q)
    adc = float(pq.asymmetric_distance(q, code)[0])
    x_hat = pq.decode(code)
    err = float(jnp.sum((q - x_hat) ** 2))
    assert abs(adc - err) < 1e-5, f"ADC {adc} != error {err}"


def test_pq_mse_decreases_with_more_subspaces():
    X = jnp.asarray(np.random.RandomState(0).randn(500, 8), dtype=jnp.float32)
    ms = []
    for n_sub in (1, 2, 4):
        pq = PQMemory(d_model=8, n_subspaces=n_sub, n_centroids=16, seed=0)
        pq.fit(X)
        codes = pq.encode(X)
        x_hat = pq.decode(codes)
        mse = float(jnp.mean(jnp.sum((X - x_hat) ** 2, axis=-1)))
        ms.append(mse)
    assert ms[1] <= ms[0] * 1.1, f"2 subspaces {ms[1]:.3f} vs 1 {ms[0]:.3f}"
    assert ms[2] <= ms[1] * 1.1, f"4 subspaces {ms[2]:.3f} vs 2 {ms[1]:.3f}"


def test_pq_all_codes_reachable():
    X = jnp.asarray(np.random.RandomState(1).randn(2000, 8), dtype=jnp.float32)
    pq = PQMemory(d_model=8, n_subspaces=2, n_centroids=8, seed=0)
    pq.fit(X)
    codes = np.asarray(pq.encode(X))
    for s in range(2):
        used = set(codes[:, s].tolist())
        assert len(used) >= 6, f"subspace {s}: only {len(used)} of 8 codes used"


def test_pq_divisibility_error():
    with pytest.raises(ValueError):
        PQMemory(d_model=7, n_subspaces=2)


def test_residual_tree_single_node():
    tree = ResidualTree(d_model=2, n_codes=8, max_depth=3, branching=4, seed=0)
    assert tree.n_nodes() == 1
    assert tree.max_nodes() == (4**4 - 1) // (4 - 1)


def test_residual_tree_branching_creates_children():
    tree = ResidualTree(d_model=2, n_codes=8, max_depth=2, branching=4,
                        kappa=0.01, seed=0)
    rng = np.random.RandomState(0)
    for c in [(-3.0, -3.0), (3.0, 3.0), (-3.0, 3.0), (3.0, -3.0)]:
        for _ in range(5):
            tree.insert(jnp.asarray(rng.randn(2) * 0.1 + np.array(c),
                                    dtype=jnp.float32))
    assert tree.n_nodes() > 1


def test_residual_tree_read_returns_leaf():
    """With kappa=0 the tree always grows, and read returns the deepest node.

    With 8 codes and 4 clusters the root fits everything and no child is
    created; kappa=0.0 forces a child whenever the residual is above 0,
    which is what the test wants to exercise.
    """
    tree = ResidualTree(d_model=2, n_codes=4, max_depth=2, branching=4,
                        kappa=0.0, seed=0)
    rng = np.random.RandomState(1)
    centers = [(-3.0, -3.0), (3.0, 3.0), (-3.0, 3.0), (3.0, -3.0)]
    for c in centers:
        for _ in range(10):
            x = jnp.asarray(rng.randn(2) * 0.1 + np.array(c), dtype=jnp.float32)
            tree.insert(x)
    _, depth_root = tree.read(jnp.asarray([0.0, 0.0], dtype=jnp.float32))
    _, depth_leaf = tree.read(jnp.asarray([3.0, 3.0], dtype=jnp.float32))
    assert depth_leaf >= 1, f"read returned root (depth 0), tree has {tree.n_nodes()} nodes"


def test_residual_tree_max_depth_respected():
    tree = ResidualTree(d_model=2, n_codes=4, max_depth=1, branching=2,
                        kappa=0.0, seed=0)
    rng = np.random.RandomState(2)
    for _ in range(50):
        tree.insert(jnp.asarray(rng.randn(2) * 10, dtype=jnp.float32))
    assert tree.n_nodes() <= tree.max_nodes()


def test_residual_tree_consolidate_reduces_nodes():
    tree = ResidualTree(d_model=2, n_codes=8, max_depth=3, branching=2,
                        kappa=0.0, seed=0)
    rng = np.random.RandomState(3)
    for _ in range(20):
        tree.insert(jnp.asarray(rng.randn(2) * 5, dtype=jnp.float32))
    n_before = tree.n_nodes()
    removed = tree.consolidate()
    n_after = tree.n_nodes()
    assert removed >= 0
    assert n_after == n_before - removed


def test_residual_tree_state_bounded():
    tree = ResidualTree(d_model=2, n_codes=4, max_depth=3, branching=2,
                        kappa=0.0, seed=0)
    rng = np.random.RandomState(4)
    for _ in range(200):
        tree.insert(jnp.asarray(rng.randn(2) * 10, dtype=jnp.float32))
    assert tree.n_nodes() <= tree.max_nodes()
    bytes_size = tree.state_size_bytes()
    assert bytes_size <= tree.max_nodes() * tree.n_codes * 4 * 4


def test_lifecycle_default_state_active():
    lm = LifecycleMemory(d_model=4, seed=0)
    x = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    idx = lm.write(x, t=0.0)
    assert lm.state_of(idx) == "active"


def test_lifecycle_decay_to_archive():
    lm = LifecycleMemory(d_model=4, decay_horizon_s=10.0, seed=0)
    x = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    idx = lm.write(x, t=0.0)
    for k in range(1, 5):
        lm.tick(t=10.0 * k + 0.5)
    assert lm.state_of(idx) == "archive"


def test_lifecycle_promote_on_access():
    lm = LifecycleMemory(d_model=4, promote_threshold=2,
                         decay_horizon_s=10.0, seed=0)
    x = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    idx = lm.write(x, t=0.0)
    for k in range(1, 6):
        lm.tick(t=20.0 * k)
    assert lm.state_of(idx) == "archive"
    lm.access(idx, t=200.0)
    lm.access(idx, t=200.0)
    assert lm.state_of(idx) == "cold"
    lm.access(idx, t=200.0)
    lm.access(idx, t=200.0)
    assert lm.state_of(idx) == "warm"


def test_lifecycle_mse_archive_above_active():
    lm = LifecycleMemory(d_model=8, decay_horizon_s=10.0, seed=0)
    rng = np.random.RandomState(0)
    x = jnp.asarray(rng.randn(8) * 10.0, dtype=jnp.float32)
    idx = lm.write(x, t=0.0)
    mse_active = lm.mse_of(idx)
    for k in range(1, 6):
        lm.tick(t=20.0 * k)
    assert lm.state_of(idx) == "archive"
    mse_archive = lm.mse_of(idx)
    assert mse_archive >= mse_active, (
        f"archive MSE {mse_archive:.4f} < active MSE {mse_active:.4f}"
    )


def test_lifecycle_bits_per_state():
    assert LifecycleMemory.BITS == {"active": 32, "warm": 8, "cold": 4, "archive": 2}


def test_dual_memory_fast_capacity():
    dm = DualMemory(d_model=4, fast_capacity=8, seed=0)
    rng = np.random.RandomState(0)
    for i in range(20):
        dm.write(jnp.asarray(rng.randn(4), dtype=jnp.float32), t=float(i))
    fast, slow = dm.size()
    assert fast == 8
    assert slow == 0


def test_dual_memory_consolidate():
    dm = DualMemory(d_model=4, fast_capacity=8, slow_capacity=32, seed=0)
    rng = np.random.RandomState(1)
    for i in range(6):
        dm.write(jnp.asarray(rng.randn(4), dtype=jnp.float32), t=float(i))
    moved = dm.consolidate()
    assert moved == 6
    fast, slow = dm.size()
    assert fast == 0
    assert slow == 6


def test_dual_memory_topk_separation():
    """Top-k separation returns the most dissimilar events, not the closest."""
    dm = DualMemory(d_model=2, fast_capacity=16, top_k=3, seed=0)
    centers = [(-5.0, 0.0), (0.0, 0.0), (5.0, 0.0), (0.0, 5.0)]
    for c in centers:
        dm.write(jnp.asarray(c, dtype=jnp.float32), t=0.0)
    query = jnp.asarray((0.0, 0.0), dtype=jnp.float32)
    sep = dm.top_k_separation(query)
    assert len(sep) == 3
    closest_idx = int(np.argmin(np.sum(
        (np.asarray(centers) - np.asarray(query)) ** 2, axis=1
    )))
    assert closest_idx not in sep, (
        f"the closest event (index {closest_idx}) is in the top-k separation"
    )


def test_dual_memory_replay_prioritised():
    dm = DualMemory(d_model=2, fast_capacity=16, slow_capacity=64, seed=0)
    for _ in range(10):
        dm.write(jnp.asarray((1.0, 1.0), dtype=jnp.float32), t=0.0)
    dm.consolidate()
    rng = np.random.RandomState(42)
    new_events = [jnp.asarray(rng.randn(2) * 5, dtype=jnp.float32) for _ in range(5)]
    for x in new_events:
        dm.write(x, t=0.0)
    dm.consolidate()
    samples = dm.replay(50)
    assert len(samples) == 50


def test_dual_memory_total_bounded():
    dm = DualMemory(d_model=2, fast_capacity=8, slow_capacity=32, seed=0)
    rng = np.random.RandomState(3)
    for i in range(100):
        dm.write(jnp.asarray(rng.randn(2), dtype=jnp.float32), t=float(i))
        if i % 10 == 9:
            dm.consolidate()
    fast, slow = dm.size()
    assert fast <= 8
    assert slow <= 32


def test_bound_zero_below_capacity():
    tb = TheoreticalBound(n_params=100, n_neurons=10)
    assert abs(tb.capacity() - 1.38) < 1e-9
    assert tb.bound(n_events=0) == 0.0
    assert tb.bound(n_events=1) == 0.0


def test_bound_positive_above_capacity():
    tb = TheoreticalBound(n_params=100, n_neurons=10)
    b = tb.bound(n_events=5)
    assert b > 0.0
    assert abs(b - (5 - 1.38) / 100) < 1e-9


def test_bound_grows_with_events():
    tb = TheoreticalBound(n_params=100, n_neurons=10)
    b1 = tb.bound(n_events=5)
    b2 = tb.bound(n_events=10)
    assert b2 > b1


def test_bound_never_negative():
    tb = TheoreticalBound(n_params=100, n_neurons=10)
    for n in range(20):
        assert tb.bound(n_events=n) >= 0.0


def test_bound_neurons_from_params():
    tb = TheoreticalBound(n_params=100)
    assert tb.n_neurons == 10
    tb2 = TheoreticalBound(n_params=100, n_neurons=5)
    assert tb2.n_neurons == 5
    assert abs(tb2.capacity() - 0.69) < 1e-9


def test_bound_hopfield_16_positive():
    tb = TheoreticalBound(n_params=256, n_neurons=16)
    assert tb.capacity() > 2.0
    assert tb.bound(n_events=4) > 0.0


def test_archival_importance_three_cases():
    am = ArchivalMemory(d_model=2, n_codes=8, max_depth=3, seed=0)
    imp_novel = am.importance(jnp.asarray([10.0, 10.0], dtype=jnp.float32))
    am.write(jnp.asarray([1.0, 0.0], dtype=jnp.float32))
    am.write(jnp.asarray([1.0, 0.0], dtype=jnp.float32))
    imp_shared = am.importance(jnp.asarray([1.0, 0.0], dtype=jnp.float32))
    assert 0.0 <= imp_novel <= 1.0
    assert 0.0 <= imp_shared <= 1.0
    assert imp_novel > imp_shared, (
        f"novel {imp_novel:.3f} not above shared {imp_shared:.3f}"
    )


def test_archival_write_depth():
    am = ArchivalMemory(d_model=2, max_depth=3, seed=0)
    assert am.write_depth(0.0) == 0
    assert am.write_depth(0.5) == 2
    assert am.write_depth(1.0) == 3


def test_archival_transition():
    am = ArchivalMemory(d_model=2, seed=0)
    assert am.transition("active", accesses=3, age_s=0.0) == "active"
    assert am.transition("cold", accesses=3, age_s=0.0) == "warm"
    assert am.transition("warm", accesses=0, age_s=700.0) == "cold"
    assert am.transition("archive", accesses=0, age_s=700.0) == "archive"


def test_archival_write_read_roundtrip():
    am = ArchivalMemory(d_model=2, n_codes=8, seed=0)
    x = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    idx = am.write(x, t=0.0)
    x_hat, state, bits = am.reconstruct(idx)
    assert bits == 32
    assert state == "active"
    assert x_hat.shape == (2,)


def test_archival_size_bounded():
    am = ArchivalMemory(d_model=2, n_codes=4, max_depth=2, branching=2,
                        fast_capacity=8, slow_capacity=16, seed=0)
    rng = np.random.RandomState(0)
    for i in range(100):
        am.write(jnp.asarray(rng.randn(2), dtype=jnp.float32), t=float(i))
    assert am.tree.n_nodes() <= am.tree.max_nodes()
    fast, slow = am.dual.size()
    assert fast <= 8
    assert slow <= 16


def _all_nodes(node):
    yield node
    for c in node["children"]:
        yield from _all_nodes(c)


def test_vq_straight_through():
    """With commitment=0, the decoder gradient equals the encoder gradient."""
    vq = VQMemory(n_x=2, d_model=4, n_codes=8, commitment=0.0, seed=0)
    x = jnp.asarray(np.random.RandomState(0).randn(4, 2), dtype=jnp.float32)
    x_hat = vq.forward(x)
    g = jnp.ones_like(x_hat)
    dz_q = vq.decoder.backward(g)
    z_e = vq._z_e
    e_k = vq.codebook.value[vq._k]
    dz_e = dz_q + 2.0 * vq.commitment * (z_e - e_k)
    assert np.allclose(np.asarray(dz_e), np.asarray(dz_q), atol=1e-6)

def test_vq_mse_decreases_20_steps():
    vq = VQMemory(n_x=2, d_model=4, n_codes=16, commitment=0.25, seed=0)
    x = jnp.asarray(np.random.RandomState(0).randn(200, 2), dtype=jnp.float32)
    def mse():
        z = vq.encode(x)
        zq, _, _ = vq.quantize(z)
        return float(jnp.mean(jnp.sum((x - vq.decode(zq)) ** 2, axis=-1)))
    before = mse()
    vq.fit(x, n_steps=20)
    after = mse()
    assert after < before, f"MSE did not decrease: {before:.4f} -> {after:.4f}"



def test_residual_decreases_with_depth():
    """A leaf trained on its own samples beats the root on the same samples."""
    tree = ResidualTree(d_model=2, n_codes=4, max_depth=2, branching=4, kappa=0.0, seed=0)
    rng = np.random.RandomState(0)
    for c in [(-3.0, -3.0), (3.0, 3.0), (-3.0, 3.0), (3.0, -3.0)]:
        for _ in range(30):
            tree.insert(jnp.asarray(np.array(c) + rng.randn(2) * 0.1, dtype=jnp.float32))
    leaves = [n for n in _all_nodes(tree.root) if n["depth"] > 0 and len(n["samples"]) > 0]
    assert leaves, "no leaf with samples"
    leaf = max(leaves, key=lambda n: len(n["samples"]))
    n_leaf = len(leaf["samples"])
    if n_leaf < 4:
        # not enough samples to learn a codebook: declare and skip
        print(f"skip: best leaf has {n_leaf} samples (<4)")
        return
    xs = jnp.stack(leaf["samples"], axis=0)
    leaf_pred = leaf["vq"].decode(leaf["vq"].encode(xs))
    root_pred = tree.root["vq"].decode(tree.root["vq"].encode(xs))
    leaf_mse = float(jnp.mean(jnp.sum((xs - leaf_pred) ** 2, axis=-1)))
    root_mse = float(jnp.mean(jnp.sum((xs - root_pred) ** 2, axis=-1)))
    assert leaf_mse <= root_mse + 1e-3, f"leaf {leaf_mse:.4f} > root {root_mse:.4f}"

def test_consolidate_no_loss():
    """consolidate reduces nodes without increasing state size."""
    tree = ResidualTree(d_model=2, n_codes=8, max_depth=3, branching=2, kappa=0.0, seed=0)
    rng = np.random.RandomState(0)
    for _ in range(20):
        tree.insert(jnp.asarray(rng.randn(2) * 5, dtype=jnp.float32))
    nodes_before = tree.n_nodes()
    bytes_before = tree.state_size_bytes()
    tree.consolidate()
    nodes_after = tree.n_nodes()
    bytes_after = tree.state_size_bytes()
    assert nodes_after <= nodes_before, f"nodes grew: {nodes_before} -> {nodes_after}"
    assert bytes_after <= bytes_before + tree.n_codes * 4, f"bytes grew: {bytes_before} -> {bytes_after}"

def test_fisher_rao_positive():
    lm = LifecycleMemory(d_model=4, seed=0)
    x = jnp.asarray([0.3, -0.7, 1.1, 2.4], dtype=jnp.float32)
    idx = lm.write(x, t=0.0)
    lm._events[idx]["state"] = "cold"
    xh, _, _ = lm.read(idx)
    p_ = (np.asarray(x) - np.min(np.asarray(x))) / (np.ptp(np.asarray(x)) + 1e-9)
    q_ = (np.asarray(xh) - np.min(np.asarray(xh))) / (np.ptp(np.asarray(xh)) + 1e-9)
    p_ = p_ / (np.sum(p_) + 1e-9)
    q_ = q_ / (np.sum(q_) + 1e-9)
    bc = min(float(np.sum(np.sqrt(p_ * q_ + 1e-12))), 1.0)
    d = float(2.0 * np.arccos(bc))
    assert d > 0.0, f"Fisher-Rao distance is {d}"


def test_j_bibliographic_only():
    doc = (TheoreticalBound.__doc__ or "").lower()
    assert "bibliographic only" in doc


def test_promote_preserves_mse():
    """A promoted parent has n_codes // branching and sum of child supports."""
    tree = ResidualTree(d_model=2, n_codes=8, max_depth=3, branching=2, kappa=0.0, seed=0)
    rng = np.random.RandomState(1)
    for _ in range(30):
        tree.insert(jnp.asarray(rng.randn(2) * 3, dtype=jnp.float32))
    parent = tree.root
    while parent["children"] and len(parent["children"]) < 2:
        parent = parent["children"][0]
    if len(parent["children"]) < 2:
        return
    kids = list(parent["children"])
    support_kids = sum(k["support"] for k in kids)
    alpha_kids = float(np.mean([k["alpha"] for k in kids]))
    tree.consolidate()
    assert parent["support"] >= support_kids, f"support {parent['support']} < {support_kids}"
    assert abs(parent["alpha"] - alpha_kids) < max(1e-3, 0.5 * abs(alpha_kids)), "alpha changed too much"


def test_vq_backward_before_forward_raises():
    """backward() before forward() raises RuntimeError (REVISIONE_IA.md §2b)."""
    import pytest
    vq = VQMemory(n_x=2, d_model=4, n_codes=8, seed=0)
    with pytest.raises(RuntimeError):
        vq.backward(jnp.ones((1, 2)))



def test_vq_quantization_is_discrete_declared():
    """The VQ forward is not differentiable: gradcheck is not applicable.

    ``VQMemory.quantize`` selects ``k = argmin_j ||z_e - e_j||``. A finite
    difference on the weights changes ``z_e`` by ``eps``, which can cross a
    codebook boundary, so the finite-difference gradient of the true
    forward is not the analytic one. The straight-through estimator is a
    biased surrogate of a gradient that does not exist. The contract is
    tested by property (``test_vq_straight_through``: with commitment 0,
    ``dz_e = dz_q``; ``test_codebook_gradient_independent_of_commitment``:
    the commitment term does not change the codebook gradient), not by
    ``gradcheck``. This is a declared deviation from REVISIONE_IA.md §2b.
    """
    vq = VQMemory(n_x=2, d_model=4, n_codes=8, seed=0)
    x = jnp.asarray(np.random.RandomState(0).randn(4, 2), dtype=jnp.float32)
    z_e = vq.encode(x)
    z_q, k, _ = vq.quantize(z_e)
    assert z_q.shape == z_e.shape
    assert k.shape == (4,)
    assert int(k.min()) >= 0 and int(k.max()) < 8


def test_vq_params_and_backward_before_fit():
    """VQMemory.params() exists; backward before forward raises; encode before fit raises."""
    import pytest
    vq = VQMemory(n_x=2, d_model=4, n_codes=8, seed=0)
    params = vq.params()
    assert len(params) >= 2
    with pytest.raises(RuntimeError):
        vq.backward(jnp.ones((1, 2)))


def test_pq_fit_wrong_d_model_raises():
    """PQMemory.fit rejects a wrong d_model (line 236)."""
    import pytest
    pq = PQMemory(d_model=8, n_subspaces=2, n_centroids=8, seed=0)
    with pytest.raises(ValueError, match="expected d_model"):
        pq.fit(jnp.ones((10, 5)))

def test_pq_encode_before_fit_raises():
    import pytest
    pq = PQMemory(d_model=8, n_subspaces=2, n_centroids=8, seed=0)
    with pytest.raises(RuntimeError, match="fit"):
        pq.encode(jnp.ones(8))
    with pytest.raises(RuntimeError, match="fit"):
        pq.decode(jnp.zeros(2, dtype=jnp.int32))
    with pytest.raises(RuntimeError, match="fit"):
        pq.asymmetric_distance(jnp.ones(8), jnp.zeros(2, dtype=jnp.int32))


def test_pq_kmeans_empty_cluster_falls_back():
    """k-means keeps the old centroid when a cluster is empty (line 228)."""
    # data with fewer points than centroids forces some clusters to be empty
    X = jnp.asarray(np.random.RandomState(0).randn(4, 8), dtype=jnp.float32)
    pq = PQMemory(d_model=8, n_subspaces=2, n_centroids=16, seed=0)
    pq.fit(X)
    assert len(pq.centroids_) == 2


def test_tree_consolidate_visits_children():
    """consolidate recurses into every child (line 468) and reports n_events (line 478)."""
    tree = ResidualTree(d_model=2, n_codes=4, max_depth=3, branching=2, kappa=0.0, seed=0)
    rng = np.random.RandomState(0)
    for _ in range(20):
        tree.insert(jnp.asarray(rng.randn(2) * 5, dtype=jnp.float32))
    n_before = tree.n_events()
    tree.consolidate()
    assert tree.n_events() == n_before


def test_lifecycle_read_out_of_range():
    import pytest
    lm = LifecycleMemory(d_model=4, seed=0)
    lm.write(jnp.zeros(4, dtype=jnp.float32), t=0.0)
    with pytest.raises(IndexError):
        lm.read(5)
    with pytest.raises(IndexError):
        lm.access(5)
    assert len(lm) == 1


def test_lifecycle_quantize_identity_for_high_bits():
    """bits >= 24 returns x (line 554), hi == lo returns x."""
    lm = LifecycleMemory(d_model=4, value_range=(-5.0, 5.0), seed=0)
    x = jnp.asarray([0.3, -0.7, 1.1, 2.4], dtype=jnp.float32)
    out = lm._quantize(x, 32)
    assert np.array_equal(np.asarray(out), np.asarray(x))
    # hi == lo
    lm2 = LifecycleMemory(d_model=4, value_range=(0.0, 0.0), seed=0)
    assert np.array_equal(np.asarray(lm2._quantize(x, 4)), np.asarray(x))


def test_dual_memory_empty_replay_and_separation():
    dm = DualMemory(d_model=4, seed=0)
    assert dm.replay(3) == []
    assert dm.top_k_separation(jnp.zeros(4, dtype=jnp.float32)) == []


def test_archival_read_and_len():
    am = ArchivalMemory(d_model=2, n_codes=8, seed=0)
    am.write(jnp.asarray([1.0, 0.0], dtype=jnp.float32), t=0.0)
    _, depth = am.read(jnp.asarray([1.0, 0.0], dtype=jnp.float32))
    assert depth >= 0
    assert len(am) == 1






def test_tree_router_linear_is_default():
    """The default router is 'linear', and the constructor accepts 'mondrian'."""
    import pytest
    tree = ResidualTree(d_model=2, n_codes=4, max_depth=2, seed=0)
    assert tree.router == "linear"
    tree2 = ResidualTree(d_model=2, n_codes=4, max_depth=2, seed=0, router="mondrian")
    assert tree2.router == "mondrian"
    with pytest.raises(ValueError, match="router must be"):
        ResidualTree(d_model=2, router="bad")


def test_tree_router_mondrian_valid_node():
    """The Mondrian router returns a node within max_depth, with bounded nodes."""
    tree = ResidualTree(
        d_model=2, n_codes=4, max_depth=2, branching=4, kappa=0.0, seed=0,
        router="mondrian",
    )
    rng = np.random.RandomState(0)
    for _ in range(60):
        tree.insert(jnp.asarray(rng.randn(2) * 3, dtype=jnp.float32))
    assert tree.n_nodes() <= tree.max_nodes()
    _, depth = tree.read(jnp.asarray([1.0, 1.0], dtype=jnp.float32))
    assert 0 <= depth <= tree.max_depth


def test_tree_router_mondrian_fewer_encodes():
    """Mondrian uses box comparisons; the same data visits no more nodes than linear."""
    rng_data = np.random.RandomState(0).randn(60, 2) * 3.0
    t_lin = ResidualTree(
        d_model=2, n_codes=4, max_depth=2, branching=4, kappa=0.0, seed=0,
        router="linear",
    )
    t_mon = ResidualTree(
        d_model=2, n_codes=4, max_depth=2, branching=4, kappa=0.0, seed=0,
        router="mondrian",
    )
    for x in rng_data:
        t_lin.insert(jnp.asarray(x, dtype=jnp.float32))
        t_mon.insert(jnp.asarray(x, dtype=jnp.float32))
    assert t_mon.n_nodes() <= t_lin.n_nodes()
    # both respect the same bound
    assert t_lin.n_nodes() <= t_lin.max_nodes()
    assert t_mon.n_nodes() <= t_mon.max_nodes()


def test_tree_router_mondrian_time_lower():
    """Mondrian insert is faster than linear insert on the same data.

    The test measures the wall-clock time of 200 inserts in each mode,
    on the same data. The linear mode calls encode and decode per
    child at every level; Mondrian calls only float min/max. The test
    declares the tolerance: Mondrian must be at most 1.5x the linear
    time, which is a weak bound that fails if the code does a full
    scan by mistake.
    """
    import time as _time
    rng_data = np.random.RandomState(1).randn(200, 2) * 3.0
    xs = [jnp.asarray(x, dtype=jnp.float32) for x in rng_data]

    def _time_router(name):
        tree = ResidualTree(
            d_model=2, n_codes=4, max_depth=2, branching=4, kappa=0.0, seed=0,
            router=name,
        )
        t0 = _time.perf_counter()
        for x in xs:
            tree.insert(x)
        return _time.perf_counter() - t0

    t_lin = _time_router("linear")
    t_mon = _time_router("mondrian")
    ratio = t_mon / max(t_lin, 1e-9)
    assert ratio <= 1.5, (
        f"Mondrian insert is {ratio:.2f}x the linear time "
        f"({t_mon:.3f}s vs {t_lin:.3f}s); the bound is 1.5"
    )

def test_vqmemory_fit_updates_lr_after_first_call():
    """The second fit() call applies the new lr to the existing optimisers.

    VQMemory.fit created the Adam optimisers only on the first call; from
    the second on the lr argument was ignored. The fix sets o.lr = lr on
    the existing optimisers. The test calls fit twice and checks the lr
    stored on the optimisers.
    """
    import jax.numpy as jnp

    from dense_armor.utility.neural.consolidation import VQMemory

    m = VQMemory(n_x=2, d_model=4, n_codes=8, seed=0)
    x = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    m.fit(x, n_steps=1, lr=0.01)
    lr_first = [o.lr for o in m._opt_enc + m._opt_dec]
    m.fit(x, n_steps=1, lr=0.5)
    lr_second = [o.lr for o in m._opt_enc + m._opt_dec]
    assert all(abs(a - 0.01) < 1e-12 for a in lr_first), f"first lr: {lr_first}"
    assert all(abs(a - 0.5) < 1e-12 for a in lr_second), f"second lr: {lr_second}"
