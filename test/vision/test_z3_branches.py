"""Branches touched by task Z3 item E: window seconds, DenseFlow jax."""

import numpy as np

from dense_armor.utility.vision.dense import DenseFlow
from dense_armor.utility.vision.window import FrameWindow


def test_frame_window_seconds_branch_drops_old():
    w = FrameWindow(seconds=0.5)
    for i in range(5):
        w.learn_one({"array": np.zeros((4, 4), dtype=np.float32),
                     "t": float(i) * 0.1})
    assert len(w._times) >= 1
    for i in range(10):
        w.learn_one({"array": np.zeros((4, 4), dtype=np.float32),
                     "t": 10.0 + float(i) * 0.1})
    assert len(w._times) <= 6


def test_dense_flow_jax_path_runs_and_sets_the_function():
    a = np.random.default_rng(0).normal(0, 1, (16, 16)).astype(np.float32)
    b = np.roll(a, (1, 1), axis=(0, 1))
    f = DenseFlow(alpha=1.0, n_iter=5, use_jax=True)
    f.learn_one(a)
    out = f.transform_one({"array": b})
    assert "u" in out and "v" in out
    assert f._jax_fn is not None
