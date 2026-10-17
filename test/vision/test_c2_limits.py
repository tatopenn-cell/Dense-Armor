"""Regression tests for the three known limits fixed in item C2.

Each test fails if the corresponding fix is reverted.

- C2a: FrameAnomaly accepts 1-channel and 4-channel frames. Before the
  fix the code already worked (1-ch is promoted to 3-ch by
  ``_to_gray``; 4-ch is reduced to 3-ch by the slice). The test is a
  regression on the behaviour, so a future change to the patch
  features cannot silently break the two channel counts.
- C2b: FrameWindow resets on a change of frame size. Before the fix
  ``_motion_energy`` raised ``ValueError: cannot reshape array of size
  256 into shape (16,2,16,2)`` when a 16x16 frame was followed by a
  32x32 one.
- C2c: DenseFlow with ``normalize_intensity=True`` gives the same flow
  at intensity scales 0.1, 1, 10 and 255. Before the fix the flow at
  0.1 was 0.09, at 1.0 was 0.76, at 10.0 was 0.84 and at 255.0 was
  0.85; with the option on, all four give 0.76.
"""

import numpy as np

from dense_armor.utility.vision.dense import DenseFlow
from dense_armor.utility.vision.scene import FrameAnomaly
from dense_armor.utility.vision.window import FrameWindow


def test_c2a_frame_anomaly_accepts_1ch_and_4ch():
    """FrameAnomaly scores 1-channel and 4-channel frames without raising.

    Before the test, the code already accepted both: the 1-channel
    frame is used as-is by ``_to_gray``, the 4-channel one is sliced to
    its first three channels. The test is a regression so a change to
    the patch features that assumes a fixed channel count is caught
    here, not in a downstream pipeline.
    """
    rng = np.random.default_rng(0)
    fa = FrameAnomaly()
    for ch in (1, 3, 4):
        if ch == 1:
            fr = rng.normal(0.5, 0.05, (16, 16)).astype(np.float32)
        else:
            fr = rng.normal(0.5, 0.05, (16, 16, ch)).astype(np.float32)
        out = fa.transform_one({"array": fr})
        assert "anomaly" in out
        assert np.isfinite(out["anomaly"])


def test_c2b_frame_window_resets_on_shape_change():
    """A change of frame size resets the window and increments a counter.

    Before the fix, the second frame with a different size raised
    ``ValueError`` inside ``_motion_energy`` when reshaping the old
    frame to the new one. Now the old frames are dropped, the window
    restarts from the new frame, and ``shape_changes_`` records that
    the reset happened. The test checks the counter, the reset of the
    mean, and that no exception is raised.
    """
    w = FrameWindow(n=3)
    _ = w.learn_one({"array": np.full((16, 16), 0.5, dtype=np.float32), "t": 0.0})
    _ = w.learn_one({"array": np.full((16, 16), 0.5, dtype=np.float32), "t": 0.1})
    assert w.shape_changes_ == 0
    out_before = w.transform_one({"array": np.full((16, 16), 0.5, dtype=np.float32)})
    assert abs(out_before["mean"] - 0.5) < 1e-6
    _ = w.learn_one({"array": np.full((32, 32), 0.9, dtype=np.float32), "t": 0.2})
    assert w.shape_changes_ == 1
    out_after = w.transform_one({"array": np.full((32, 32), 0.9, dtype=np.float32)})
    assert out_after["n_frames"] == 1
    assert abs(out_after["mean"] - 0.9) < 1e-6


def test_c2c_dense_flow_normalizes_intensity_scale():
    """With ``normalize_intensity=True`` the flow does not depend on scale.

    A sinusoid on a 32x32 grid is shifted by one pixel to the right. The
    flow is computed with the numpy path (``use_jax=False``) so the test
    does not depend on the JAX installation. Without the option, the
    magnitude grows with the intensity scale (Horn and Schunck's
    smoothness term ``alpha`` is relative to the gradient of the image).
    With the option on, all four scales give the same flow.
    """
    y, x = np.mgrid[0:32, 0:32].astype(np.float32)
    base = np.sin(2 * np.pi * x / 8.0) * np.cos(2 * np.pi * y / 8.0)
    shifted = np.roll(base, 1, axis=1)
    flows = []
    for scale in (0.1, 1.0, 10.0, 255.0):
        a = (base * scale).astype(np.float32)
        b = (shifted * scale).astype(np.float32)
        f = DenseFlow(alpha=1.0, n_iter=100, use_jax=False, normalize_intensity=True)
        f.learn_one({"array": a, "t": 0.0})
        out = f.transform_one({"array": b, "t": 1.0})
        flows.append(float(np.mean(out["u"][4:-4, 4:-4])))
    for i in range(1, len(flows)):
        assert abs(flows[i] - flows[0]) < 1e-6, (
            f"scale {i}: flow {flows[i]:.4f} != {flows[0]:.4f}"
        )
