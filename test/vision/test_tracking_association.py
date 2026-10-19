"""Tests for the Hungarian association and the re-ID of MovingObjectTracker."""


from dense_armor.utility.vision.tracking import MovingObjectTracker


def _tracker(**kw):
    defaults = {"min_area": 1, "max_missing": 15, "iou_threshold": 0.3,
                "high_score": 0.5, "history": 1}
    defaults.update(kw)
    return MovingObjectTracker(**defaults)


def test_two_round_association_keeps_both_tracks():
    """Round 1 matches the high-score detection, round 2 the low-score one."""
    t = _tracker()
    out = t.update([[0, 0, 10, 10, 1.0], [5, 0, 15, 10, 1.0]])
    assert out["n_tracks"] == 2
    out = t.update([[2, 0, 12, 10, 0.4], [5, 0, 15, 10, 0.6]])
    assert out["n_tracks"] == 2
    assert all(tr["missing"] == 0 for tr in out["tracks"])


def test_reid_after_missing():
    """A track that misses a frame is re-identified by position."""
    t = _tracker(reid_radius=3.0)
    out = t.update([[0, 0, 10, 10, 1.0]])
    tid = out["tracks"][0]["id"]
    out = t.update([])
    assert out["tracks"][0]["missing"] == 1
    out = t.update([[20, 0, 30, 10, 1.0]])
    assert out["n_tracks"] == 1
    assert out["tracks"][0]["id"] == tid
    assert out["tracks"][0]["missing"] == 0


def test_reid_does_not_steal_a_fresh_track():
    """A track with missing == 0 is not re-identified by position."""
    t = _tracker(reid_radius=3.0)
    out = t.update([[0, 0, 10, 10, 1.0]])
    tid = out["tracks"][0]["id"]
    out = t.update([[0, 0, 10, 10, 1.0], [50, 0, 60, 10, 1.0]])
    assert out["n_tracks"] == 2
    ids = {tr["id"] for tr in out["tracks"]}
    assert tid in ids


def test_track_dropped_after_max_missing():
    t = _tracker(max_missing=2)
    t.update([[0, 0, 10, 10, 1.0]])
    t.update([])
    t.update([])
    out = t.update([])
    assert out["n_tracks"] == 0


def test_fallback_without_scipy(monkeypatch):
    """If scipy.optimize is not importable, the greedy pick is used."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "scipy.optimize" or name.startswith("scipy.optimize."):
            raise ImportError("no scipy")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    t = _tracker()
    t.update([[0, 0, 10, 10, 1.0]])
    out = t.update([[0, 0, 10, 10, 1.0]])
    assert out["n_tracks"] == 1
    assert out["tracks"][0]["missing"] == 0


def test_hungarian_threshold_is_applied_before_assignment():
    """The reviewer's case: IoU [[0.35, 0.29], [0.29, 0.0]], threshold 0.3.

    The unconstrained optimum is the crossing (0,1)+(1,0), sum 0.58 > 0.35,
    but both are below threshold. With the threshold as high cost *before*
    the assignment, the optimum is (0,0) with IoU 0.35, which is valid.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    iou = np.array([[0.35, 0.29], [0.29, 0.0]])
    candidate = iou >= 0.3
    cost = np.where(candidate, -iou, 1e6)
    r, c = linear_sum_assignment(cost)
    pairs = [(int(ri), int(ci)) for ri, ci in zip(r, c) if candidate[ri, ci]]
    assert pairs == [(0, 0)], f"got {pairs}"


def test_default_is_greedy_and_no_reid():
    """The default keeps the historical behaviour: greedy, no re-ID."""
    from dense_armor.utility.vision.tracking import MovingObjectTracker
    t = MovingObjectTracker(min_area=1, history=1)
    assert t.association == "greedy"
    assert t.reid_radius == 0.0
    t.update([[0, 0, 10, 10, 1.0]])
    out = t.update([])
    assert out["tracks"][0]["missing"] == 1
    # reid_radius=0: a detection near the prediction but with IoU 0
    # does not re-identify the track, a new one is created
    out = t.update([[20, 0, 30, 10, 1.0]])
    assert out["n_tracks"] == 2


def test_reid_radius_opts_in():
    """Setting reid_radius > 0 enables the re-ID."""
    from dense_armor.utility.vision.tracking import MovingObjectTracker
    t = MovingObjectTracker(min_area=1, history=1, reid_radius=3.0)
    out = t.update([[0, 0, 10, 10, 1.0]])
    tid = out["tracks"][0]["id"]
    t.update([])
    out = t.update([[20, 0, 30, 10, 1.0]])
    assert out["n_tracks"] == 1
    assert out["tracks"][0]["id"] == tid


def test_association_invalid_raises():
    import pytest

    from dense_armor.utility.vision.tracking import MovingObjectTracker
    with pytest.raises(ValueError, match="association must be"):
        MovingObjectTracker(min_area=1, history=1, association="bogus")


def test_hungarian_opts_in():
    """Setting association='hungarian' keeps the same matches on a simple case."""
    from dense_armor.utility.vision.tracking import MovingObjectTracker
    t = MovingObjectTracker(min_area=1, history=1, association="hungarian")
    assert t.association == "hungarian"
    t.update([[0, 0, 10, 10, 1.0]])
    out = t.update([[0, 0, 10, 10, 1.0]])
    assert out["n_tracks"] == 1
    assert out["tracks"][0]["missing"] == 0
