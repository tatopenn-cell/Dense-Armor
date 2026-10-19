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
