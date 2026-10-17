"""Tests for offline_replay."""

import numpy as np
import pytest

from dense_armor.utility.bandit.evaluate import offline_replay


def _stream(n=2000, p0=0.8, p1=0.2, seed=0):
    rng = np.random.default_rng(seed)
    events = []
    for _ in range(n):
        ctx = {"x": float(rng.uniform())}
        arm = int(rng.integers(2))
        r = float(rng.binomial(1, [p0, p1][arm]))
        events.append((ctx, arm, r))
    return events


def test_replay_estimates_the_policy_mean():
    out = offline_replay(_stream(), lambda c: 0, seed=0)
    assert abs(out["mean"] - 0.8) < 0.05
    assert out["ci_low"] < 0.8 < out["ci_high"]


def test_replay_second_arm_estimates_p1():
    out = offline_replay(_stream(n=10000), lambda c: 1, seed=0)
    assert abs(out["mean"] - 0.2) < 0.03
    assert out["ci_low"] < out["mean"] < out["ci_high"]


def test_replay_keeps_about_half_the_events():
    out = offline_replay(_stream(n=4000), lambda c: 0, seed=0)
    assert 1700 <= out["n_retained"] <= 2300
    assert out["n_total"] == 4000


def test_replay_rejects_empty_stream():
    with pytest.raises(ValueError):
        offline_replay([], lambda c: 0)


def test_replay_rejects_policy_with_no_match():
    with pytest.raises(ValueError):
        offline_replay(_stream(n=200), lambda c: 99)


def test_replay_rejects_bad_alpha():
    with pytest.raises(ValueError):
        offline_replay(_stream(n=200), lambda c: 0, alpha=0.0)


def test_offline_replay_skips_a_retained_event_with_a_bad_reward():
    events = [({}, 0, "bad"), ({}, 0, 1.0), ({}, 0, 0.0)]
    out = offline_replay(events, lambda c: 0, seed=0)
    assert out["n_retained"] == 2
