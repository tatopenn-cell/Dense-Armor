# -*- coding: utf-8 -*-
"""Unit tests for dense_armor/utility/streaming_arbiter.py.

The streaming arbiter decides with a bounded delay: each sample gets a
provisional label at once and a final label after at most ``max_delay``
future samples. With ``max_delay`` large enough to cover both the longest
run and the widest window the batch correction uses (``radius * ref_mult``),
final labels and corrected values equal the batch
``arbiter.classify_segments`` + ``arbiter.route_and_correct`` on the same
series.

Two thresholds matter, and they are separate. Full label agreement needs
``max_delay`` at least the longest *regime* run plus the post-run window
``radius``. Full *correction* agreement needs additionally the longest
*spike-replaced* run plus the wide window ``radius * ref_mult`` -- a run
labelled ``spike`` that returns to baseline has no regime-length, but its
wide correction window still extends ``radius * ref_mult`` past the last
index of the run.
"""
import doctest
import numpy as np
import pytest

import dense_armor.utility.protect.streaming_arbiter as streaming_arbiter
from dense_armor.utility.protect.arbiter import classify_segments, route_and_correct
from dense_armor.utility.protect.healing import healing_filter
from dense_armor.utility.protect.streaming_arbiter import StreamingArbiter, StreamingHealing


def _seeded_spike_signal(seed=0, n=400):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.5, n)
    x[50] = 50.0
    x[150] = -50.0
    x[200:203] = [30.0, -30.0, 30.0]
    x[280:295] = 5.0
    return x


def _seeded_regime_signal(seed=1, n=400):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(0, 0.5, n // 2),
                           rng.normal(5, 0.5, n - n // 2)])


def _stream(x, max_delay, radius=10, ref_mult=3, n_sigmas=3.0,
            spike_run_max=2):
    s = StreamingArbiter(radius=radius, ref_mult=ref_mult, n_sigmas=n_sigmas,
                         spike_run_max=spike_run_max, max_delay=max_delay)
    out = {}
    for v in x:
        for i, lab, corr in s.update(float(v)):
            out[i] = (lab, corr)
    for i, lab, corr in s.flush():
        out[i] = (lab, corr)
    labels = [out[i][0] for i in range(len(x))]
    corr = np.array([out[i][1] for i in range(len(x))])
    return labels, corr


def test_equivalence_spike_signal():
    x = _seeded_spike_signal()
    bl, _, _ = classify_segments(x)
    bc, _, _ = route_and_correct(x)
    sl, sc = _stream(x, max_delay=len(x) + 100)
    assert sl == list(bl)
    np.testing.assert_allclose(sc, bc, atol=1e-12)


def test_equivalence_regime_signal():
    x = _seeded_regime_signal()
    bl, _, _ = classify_segments(x)
    bc, _, _ = route_and_correct(x)
    sl, sc = _stream(x, max_delay=len(x) + 100)
    assert sl == list(bl)
    np.testing.assert_allclose(sc, bc, atol=1e-12)


def test_smaller_delay_labels_are_valid_and_count_is_full():
    x = _seeded_spike_signal()
    for md in (1, 5, 20, 60, 200):
        sl, sc = _stream(x, max_delay=md)
        assert len(sl) == len(x)
        assert len(sc) == len(x)
        assert all(lab in ("clean", "spike", "regime") for lab in sl)


def test_max_delay_200_is_exact_on_seeded_spike_signal():
    # 200 > 45 = longest spike-replaced run (15) + wide window (30); with
    # this margin the full wide window is available before any forced
    # decision, and the streaming result matches the batch one bit for bit.
    x = _seeded_spike_signal()
    bl, _, _ = classify_segments(x)
    bc, _, _ = route_and_correct(x)
    sl, sc = _stream(x, max_delay=200)
    assert sl == list(bl)
    np.testing.assert_allclose(sc, bc, atol=1e-12)


def test_max_delay_40_is_exact_on_latency_signal():
    # The latency-table signal has a longest spike-replaced run of 5
    # samples (at 1500-1504); 5 + 30 = 35 < 40, so max_delay=40 is exact.
    rng = np.random.default_rng(42)
    n = 2000
    x = rng.normal(0.0, 0.5, n)
    x[100] = 30.0
    x[300:303] = [25.0, -25.0, 25.0]
    x[500:520] = 3.0
    x[800] = -40.0
    x[1500:1505] = 4.0
    bl, _, _ = classify_segments(x, radius=10, ref_mult=3,
                                 n_sigmas=3.0, spike_run_max=2)
    bc, _, _ = route_and_correct(x, radius=10, ref_mult=3,
                                 n_sigmas=3.0, spike_run_max=2)
    sl, sc = _stream(x, max_delay=40)
    assert sl == list(bl)
    np.testing.assert_allclose(sc, bc, atol=1e-12)


def test_flush_finalizes_everything():
    # The signal ends with an open deviant run (a spike in the last position).
    # The streaming arbiter never emits a deviant sample whose run is still
    # open, and with max_delay large the forced-decision branch never fires
    # during update -- so the trailing spike is only finalised by flush().
    rng = np.random.default_rng(0)
    x = list(rng.normal(0, 0.5, 50))
    x.append(50.0)
    n = len(x)
    s = StreamingArbiter(radius=10, ref_mult=3, n_sigmas=3.0,
                         spike_run_max=2, max_delay=n + 100)
    seen = set()
    for v in x:
        for i, _, _ in s.update(float(v)):
            seen.add(i)
    assert len(set(range(n)) - seen) > 0
    for i, _, _ in s.flush():
        seen.add(i)
    assert seen == set(range(n))


def test_warmup_is_clean():
    s = StreamingArbiter(radius=10, ref_mult=3, max_delay=5)
    for v in [1.0, 2.0, 3.0]:
        s.update(v)
    decided = s.flush()
    assert all(lab == "clean" for _, lab, _ in decided)


def test_empty_and_single_sample():
    for x in ([], [1.0]):
        s = StreamingArbiter(max_delay=5)
        decided = []
        for v in x:
            decided.extend(s.update(v))
        decided.extend(s.flush())
        assert len(decided) == len(x)


def test_doctest():
    assert doctest.testmod(streaming_arbiter).failed == 0


def test_streaming_healing_matches_batch():
    rng = np.random.default_rng(0)
    x = list(rng.normal(0, 0.5, 100)) + [50.0] + list(rng.normal(0, 0.5, 100))
    sh = StreamingHealing(radius=2, wide_mult=3, sustain_threshold=0.7,
                          max_delay=200)
    out = {}
    for v in x:
        for i, val in sh.update(float(v)):
            out[i] = val
    for i, val in sh.flush():
        out[i] = val
    stream = np.array([out[i] for i in range(len(x))])
    batch = healing_filter(np.asarray(x), radius=2, wide_mult=3,
                           sustain_threshold=0.7)
    np.testing.assert_allclose(stream, batch, atol=1e-12)


def test_latency_table_seeded_robot_signal():
    rng = np.random.default_rng(42)
    n = 2000
    x = rng.normal(0.0, 0.5, n)
    x[100] = 30.0
    x[300:303] = [25.0, -25.0, 25.0]
    x[500:520] = 3.0
    x[800] = -40.0
    x[1500:1505] = 4.0
    bl, _, _ = classify_segments(x, radius=10, ref_mult=3,
                                 n_sigmas=3.0, spike_run_max=2)
    sl20, _ = _stream(x, max_delay=20)
    agree = sum(1 for a, b in zip(bl, sl20) if a == b)
    assert agree == n
