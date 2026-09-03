# -*- coding: utf-8 -*-
"""
Unit tests for dense_armor/utility/streaming.py -- promoted from
Dense-Evolution-Discovery (Experiments 48-49) after validation on two
independent real physical domains (SO-101 robot arm, real UCI HAR
IMU). The real correctness bar re-verified here: StreamingDeviation
Detector, fed one point at a time, must match classify_segments' own
per-point `deviante` computation bit-for-bit -- not "looks similar" --
on Dense-Armor's own already-frozen real telemetry, so this doesn't
need a new external dataset dependency.
"""
import json
import pathlib
from collections import defaultdict

import numpy as np

from dense_armor.utility.streaming import (
    StreamingDeviationDetector,
    MultiChannelStreamingDeviationDetector,
    classify_segments_multichannel,
)
from dense_armor.utility.arbiter import classify_segments

_AGENT_TELEMETRY = pathlib.Path(__file__).resolve().parent / "agent_v2" / "telemetry_v2_frozen.jsonl"
RADIUS, REF_MULT, N_SIGMAS = 5, 2, 3.0


def _batch_deviante(x: np.ndarray) -> np.ndarray:
    """Reimplements classify_segments' OWN per-point deviante logic
    directly (arbiter.py's `_window_causal` + `_robust_center_scale`),
    so this checks the exact per-point computation being ported, not
    the full function's later run-length post-processing."""
    n = x.size
    deviante = np.zeros(n, dtype=bool)
    span = RADIUS * REF_MULT
    for i in range(n):
        lo = max(0, i - span)
        w = x[lo:i]
        if w.size < 4:
            continue
        med = float(np.median(w))
        mad = float(np.median(np.abs(w - med)))
        scala = 1.4826 * mad
        scarto = abs(x[i] - med)
        deviante[i] = (scarto > 1e-9) if scala < 1e-9 else ((scarto / scala) > N_SIGMAS)
    return deviante


def _load_agent_scenarios():
    records = [json.loads(l) for l in open(_AGENT_TELEMETRY, encoding="utf-8")]
    by_scenario = defaultdict(list)
    for r in records:
        by_scenario[r["scenario"]].append(r)
    for scen in by_scenario:
        by_scenario[scen].sort(key=lambda r: r["step_id"])
    return by_scenario


def test_detects_a_clear_outlier():
    det = StreamingDeviationDetector(radius=5, ref_mult=2, n_sigmas=3.0)
    rng = np.random.default_rng(0)
    stream = list(rng.normal(0, 1, 20)) + [50.0]
    flags = [det.update(x) for x in stream]
    assert flags[-1] is True
    assert sum(flags[:20]) < 5, "shouldn't flood-flag pure noise during warmup+steady state"


def test_first_points_never_flagged_before_warmup():
    det = StreamingDeviationDetector(radius=5, ref_mult=2, n_sigmas=3.0)
    flags = [det.update(x) for x in [1.0, 2.0, 3.0]]
    assert not any(flags), "fewer than 4 buffered points -- no valid reference window yet"


def test_degenerate_constant_baseline_flags_any_nonzero_deviation():
    det = StreamingDeviationDetector(radius=5, ref_mult=2, n_sigmas=3.0)
    for _ in range(15):
        det.update(5.0)
    assert det.update(5.0) is False
    assert det.update(5.1) is True, "flat baseline: ANY nonzero deviation counts as deviant"


def test_multichannel_wrong_length_raises():
    det = MultiChannelStreamingDeviationDetector(n_channels=3, radius=5, ref_mult=2, n_sigmas=3.0)
    try:
        det.update([1.0, 2.0])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_streaming_matches_batch_deviante_on_real_agent_telemetry():
    by_scenario = _load_agent_scenarios()
    for scen, records in by_scenario.items():
        x = np.array([r["latency_s"] for r in records])
        batch = _batch_deviante(x)
        det = StreamingDeviationDetector(radius=RADIUS, ref_mult=REF_MULT, n_sigmas=N_SIGMAS)
        streaming = np.array([det.update(float(v)) for v in x], dtype=bool)
        assert np.array_equal(batch, streaming), f"scenario {scen}: streaming must match batch exactly"


def test_streaming_matches_batch_deviante_on_synthetic_outliers():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 200)
    x[[50, 51, 150]] = [30.0, -30.0, 25.0]
    batch = _batch_deviante(x)
    det = StreamingDeviationDetector(radius=RADIUS, ref_mult=REF_MULT, n_sigmas=N_SIGMAS)
    streaming = np.array([det.update(float(v)) for v in x], dtype=bool)
    assert np.array_equal(batch, streaming)


def test_classify_segments_multichannel_matches_per_channel_loop():
    """Real multi-channel data isn't yet in this repo's own fixtures --
    stacking the 4 real agent-telemetry scenarios as 4 independent
    channels is a legitimate way to test WRAPPER correctness (each
    channel is processed independently by design; this doesn't claim
    the 4 channels are physically related)."""
    by_scenario = _load_agent_scenarios()
    scenarios = sorted(by_scenario.keys())
    X = np.stack([[r["latency_s"] for r in by_scenario[s]] for s in scenarios], axis=1)

    kw = dict(radius=RADIUS, ref_mult=REF_MULT, n_sigmas=N_SIGMAS, spike_run_max=2)
    etichette_mc, deviazione_mc, incertezza_mc = classify_segments_multichannel(X, **kw)
    for j, scen in enumerate(scenarios):
        e, d, u = classify_segments(X[:, j], **kw)
        assert np.array_equal(etichette_mc[:, j], e), f"{scen}: label mismatch"
        assert np.allclose(deviazione_mc[:, j], d), f"{scen}: deviation mismatch"
        assert np.allclose(incertezza_mc[:, j], u), f"{scen}: uncertainty mismatch"


def test_multichannel_streaming_matches_per_channel_streaming():
    by_scenario = _load_agent_scenarios()
    scenarios = sorted(by_scenario.keys())
    X = np.stack([[r["latency_s"] for r in by_scenario[s]] for s in scenarios], axis=1)
    n, c = X.shape

    mc_det = MultiChannelStreamingDeviationDetector(n_channels=c, radius=RADIUS, ref_mult=REF_MULT, n_sigmas=N_SIGMAS)
    mc_flags = np.array([mc_det.update(X[i]) for i in range(n)])

    solo = [StreamingDeviationDetector(radius=RADIUS, ref_mult=REF_MULT, n_sigmas=N_SIGMAS) for _ in range(c)]
    solo_flags = np.array([[solo[j].update(float(X[i, j])) for j in range(c)] for i in range(n)])

    assert np.array_equal(mc_flags, solo_flags)
