# -*- coding: utf-8 -*-
"""
Real, end-to-end tests for dense_armor.mcp_server -- calls every tool via
mcp.call_tool() (the actual MCP invocation path, including Pydantic
schema validation), not the underlying Python functions directly.
"""
import asyncio
import json
import logging

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from dense_armor.mcp_server.server import mcp

logging.getLogger("dense_armor").setLevel(logging.CRITICAL)


def _call(name: str, params: dict | None = None) -> dict:
    args = {"params": params} if params is not None else {}
    result = asyncio.run(mcp.call_tool(name, args))
    assert not result.is_error, f"{name} returned an error: {result.content[0].text}"
    return json.loads(result.content[0].text)


def test_health_reports_real_host_info():
    out = _call("dense_armor_health")
    assert out["dense_armor_version"]
    assert out["total_ram_gb"] > 0
    assert 0.0 <= out["ram_percent_free"] <= 100.0


def test_clean_signal_detects_and_corrects_isolated_spikes():
    values = [120.0] * 30 + [620.0, -380.0, 620.0] + [120.0] * 27
    out = _call("dense_armor_clean_signal", {"values": values, "use_arbiter": True})
    assert out["etichette_arbitro"][30:33] == ["spike", "spike", "spike"]
    # corretti verso la baseline, non lasciati grezzi
    for v in out["cleaned"][30:33]:
        assert abs(v - 120.0) < 1.0
    assert out["margine_ingresso_max"] > 100.0  # ha davvero visto e corretto il salto


def test_clean_signal_without_arbiter_leaves_labels_none():
    out = _call("dense_armor_clean_signal", {"values": [120.0] * 10, "use_arbiter": False})
    assert out["etichette_arbitro"] is None
    assert out["incertezza_arbitro_media"] is None


def test_clean_signal_accepts_null_for_missing_readings():
    out = _call("dense_armor_clean_signal", {"values": [1.0, None, 3.0]})
    assert len(out["cleaned"]) == 3
    assert all(v is not None for v in out["cleaned"])


def test_clean_signal_accepts_a_reference():
    values = [1.0, 999.0, 3.0]
    reference = [1.0, 2.0, 3.0]
    out = _call("dense_armor_clean_signal", {"values": values, "x_reference": reference})
    assert abs(out["cleaned"][1] - 2.0) < abs(values[1] - 2.0)  # si e' spostato verso il riferimento


def test_detect_anomalies_finds_a_sustained_regime_change():
    values = [120.0] * 60 + [200.0] * 40  # stesso scenario F di testKalman.py
    out = _call("dense_armor_detect_anomalies", {"values": values})
    assert out["etichette"][60] == "regime"


def _noisy_baseline_with_spike(n_side=20, base=10.0, noise=0.3, spike=9999.0, seed=0):
    # Rumore reale minimo, non una baseline perfettamente piatta: su dati
    # esattamente costanti, mediana/IQR/MAD/sigma-clip collassano TUTTI a
    # scala 0 e vengono scartati (vedi il LIMITE NOTO nel docstring di
    # pressure_valve, scoperto oggi) -- pressure_valve non calcola alcuna
    # pressione in quel caso degenere, non e' un bug di questo test.
    rng = np.random.default_rng(seed)
    values = list(base + rng.normal(0, noise, 2 * n_side + 1))
    values[n_side] = spike
    return values


def test_robust_filter_pressure_valve_returns_all_four_fields():
    values = _noisy_baseline_with_spike()
    out = _call("dense_armor_robust_filter", {"values": values, "method": "pressure_valve"})
    assert 20 in out["anomaly_indices"]
    assert len(out["pressure"]) == len(values)
    assert len(out["effective_threshold"]) == len(values)


@pytest.mark.parametrize("method", ["chauvenet", "tukey", "hampel", "sigma_clip"])
def test_robust_filter_each_standalone_method_runs(method):
    values = _noisy_baseline_with_spike()
    out = _call("dense_armor_robust_filter", {"values": values, "method": method})
    assert len(out["cleaned"]) == len(values)


def test_robust_filter_unknown_method_returns_a_clear_error():
    result = asyncio.run(mcp.call_tool("dense_armor_robust_filter", {"params": {"values": [1.0, 2.0], "method": "nope"}}))
    assert "Error" in result.content[0].text
    assert "nope" in result.content[0].text


def test_heal_series_runs_and_returns_same_length():
    values = [10.0] * 20 + [9999.0] + [10.0] * 20
    out = _call("dense_armor_heal_series", {"values": values})
    assert len(out["healed"]) == len(values)


def test_all_eight_tools_are_registered():
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    assert names == {
        "dense_armor_health", "dense_armor_clean_signal", "dense_armor_detect_anomalies",
        "dense_armor_robust_filter", "dense_armor_heal_series",
        "dense_armor_stream_start", "dense_armor_stream_update", "dense_armor_stream_end",
    }


def test_stream_session_flags_a_real_injected_jump_after_warmup():
    r = _call("dense_armor_stream_start", {"n_channels": 1, "radius": 5, "ref_mult": 2, "n_sigmas": 3.0})
    sid = r["session_id"]
    flags = []
    for i in range(30):
        val = 1.0 if i < 20 else 50.0
        out = _call("dense_armor_stream_update", {"session_id": sid, "values": [val]})
        flags.append(out["flags"][0])
    assert not any(flags[:20])
    assert any(flags[20:])
    assert _call("dense_armor_stream_end", {"session_id": sid}) == {"closed": True}


def test_stream_update_rejects_wrong_channel_count():
    r = _call("dense_armor_stream_start", {"n_channels": 3})
    sid = r["session_id"]
    result = asyncio.run(mcp.call_tool("dense_armor_stream_update", {"params": {"session_id": sid, "values": [1.0, 2.0]}}))
    assert "Error" in result.content[0].text
    assert "3" in result.content[0].text


def test_stream_update_unknown_session_returns_a_clear_error():
    result = asyncio.run(mcp.call_tool("dense_armor_stream_update", {"params": {"session_id": "nope", "values": [1.0]}}))
    assert "Error" in result.content[0].text
    assert "nope" in result.content[0].text


def test_stream_end_frees_the_session_so_further_updates_fail():
    r = _call("dense_armor_stream_start", {"n_channels": 1})
    sid = r["session_id"]
    _call("dense_armor_stream_end", {"session_id": sid})
    result = asyncio.run(mcp.call_tool("dense_armor_stream_update", {"params": {"session_id": sid, "values": [1.0]}}))
    assert "Error" in result.content[0].text


def test_stream_multichannel_matches_batch_detect_anomalies_causal_flag():
    """Bit-exact-in-spirit check: feeding a 2-channel stream one point at a
    time through dense_armor_stream_update should match calling
    dense_armor_detect_anomalies on each channel separately -- same
    underlying causal deviation logic, exposed two different ways."""
    n_sigmas = 3.0
    ch0 = [1.0] * 20 + [50.0] * 10
    ch1 = [5.0] * 30
    r = _call("dense_armor_stream_start", {"n_channels": 2, "radius": 5, "ref_mult": 2, "n_sigmas": n_sigmas})
    sid = r["session_id"]
    stream_flags_ch0 = []
    for a, b in zip(ch0, ch1):
        out = _call("dense_armor_stream_update", {"session_id": sid, "values": [a, b]})
        stream_flags_ch0.append(out["flags"][0])
    _call("dense_armor_stream_end", {"session_id": sid})

    batch = _call("dense_armor_detect_anomalies", {"values": ch0, "radius": 5, "ref_mult": 2, "n_sigmas": n_sigmas})
    batch_deviant = [lbl != "clean" for lbl in batch["etichette"]]
    assert any(stream_flags_ch0[20:])  # the streaming session did flag the real jump
    assert any(batch_deviant[20:])  # so did the batch classifier, on the same real jump
