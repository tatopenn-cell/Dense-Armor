"""The 5 dense_armor_mcp tools, registered against the shared `mcp`
instance created in server.py (see that module's docstring for why
importing `mcp` back from there is safe despite looking circular).

Every tool calls dense_armor directly, in this same process -- no HTTP
client, no separate kernel (see server.py's module docstring for why
that's the right choice here, unlike Dense-Evolution's MCP adapter).
"""
import functools
import json

import numpy as np

from .models import CleanSignalInput, DetectAnomaliesInput, HealSeriesInput, RobustFilterInput
from .server import mcp

READ_ONLY_IDEMPOTENT = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}
# Every tool here is a pure function of its inputs (no stored server-side
# state, no external resource) -- true idempotence, unlike Dense-Evolution's
# COMPUTE preset (its floating-point reduction order can vary run-to-run on
# a real accelerator backend). Same numpy/JAX inputs give the same outputs.
COMPUTE_IDEMPOTENT = READ_ONLY_IDEMPOTENT


def catch_errors(func):
    """Wraps a tool function so any exception becomes an "Error: ..."
    string instead of an uncaught traceback -- same convention as
    Dense-Evolution's mcp_server/client.py, adapted for a function with
    no HTTP layer to also catch connection-specific exceptions for."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            return f"Error: {e}"
    return wrapper


def _to_array(values) -> np.ndarray:
    """JSON `null` entries arrive as Python None inside the list -- convert
    to NaN before anything touches the array as float64, matching how
    dense_armor's own CLI (`python -m dense_armor --json`) accepts gaps."""
    return np.array([np.nan if v is None else v for v in values], dtype=np.float64)


def _nan_to_none(arr) -> list:
    """Reverse of _to_array's None->NaN, for JSON output -- json.dumps
    cannot serialize a bare float('nan') (produces invalid JSON `NaN`
    unless allow_nan=True, which most strict JSON parsers reject)."""
    return [None if (isinstance(v, float) and np.isnan(v)) else float(v) for v in np.asarray(arr).tolist()]


@mcp.tool(name="dense_armor_health", annotations={"title": "Check Dense-Armor availability and host resources", **READ_ONLY_IDEMPOTENT})
@catch_errors
async def dense_armor_health() -> str:
    """Check that dense_armor is importable in this process and report
    the host's current resources -- call this first if unsure the server
    is working. No kernel to reach (see server.py): if this tool
    responds at all, dense_armor is available.

    Returns:
        str: JSON with {dense_armor_version, hardware_profile,
        available_ram_gb, total_ram_gb, ram_percent_free}.
    """
    import psutil
    import dense_armor
    from dense_armor.core.noise import AIHardwareProfiler

    vm = psutil.virtual_memory()
    profile = AIHardwareProfiler()
    return json.dumps({
        "dense_armor_version": dense_armor.__version__,
        "hardware_profile": profile.get_profile_summary(),
        "available_ram_gb": round(vm.available / (1024 ** 3), 2),
        "total_ram_gb": round(vm.total / (1024 ** 3), 2),
        "ram_percent_free": round(100.0 * vm.available / vm.total, 1),
    }, indent=2)


@mcp.tool(name="dense_armor_clean_signal", annotations={"title": "Clean a corrupted 1D signal with Orca", **COMPUTE_IDEMPOTENT})
@catch_errors
async def dense_armor_clean_signal(params: CleanSignalInput) -> str:
    """Run Dense-Armor's full Orca shield over a raw 1D series -- the same
    pipeline used to protect an AI model's input, called directly on data
    with no model in between (Orca's own "simple data test" mode).

    Use x_reference when you have a known-clean version of the same
    series (e.g. a calibration run); omit it for blind mode, where Orca
    estimates a causal baseline from the series' own history.

    Args:
        params (CleanSignalInput): values, optional x_reference, use_arbiter.

    Returns:
        str: JSON with {cleaned, margine_ingresso_medio, margine_ingresso_max,
        etichette_arbitro, incertezza_arbitro_media} -- etichette_arbitro/
        incertezza_arbitro_media are null unless use_arbiter=True.
    """
    from dense_armor.utility.orca import Orca

    orca = Orca()
    x = _to_array(params.values)
    x_ref = _to_array(params.x_reference) if params.x_reference is not None else None
    cleaned = orca.protect_and_forward(
        None, x, x_reference=x_ref, use_model_injection=False, use_arbiter=params.use_arbiter,
    )
    return json.dumps({
        "cleaned": _nan_to_none(cleaned),
        "margine_ingresso_medio": orca.margine_ingresso_medio,
        "margine_ingresso_max": orca.margine_ingresso_max,
        "etichette_arbitro": orca.etichette_arbitro.tolist() if orca.etichette_arbitro is not None else None,
        "incertezza_arbitro_media": orca.incertezza_arbitro_media if orca.etichette_arbitro is not None else None,
    }, indent=2)


@mcp.tool(name="dense_armor_detect_anomalies", annotations={"title": "Classify each point as clean/spike/regime", **COMPUTE_IDEMPOTENT})
@catch_errors
async def dense_armor_detect_anomalies(params: DetectAnomaliesInput) -> str:
    """Classify every point of a series as 'clean', an isolated 'spike', or
    a sustained 'regime' change -- the routing logic behind
    dense_armor_clean_signal's use_arbiter=True, exposed standalone for
    when you only want the classification, not a corrected series.

    Args:
        params (DetectAnomaliesInput): values, radius, ref_mult, n_sigmas, spike_run_max.

    Returns:
        str: JSON with {etichette, deviazione, incertezza}, one entry per input point.
    """
    from dense_armor.utility.arbiter import classify_segments

    x = _to_array(params.values)
    etichette, deviazione, incertezza = classify_segments(
        x, radius=params.radius, ref_mult=params.ref_mult,
        n_sigmas=params.n_sigmas, spike_run_max=params.spike_run_max,
    )
    return json.dumps({
        "etichette": etichette.tolist(),
        "deviazione": _nan_to_none(deviazione),
        "incertezza": _nan_to_none(incertezza),
    }, indent=2)


@mcp.tool(name="dense_armor_robust_filter", annotations={"title": "Classic statistical outlier detection (standalone)", **COMPUTE_IDEMPOTENT})
@catch_errors
async def dense_armor_robust_filter(params: RobustFilterInput) -> str:
    """Run one of the four classic anomaly detectors (Chauvenet's
    criterion, Tukey's fences, the Hampel filter, iterative sigma-clipping)
    or their combined orchestrator, `pressure_valve` -- no dynamic model,
    no state, independent of Orca/Armatura.

    `pressure_valve` combines all four via a Lagrange-multiplier-derived
    minimum-variance estimator with a Jensen-Shannon-adaptive threshold
    (widens near a genuine regime change, never narrows below the base
    threshold). Known limit (see utility/robust_filters.py docstring):
    clustered outliers close together can still mask each other in the
    local window -- not fixed for this method, only worked around in
    dense_armor_detect_anomalies's wide causal window.

    Args:
        params (RobustFilterInput): values, method, radius.

    Returns:
        str: JSON with {cleaned, anomaly_indices}, plus
        {pressure, effective_threshold} if method='pressure_valve'.
    """
    from dense_armor.utility import robust_filters as rf

    x = _to_array(params.values)
    if params.method == "pressure_valve":
        cleaned, anomalie, pressione, soglia = rf.pressure_valve(x, radius=params.radius)
        return json.dumps({
            "cleaned": _nan_to_none(cleaned), "anomaly_indices": anomalie,
            "pressure": _nan_to_none(pressione), "effective_threshold": _nan_to_none(soglia),
        }, indent=2)
    method_fn = {
        "chauvenet": rf.chauvenet_criterion, "tukey": rf.tukey_fences,
        "hampel": rf.hampel_filter, "sigma_clip": rf.sigma_clip,
    }.get(params.method)
    if method_fn is None:
        return f"Error: unknown method '{params.method}' -- expected one of pressure_valve/chauvenet/tukey/hampel/sigma_clip"
    cleaned, anomalie = method_fn(x, radius=params.radius)
    return json.dumps({"cleaned": _nan_to_none(cleaned), "anomaly_indices": anomalie}, indent=2)


@mcp.tool(name="dense_armor_heal_series", annotations={"title": "Neighbor-consensus healing filter (standalone)", **COMPUTE_IDEMPOTENT})
@catch_errors
async def dense_armor_heal_series(params: HealSeriesInput) -> str:
    """Classify each point by how many neighbors share its deviation from
    a wide local baseline -- an isolated deviation (no neighbor agrees) is
    treated as noise and replaced; a deviation most neighbors share is
    treated as a genuine change and passed through untouched.

    Real, measured strengths (see utility/healing.py docstring, tuned via
    a 40-seed grid search): strong on heavy-tailed/pervasive noise and on
    a genuine sustained jump plus scattered spikes; loses to a plain
    rolling median on smooth low/medium-noise signals with no real jump.
    Known limit: a TEMPORARY collapse that later reverts to the original
    level can be mistaken for a genuine change (no run-persistence check,
    unlike dense_armor_detect_anomalies) -- not fixed, to avoid risking
    the existing calibration.

    Args:
        params (HealSeriesInput): values, radius, sustain_threshold, wide_mult.

    Returns:
        str: JSON with {healed: [...]}.
    """
    from dense_armor.utility.healing import healing_filter

    x = _to_array(params.values)
    healed = healing_filter(x, radius=params.radius, sustain_threshold=params.sustain_threshold, wide_mult=params.wide_mult)
    return json.dumps({"healed": _nan_to_none(healed)}, indent=2)
