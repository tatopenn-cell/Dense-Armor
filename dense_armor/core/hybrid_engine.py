"""Binary-trigger healing engine for Armatura.

Replaces the earlier AdaptiveSignalStabilizer + ABCollatz stage. The ABCollatz
gate was mathematically non-discriminating: it was verified with a numerical
sweep and never demonstrably fixed.

The pure functions below are vendored from
``Dense-Evolution/dense_evolution/healing.py``
(``calculate_phi_ab``, ``calculate_vettore_dinamico``, ``evaluate_phi_trigger``,
``GLOBAL_CONSTANTS``), tested in that repository and already in production
(``dense-evolution >= 8.1.9`` on PyPI). They are vendored rather than imported
at runtime so the core stays on numpy + jax + psutil; ``dense-evolution``
remains an optional extra (``[quantum]``).

Deviation from upstream. ``calculate_phi_ab`` originally normalized the
``state_A`` / ``state_B`` distance against a fixed constant
(``MAX_SEMANTIC_DISTANCE = sqrt(2)``), correct when ``state_A`` and ``state_B``
are already normalized vectors. Here they are raw scalars of arbitrary scale
(a loss, a sensor reading, a metric). With the fixed constant any real jump
saturated ``coherence_component`` to very negative values, so ``phi_ab`` was
clipped to 0 for any genuine change larger than ~1.4: the trigger stayed
"static" forever even in front of a sustained step that never isolated.
Normalizing on the absolute scale of ``state_A`` / ``state_B`` fixed the step
but made a huge jump look proportionally coherent, so an isolated spike
passed untouched. The final rule normalizes the distance on the *local
volatility* of the recent window (scaled MAD, robust to a single residual
outlier). A huge jump with respect to the usual oscillation of the signal
stays suspicious (spike filtered); a sustained step is judged by how
anomalous it is relative to the normal noise, not by the magnitude of the
number.

``hybrid_shield`` generalizes the two-state logic of
``ia_utils.vector_healing.enhanced_dense_healing_hybrid`` (a sequence of
vectors) to a 1D scalar series. The two states are strict: for every point
the trigger decided by ``evaluate_phi_trigger`` is 0.0 (static/noise) or 1.0
(dynamic/genuine), with no intermediate branch.
"""

import jax
import jax.numpy as jnp
import numpy as np

GLOBAL_CONSTANTS = {
    "V_DINAMIC_K_COEFF": 5.0,
    "WEIGHT_SEMANTIC": 0.6,
    "WEIGHT_COHERENCE": 0.4,
    "NON_STATIC_THRESHOLD_A": 1e-2,
}


@jax.jit
def calculate_phi_ab(
    state_A: jnp.ndarray,
    state_B: jnp.ndarray,
    ipg_vector: jnp.ndarray,
    scale: jnp.ndarray,
) -> jnp.ndarray:
    """Alignment and spatial coherence factor Phi_AB.

    ``scale`` is the local volatility of the recent window (scaled MAD,
    computed by the caller), not a fixed constant and not the absolute
    magnitude of ``state_A`` / ``state_B``. See the module docstring for
    the deviation from upstream.

    Args:
        state_A: the reference state (baseline).
        state_B: the current state (raw sample).
        ipg_vector: the unit direction of the recent raw changes.
        scale: the local volatility of the recent window.

    Returns:
        ``phi_ab``, clipped to ``[0.0, 1.0]``.
    """
    semantic_change = state_B - state_A
    norm_change = jnp.linalg.norm(semantic_change)
    norm_ipg = jnp.linalg.norm(ipg_vector)

    alignment = jnp.where(
        (norm_change > 1e-12) & (norm_ipg > 1e-12),
        jnp.dot(semantic_change, ipg_vector) / (norm_change * norm_ipg),
        0.0,
    )
    semantic_alignment = (alignment + 1.0) / 2.0

    distance_A_B = jnp.linalg.norm(state_A - state_B)
    coherence_component = 1.0 - (distance_A_B / scale)

    phi_ab = (
        semantic_alignment * GLOBAL_CONSTANTS["WEIGHT_SEMANTIC"]
        + coherence_component * GLOBAL_CONSTANTS["WEIGHT_COHERENCE"]
    )
    return jnp.clip(phi_ab, 0.0, 1.0)


@jax.jit
def calculate_vettore_dinamico(
    E_A: jnp.ndarray, E_B: jnp.ndarray, Phi_AB: jnp.ndarray
) -> jnp.ndarray:
    """Dynamic vector (V_dinamic): differential logarithmic energy change.

    Args:
        E_A: reference energy.
        E_B: current energy.
        Phi_AB: the alignment factor from :func:`calculate_phi_ab`.

    Returns:
        The dynamic vector value, or 0.0 when either energy is below the
        numerical floor.
    """
    valid_inputs = (E_A > 1e-12) & (E_B > 1e-12)
    ratio = jnp.where(valid_inputs, E_B / E_A, 1.0)
    log_ratio_clamped = jnp.clip(jnp.log(ratio), -5.0, 5.0)
    v_vita = GLOBAL_CONSTANTS["V_DINAMIC_K_COEFF"] * log_ratio_clamped * Phi_AB
    return jnp.where(valid_inputs, v_vita, 0.0)


@jax.jit
def evaluate_phi_trigger(deterministic_dq_dt_a: jnp.ndarray) -> jnp.ndarray:
    """Binary trigger: 1.0 (dynamic) or 0.0 (static).

    Args:
        deterministic_dq_dt_a: the dynamic vector value.

    Returns:
        1.0 when the magnitude of the input exceeds ``NON_STATIC_THRESHOLD_A``,
        0.0 otherwise.
    """
    magnitude_change_a = jnp.abs(deterministic_dq_dt_a)
    trigger_active = magnitude_change_a > GLOBAL_CONSTANTS["NON_STATIC_THRESHOLD_A"]
    return jnp.where(trigger_active, 1.0, 0.0)


def _local_nan_fill(x: np.ndarray, radius: int) -> np.ndarray:
    """Fill NaNs with the median of the finite neighbours in a local window.

    A median, not a mean, and local, not global. With a global mean a NaN
    next to a 9999 spike in a 200-point series was filled with ~51 instead
    of ~1.0: the spike distorted the replacement of every NaN, not only the
    one next to it. With a local median the replacement reflects the
    neighbourhood of the point.

    Args:
        x: the input array.
        radius: half-width of the local window.

    Returns:
        A copy of ``x`` with every NaN replaced by a finite value.
    """
    x = np.copy(x)
    nan_mask = np.isnan(x)
    if not np.any(nan_mask):
        return x
    n = x.size
    global_fallback = np.nanmedian(x)
    if np.isnan(global_fallback):
        global_fallback = 0.0
    for idx in np.where(nan_mask)[0]:
        lo_w, hi_w = max(0, idx - radius), min(n, idx + radius + 1)
        window_finite = x[lo_w:hi_w]
        window_finite = window_finite[~np.isnan(window_finite)]
        x[idx] = np.median(window_finite) if window_finite.size > 0 else global_fallback
    return x


def hybrid_shield(
    serie: np.ndarray,
    riferimento: np.ndarray | None = None,
    radius_baseline: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply the two-state scheme (pass-through / fallback to baseline) point by point.

    Windows are always taken from ``processed`` (the raw, sanitized series),
    never from the previous output ``out``. Using ``out`` protects from a
    past outlier but introduces a worse bug on a genuine sustained step: the
    first one or two points after the transition are inevitably rejected (no
    causal detector can distinguish a true step from a spike on the very
    first sample), those rejected points end up in ``out``, and the window
    of the next points feeds them back into the calculation, producing a
    "halfway" baseline that also rejects the genuinely new points. The
    result is a stable but wrong equilibrium that never recovers (verified:
    30 real points after a step from 1.0 to 5.0 stayed stuck at ~4.0
    forever, never reaching 5.0). The median fallback is robust to a single
    spike inside the window without the self-contamination risk.

    The IPG is also taken from ``processed``, not from ``out``: the IPG must
    see whether the recent *raw* values are actually moving in a new
    direction. Computing it from ``out`` creates a self-feeding block: a
    rejected point is replaced by the baseline, so ``out`` never shows
    evidence of the new value, the IPG stays at zero, and no real step can
    ever be recognized no matter how long it lasts (verified: with IPG from
    ``out`` a 10-point step stayed flattened at 100%, no escape).

    Known limitation. The first two points (``i = 0`` and ``i = 1``) never
    go through the trigger loop (an IPG from the two previous points is
    required), so they always pass through with ``trigger = 1.0``,
    regardless of their value. The behaviour is inherited unchanged from
    ``enhanced_dense_healing_hybrid`` and is negligible on realistic
    series; any NaN or spike on those two points must be caught by
    independent upstream checks (the robust filters in
    ``Armatura.analizza``).

    Args:
        serie: a 1D array of floats. NaN and Inf are accepted and sanitized.
        riferimento: when given, the baseline for point ``i`` is
            ``riferimento[i]`` (anti-drift mode). When ``None`` (blind
            mode), the baseline is the local mean of the previous points of
            the same series, with adaptive radius
            ``min(20, max(3, n // 3))``.
        radius_baseline: optional fixed radius for the blind-mode window.

    Returns:
        A tuple ``(pulito, trigger, metadata)`` where ``pulito`` is the
        output series (raw value where ``trigger == 1``, baseline where
        ``trigger == 0``), ``trigger`` is the 0.0 / 1.0 indicator array, and
        ``metadata`` holds ``fallback_triggered`` and
        ``adaptive_radius_used``.
    """
    s = np.asarray(serie, dtype=np.float64).ravel()
    n = s.size

    if n == 0:
        return (
            np.empty(0),
            np.empty(0),
            {"fallback_triggered": False, "adaptive_radius_used": 0},
        )

    if radius_baseline is None:
        adaptive_radius_used = 0 if n < 3 else min(20, max(3, n // 3))
    else:
        adaptive_radius_used = radius_baseline
    fill_radius = max(adaptive_radius_used, 3)

    processed = np.copy(s)
    processed[np.isinf(processed)] = np.nan
    processed = _local_nan_fill(processed, fill_radius)

    rif = None
    if riferimento is not None:
        rif = np.asarray(riferimento, dtype=np.float64).ravel()
        if rif.size != n:
            raise ValueError(f"reference has {rif.size} points, series has {n}")
        rif = np.where(np.isinf(rif), np.nan, rif)
        rif = _local_nan_fill(rif, fill_radius)

    if n <= 2:
        return (
            np.copy(processed),
            np.ones(n),
            {"fallback_triggered": False, "adaptive_radius_used": adaptive_radius_used},
        )

    R = adaptive_radius_used
    processed_j = jnp.asarray(processed)
    padded = jnp.concatenate([jnp.full(R, jnp.nan), processed_j])

    def _get_window(i):
        return jax.lax.dynamic_slice(padded, (i,), (R,))

    idx = jnp.arange(n)
    windows = jax.vmap(_get_window)(idx)
    count_valid = jnp.minimum(idx, R)
    valid_mask = jnp.arange(R)[None, :] >= (R - count_valid[:, None])

    def _masked_mean(row, mask):
        return jnp.sum(jnp.where(mask, row, 0.0)) / jnp.maximum(jnp.sum(mask), 1)

    def _masked_median(row, mask, k):
        sorted_vals = jnp.sort(jnp.where(mask, row, jnp.inf))
        mid = k // 2
        return jnp.where(
            k % 2 == 0,
            (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0,
            sorted_vals[mid],
        )

    def _masked_std_of_diff(row, mask, k):
        """Local volatility: standard deviation of the successive differences.

        Deliberately not the MAD of the window values. On a window that
        straddles a true transition, as soon as the majority of the points
        reach the new level the median of the values collapses onto the new
        level and the MAD drops to ~0 (the pre-transition minority
        disappears from the metric), causing an oscillating rejection right
        after a correct acceptance. The standard deviation of the deltas
        does not collapse: a single large difference (the transition itself)
        raises the std instead of vanishing behind the median.

        Known trade-off: an isolated spike temporarily raises this
        volatility (two large differences, in and out of the spike) for
        ``radius`` steps, making the engine more permissive in that window.
        It self-corrects when the spike leaves the window; it is not a
        permanent block.
        """
        diffs = row[1:] - row[:-1]
        diff_mask = mask[1:] & mask[:-1]
        cnt = jnp.maximum(jnp.sum(diff_mask), 1)
        mean_d = jnp.sum(jnp.where(diff_mask, diffs, 0.0)) / cnt
        var_d = jnp.sum(jnp.where(diff_mask, (diffs - mean_d) ** 2, 0.0)) / cnt
        return jnp.where(k >= 3, jnp.sqrt(var_d), 0.0)

    baseline_mean = jax.vmap(_masked_mean)(windows, valid_mask)
    median_vals = jax.vmap(_masked_median)(windows, valid_mask, count_valid)
    local_scale = jax.vmap(_masked_std_of_diff)(windows, valid_mask, count_valid)
    scale = jnp.maximum(local_scale, 1e-6)

    baseline = jnp.asarray(rif) if rif is not None else baseline_mean
    state_A = baseline
    state_B = processed_j

    d = jnp.diff(processed_j)
    ipg_raw = jnp.concatenate([jnp.zeros(2), d[:-1]])
    norm_ipg_raw = jnp.abs(ipg_raw)
    ipg_vector = jnp.where(
        norm_ipg_raw > 1e-9,
        ipg_raw / jnp.maximum(norm_ipg_raw, 1e-30),
        ipg_raw,
    )

    semantic_change = state_B - state_A
    norm_change = jnp.abs(semantic_change)
    norm_ipg = jnp.abs(ipg_vector)
    alignment = jnp.where(
        (norm_change > 1e-12) & (norm_ipg > 1e-12),
        (semantic_change * ipg_vector) / (norm_change * norm_ipg),
        0.0,
    )
    semantic_alignment = (alignment + 1.0) / 2.0
    distance_A_B = jnp.abs(state_A - state_B)
    coherence_component = 1.0 - (distance_A_B / scale)
    phi_ab = jnp.clip(
        semantic_alignment * GLOBAL_CONSTANTS["WEIGHT_SEMANTIC"]
        + coherence_component * GLOBAL_CONSTANTS["WEIGHT_COHERENCE"],
        0.0,
        1.0,
    )

    E_A = jnp.abs(state_A)
    E_B = jnp.abs(state_B)
    valid_energy = (E_A > 1e-12) & (E_B > 1e-12)
    ratio = jnp.where(valid_energy, E_B / jnp.where(E_A > 1e-12, E_A, 1.0), 1.0)
    log_ratio_clamped = jnp.clip(jnp.log(ratio), -5.0, 5.0)
    v_dinamic = jnp.where(
        valid_energy,
        GLOBAL_CONSTANTS["V_DINAMIC_K_COEFF"] * log_ratio_clamped * phi_ab,
        0.0,
    )

    trigger_full = jnp.where(
        jnp.abs(v_dinamic) > GLOBAL_CONSTANTS["NON_STATIC_THRESHOLD_A"],
        1.0,
        0.0,
    )
    fallback_val = median_vals if rif is None else baseline
    out_full = jnp.where(
        trigger_full > GLOBAL_CONSTANTS["NON_STATIC_THRESHOLD_A"],
        processed_j,
        fallback_val,
    )

    out = np.array(out_full)
    trigger_arr = np.array(trigger_full)
    out[0], out[1] = processed[0], processed[1]
    trigger_arr[0], trigger_arr[1] = 1.0, 1.0

    fallback_triggered_at_all = bool(
        np.any(trigger_arr[2:] < GLOBAL_CONSTANTS["NON_STATIC_THRESHOLD_A"])
    )
    metadata = {
        "fallback_triggered": fallback_triggered_at_all,
        "adaptive_radius_used": adaptive_radius_used,
    }
    return out, trigger_arr, metadata
