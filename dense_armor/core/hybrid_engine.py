"""Two-state filter for a 1D series: keep the raw value or fall back to a baseline.

``hybrid_shield`` decides for every point whether the raw value is a genuine
change (trigger 1.0, the raw value is kept) or noise (trigger 0.0, the value
is replaced by a baseline). The baseline is the local median of the previous
points, or a reference series when one is given.

The decision for a point uses two things: whether the change from the
baseline points in the direction of the recent raw changes, and how large the
change is in units of the local volatility (the standard deviation of the
successive differences in the recent window). They are combined into a score
``phi_ab`` in [0, 1] with the weights 0.6 and 0.4. The log of the ratio
``|raw| / |baseline|``, clipped to [-5, 5], times 5 and times ``phi_ab``, is
``v_dinamic``; the trigger is 1.0 when ``|v_dinamic|`` is above 0.01.

``calculate_phi_ab``, ``calculate_vettore_dinamico`` and
``evaluate_phi_trigger`` are the vector versions of these three steps,
copied from ``dense_evolution/healing.py``. ``hybrid_shield`` has its own
scalar version of the same steps and does not call them.
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
    """Fill every NaN with the median of the finite values around it.

    The window has ``radius`` points on each side; when it has no finite value
    the median of the whole array is used. A median, so that a spike next to
    a NaN does not bias the fill.

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
    """Filter a 1D series: the raw value where the trigger is 1.0, the baseline where it is 0.0.

    Notes:
        The windows and the direction of the recent changes come from the
        raw sanitized series, not from the output. Taking them from the
        output feeds the baseline back into the windows, and a sustained
        step can then stay rejected.

        The first two points always pass with trigger 1.0, because the
        direction needs two previous points. A NaN or a spike there must be
        caught before this function.

        The local volatility is the standard deviation of the successive
        differences of the window. A median-based scale would collapse to
        zero in a window that crosses a transition. An isolated spike raises
        the volatility for ``radius`` steps and makes the filter more
        permissive in that window.

    Args:
        serie: a 1D array of floats. NaN and Inf are accepted and sanitized.
        riferimento: when given, the baseline for point ``i`` is
            ``riferimento[i]``. When ``None``, the baseline is the local mean
            of the previous points of the same series, with radius
            ``min(20, max(3, n // 3))``.
        radius_baseline: optional fixed radius of the window of the blind
            mode.

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
        """Local volatility: standard deviation of the successive differences of the window."""
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
