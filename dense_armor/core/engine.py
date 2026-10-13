"""Adaptive signal stabilizer for time series and AI gradients.

Features:
    - Anomaly detection through a dynamic threshold based on rolling volatility.
    - Selective damping with L2-norm preservation to avoid local blow-ups.
    - Macro-context calibration on 2D batches ``(N_scenarios, T_steps)`` with
      dynamic parameters.
    - Fully JAX-compatible, with precompiled ``vmap`` + ``scan`` kernels.

The numeric constants below are fixed hyper-parameters of the damping curve.
They have no physical or mathematical meaning beyond shaping the gain.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

_PHI_128 = np.longdouble(1.0 + np.sqrt(5.0)) / 2.0
_ZETA_TOTAL_128 = np.longdouble(60.16762236065194)

_PHI = float(_PHI_128)
_ALPHA = 0.25
_SIGMA = float(np.longdouble(1.0) / (_PHI_128**4))
_K_MAX_ZETA = float(_ZETA_TOTAL_128 / (_PHI_128**4))


@jax.jit
def dynamic_damping_gain(
    local_noise: jnp.ndarray,
    k_min: float = 0.0,
    k_max: float = _K_MAX_ZETA,
) -> jnp.ndarray:
    """Sigmoid damping curve as a function of the local noise.

    A low ``local_noise`` gives a gain close to ``k_min`` (maximum
    transparency); a high ``local_noise`` gives a gain close to ``k_max``
    (maximum purification).

    Args:
        local_noise: the local noise estimate.
        k_min: minimum gain (maximum transparency).
        k_max: maximum gain (maximum purification).

    Returns:
        The gain value in ``[k_min, k_max]``.
    """
    x = jnp.clip(local_noise, 0.0, 5.0)
    steering = 1.0 / (1.0 + jnp.exp(-(x - 2.0)))
    return k_min + steering * (k_max - k_min)


class AdaptiveSignalStabilizer:
    """Adaptive anti-panic filter for chaotic AI signals.

    Args:
        static_threshold: base detection threshold.
        initial_damping: initial damping coefficient.
        alpha: EMA coefficient for the rolling volatility.
        anomaly_sigma_mult: multiplier on the rolling volatility used to
            build the dynamic threshold.
        k_anom_min: minimum gain on the anomalous branch.
        k_anom_max: maximum gain on the anomalous branch.
        window_radius: radius of the local mean window.
        smooth_l2_blend: blend between the raw smoothed value and its
            L2-rescaled version, in ``[0.0, 1.0]``.
    """

    def __init__(
        self,
        static_threshold: float = 1e-5,
        initial_damping: float = 0.1,
        alpha: float = _SIGMA,
        anomaly_sigma_mult: float = 2.0,
        k_anom_min: float = _ALPHA,
        k_anom_max: float = _K_MAX_ZETA,
        window_radius: int = 1,
        smooth_l2_blend: float = 0.5,
    ) -> None:
        """Configure the thresholds and precompile the JAX kernels."""
        self.threshold: float = float(static_threshold)
        self.damping: float = float(initial_damping)
        self.alpha: float = float(alpha)

        self.anomaly_sigma_mult: float = float(anomaly_sigma_mult)
        self.k_anom_min: float = float(k_anom_min)
        self.k_anom_max: float = float(k_anom_max)
        self.window_radius: int = int(max(window_radius, 0))
        self.smooth_l2_blend: float = float(np.clip(smooth_l2_blend, 0.0, 1.0))

        self.dyn_thr: float = self.threshold
        self.dyn_dmp: float = self.damping
        self.dyn_alp: float = self.alpha
        self.noise_scalar: float = 1.0

        self._compiled_stream_filter = self._build_stream_filter()
        self._compiled_batch_filter = jax.jit(
            jax.vmap(
                self._process_single_scenario,
                in_axes=(0, None, None, None, None, 0),
            )
        )
        self._compiled_batch_filter_independent = jax.jit(
            jax.vmap(
                self._process_single_scenario,
                in_axes=(0, 0, 0, 0, 0, 0),
            )
        )
        self._compiled_single_stream_filter = jax.jit(self._run_single_stream_scan)

    def _run_single_stream_scan(
        self,
        init_state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        rest: jnp.ndarray,
        thr: jnp.ndarray,
        dmp: jnp.ndarray,
        alp: jnp.ndarray,
        noise_scalar: jnp.ndarray,
        init_val: jnp.ndarray,
    ) -> tuple:
        """1D scan body used by :meth:`filter_data_stream`."""
        n = rest.shape[0]
        thrs = jnp.full((n,), thr)
        dmps = jnp.full((n,), dmp)
        alps = jnp.full((n,), alp)
        n_scalars = jnp.full((n,), noise_scalar)
        left_neighbors = jnp.full((n,), init_val)
        upper_neighbors = jnp.full((n,), init_val)
        is_1d_array = jnp.full((n,), True)
        hard_clamp_flags = jnp.full((n,), False)

        def _wrapped_step(carry: tuple, step_inputs: tuple) -> tuple:
            """Adapt :meth:`_step_kernel` to the ``jax.lax.scan`` signature."""
            return self._step_kernel(carry, step_inputs)

        scan_inputs = (
            rest,
            thrs,
            dmps,
            alps,
            n_scalars,
            left_neighbors,
            upper_neighbors,
            is_1d_array,
            hard_clamp_flags,
        )
        return jax.lax.scan(_wrapped_step, init_state, scan_inputs)

    def calibrate_macro_context(self, raw_batch: np.ndarray) -> None:
        """Calibrate the dynamic thresholds on the global context of the batch.

        Analyses the distribution of jumps between samples to tell an
        extremely turbulent environment (a shock) from physiological noise.
        The parameters are stored on the instance (``dyn_thr``, ``dyn_dmp``,
        ``dyn_alp``, ``noise_scalar``); the method is not pure by design and
        leaves existing callers unchanged.

        Args:
            raw_batch: the batch of raw signals.
        """
        if raw_batch.size == 0:
            return

        sample_diff = np.abs(np.diff(raw_batch))
        global_std = float(np.std(raw_batch))

        self.noise_scalar = float(1.0 / (1.0 + global_std))

        if sample_diff.size == 0:
            return

        max_jump = float(np.max(sample_diff))
        mean_jump = float(np.mean(sample_diff))

        if max_jump > 3.0 and (max_jump / (mean_jump + 1e-5)) > 5.0:
            self.dyn_thr = 1e-5
            self.dyn_dmp = 1e-3
            self.dyn_alp = 0.999
            return

        self.dyn_thr = float(0.10 * global_std)
        self.dyn_dmp = float(_SIGMA * (1.0 + self.noise_scalar))
        self.dyn_alp = float(0.75 + (_SIGMA * self.noise_scalar))

    def calibrate_macro_context_independent(
        self, raw_batch: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Same formulas as :meth:`calibrate_macro_context`, computed per row.

        The parameters are computed along ``axis=-1`` instead of on the whole
        flattened batch. The method is pure and does not mutate ``self``, so
        it is safe to use inside ``vmap`` downstream. The existing
        batch-shared :meth:`calibrate_macro_context` is kept unchanged for
        its callers (``test_boundA.py`` and the adversarial suite use the
        shared calibration on purpose: correlated scenarios share one
        context).

        Args:
            raw_batch: shape ``(n_scenarios, F)``. When ``F < 2`` no row has
                a "jump" to measure and the method returns the ``__init__``
                defaults (``self.threshold``/``damping``/``alpha``) for
                every row, instead of trying to replicate the stateful
                early-return of :meth:`calibrate_macro_context`.

        Returns:
            A tuple ``(dyn_thr, dyn_dmp, dyn_alp, noise_scalar)``, each of
            shape ``(n_scenarios,)``.
        """
        n_scenarios = int(raw_batch.shape[0])
        per_row_std = np.std(raw_batch, axis=-1)
        noise_scalar = 1.0 / (1.0 + per_row_std)
        F = raw_batch.shape[-1] if raw_batch.ndim > 1 else 0
        if F < 2:
            return (
                np.full(n_scenarios, self.threshold),
                np.full(n_scenarios, self.damping),
                np.full(n_scenarios, self.alpha),
                noise_scalar,
            )
        sample_diff = np.abs(np.diff(raw_batch, axis=-1))
        max_jump = np.max(sample_diff, axis=-1)
        mean_jump = np.mean(sample_diff, axis=-1)
        panic = (max_jump > 3.0) & ((max_jump / (mean_jump + 1e-5)) > 5.0)
        dyn_thr = np.where(panic, 1e-5, 0.10 * per_row_std)
        dyn_dmp = np.where(panic, 1e-3, _SIGMA * (1.0 + noise_scalar))
        dyn_alp = np.where(panic, 0.999, 0.75 + (_SIGMA * noise_scalar))
        return dyn_thr, dyn_dmp, dyn_alp, noise_scalar

    def filter_batch_scenarios_independent(
        self, raw_batch: np.ndarray, hard_clamp_mask: np.ndarray | None = None
    ) -> np.ndarray:
        """Like :meth:`filter_batch_scenarios` with per-scenario calibration.

        Every scenario receives its own calibration from
        :meth:`calibrate_macro_context_independent` instead of one shared on
        the whole batch. Numerically equivalent to calling
        :meth:`filter_batch_scenarios` once per scenario
        (``n_scenarios=1`` each), but in a single ``vmap`` call instead of N
        separate JAX dispatches. Added for
        :meth:`Orca.protect_and_forward`: the library has never exercised
        ``n_scenarios > 1`` on independent scenarios before (the only
        existing multi-scenario caller, the adversarial suite in
        ``test_boundA.py``, wants the shared calibration on purpose).

        Args:
            raw_batch: the batch of raw signals, shape ``(N, ...)``.
            hard_clamp_mask: optional boolean mask, same shape as
                ``raw_batch``; ``True`` where the caller (Orca) has already
                flagged a macro anomaly. Activates the State Flush in
                :meth:`_step_kernel`.

        Returns:
            The filtered batch, same shape as ``raw_batch``.
        """
        if raw_batch.size == 0:
            return np.zeros_like(raw_batch)
        if hard_clamp_mask is None:
            hard_clamp_mask = np.zeros_like(raw_batch, dtype=bool)

        original_shape = raw_batch.shape
        n_scenarios = int(original_shape[0])

        if len(original_shape) == 2:
            h_dim = 1
            w_dim = int(original_shape[1])
            structured_batch = raw_batch.reshape(n_scenarios, h_dim, w_dim)
            structured_mask = np.asarray(hard_clamp_mask).reshape(
                n_scenarios, h_dim, w_dim
            )
        elif len(original_shape) == 3:
            h_dim, w_dim = int(original_shape[1]), int(original_shape[2])
            structured_batch = raw_batch
            structured_mask = np.asarray(hard_clamp_mask)
        elif len(original_shape) == 4:
            c_channels, h_dim, w_dim = (
                int(original_shape[1]),
                int(original_shape[2]),
                int(original_shape[3]),
            )
            structured_batch = raw_batch.reshape(n_scenarios * c_channels, h_dim, w_dim)
            structured_mask = np.asarray(hard_clamp_mask).reshape(
                n_scenarios * c_channels, h_dim, w_dim
            )
        else:
            raise ValueError(
                f"tensor geometry not supported by the engine: {original_shape}"
            )

        flat_per_unit = structured_batch.reshape(structured_batch.shape[0], -1)
        dyn_thr, dyn_dmp, dyn_alp, noise_scalar = (
            self.calibrate_macro_context_independent(flat_per_unit)
        )

        j_batch = jnp.array(structured_batch, dtype=jnp.float64)
        j_mask = jnp.array(structured_mask, dtype=jnp.bool_)

        filtered_structured = self._compiled_batch_filter_independent(
            j_batch,
            jnp.array(dyn_thr, dtype=jnp.float64),
            jnp.array(dyn_dmp, dtype=jnp.float64),
            jnp.array(dyn_alp, dtype=jnp.float64),
            jnp.array(noise_scalar, dtype=jnp.float64),
            j_mask,
        )

        filtered_np = np.array(filtered_structured, dtype=jnp.float64)
        return filtered_np.reshape(original_shape)

    def _step_kernel(
        self,
        carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        inputs: tuple[
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
        ],
    ):
        """One step of the causal scan: update state/gain/volatility and filter.

        The ninth input element ``hard_clamp_flag`` is the same authoritative
        signal already used by the Stage 2 Hard-Clamp in Orca
        (``raw_noise > 0.05`` on the original raw data, not a reconstruction
        here). Where ``True``, the minimum gain floor ``k_anom_min`` is
        instantaneously zeroed for this step only. Otherwise even a huge
        shock would leak at least ``k_anom_min`` (25% by default) into the
        recursive state (``prev_filtered`` / ``rolling_volatility``), and
        that residual contaminates the next steps with a slow decay
        (verified: without this, 3 normal samples after a 9999 spike stayed
        above 1000 instead of returning close to the true value). The normal
        smooth design (full ``k_anom_min``) is left unchanged below the
        threshold: this only acts when the Hard-Clamp has already fired.

        Args:
            carry: ``(prev_filtered, current_damping, rolling_volatility, local_mean)``.
            inputs: nine arrays; the first six are the current value and its
                context, the seventh and eighth the spatial neighbours, the
                ninth the hard-clamp flag.

        Returns:
            The updated carry and the filtered value for this step.
        """
        prev_filtered, current_damping, rolling_volatility, local_mean = carry
        (
            current_val,
            thr,
            _dmp,
            _alp,
            _n_scalar,
            left_neighbor,
            upper_neighbor,
            is_1d,
            hard_clamp_flag,
        ) = inputs

        eps = 1e-7

        is_nan_input = jnp.isnan(current_val)

        current_val_safe = jnp.where(is_nan_input, prev_filtered, current_val)
        left_neighbor_safe = jnp.where(
            jnp.isnan(left_neighbor), prev_filtered, left_neighbor
        )
        upper_neighbor_safe = jnp.where(
            jnp.isnan(upper_neighbor), prev_filtered, upper_neighbor
        )

        diff_decay = jnp.abs(current_val_safe - prev_filtered)
        diff_spatial_x = jnp.abs(current_val_safe - left_neighbor_safe)
        diff_spatial_y = jnp.abs(current_val_safe - upper_neighbor_safe)

        diff = jnp.where(
            is_1d,
            diff_decay,
            0.5 * diff_decay + 0.25 * diff_spatial_x + 0.25 * diff_spatial_y,
        )

        local_ref = local_mean
        local_diff = jnp.abs(current_val_safe - local_ref)
        local_scale = jnp.maximum(jnp.abs(local_ref), 1e-3)
        local_coherence = jnp.clip(1.0 - (local_diff / (2.0 * local_scale)), 0.0, 1.0)

        alpha_j = jnp.float64(self.alpha)
        new_vol = (1.0 - alpha_j) * rolling_volatility + alpha_j * diff

        sigma_mult = jnp.float64(self.anomaly_sigma_mult)
        dyn_thr = thr + (new_vol * sigma_mult)

        is_anomaly = jnp.logical_or(diff > dyn_thr, is_nan_input)

        coherence_t = 1.0 - (diff_decay / 2.0)
        coherence_x = 1.0 - (diff_spatial_x / 2.0)
        coherence_y = 1.0 - (diff_spatial_y / 2.0)

        coherence_2d = jnp.clip(
            0.4 * coherence_t + 0.3 * coherence_x + 0.3 * coherence_y, 0.0, 1.0
        )
        coherence_1d = jnp.clip(coherence_t, 0.0, 1.0)
        coherence_signal = jnp.where(is_1d, coherence_1d, coherence_2d)

        coherence_mix = 0.5 * coherence_signal + 0.5 * local_coherence

        c_gate = jnp.float64(_ALPHA + _SIGMA)
        K_coherent = jnp.clip(
            coherence_mix * (c_gate / (c_gate + new_vol + eps)),
            jnp.float64(_ALPHA - _SIGMA),
            jnp.float64(1.0 - (_ALPHA - _SIGMA)),
        )

        k_anom_min = jnp.where(
            hard_clamp_flag, jnp.float64(0.0), jnp.float64(self.k_anom_min)
        )
        k_anom_max = jnp.float64(self.k_anom_max)

        c_anom_base = jnp.float64(1.0 / (float(_PHI_128) ** 2))
        local_scale_ref = jnp.maximum(jnp.abs(prev_filtered), jnp.float64(1e-8))
        c_anom = c_anom_base * local_scale_ref
        K_anomalous_raw = c_anom / (c_anom + diff + eps)
        K_anomalous = jnp.clip(K_anomalous_raw, k_anom_min, k_anom_max)

        anomaly_strength = jnp.clip((diff - dyn_thr) / (dyn_thr + eps), 0.0, 1.0)

        anomaly_strength_final = jnp.where(is_nan_input, 1.0, anomaly_strength)
        K = (
            1.0 - anomaly_strength_final
        ) * K_coherent + anomaly_strength_final * K_anomalous

        raw_next = (1.0 - K) * prev_filtered + K * current_val_safe

        target_energy = jnp.sqrt(prev_filtered**2 + eps)
        current_energy = jnp.sqrt(raw_next**2 + eps)
        base_scale = target_energy / (current_energy + eps)
        soft_scale = 0.7 * base_scale + 0.3
        l2_scale = jnp.where(is_anomaly, soft_scale, 1.0)

        smooth_weight_local = local_coherence
        l2_weight_local = 1.0 - local_coherence
        raw_smooth = raw_next
        raw_l2 = raw_next * l2_scale
        local_blend = smooth_weight_local * raw_smooth + l2_weight_local * raw_l2

        g = jnp.float64(self.smooth_l2_blend)
        next_filtered = g * raw_smooth + (1.0 - g) * local_blend

        calculated_damping = dynamic_damping_gain(diff)
        next_damping = jnp.where(is_anomaly, calculated_damping, current_damping)

        w_rad = max(self.window_radius, 1)
        ema_alpha = 1.0 / float(w_rad + 1)
        ema_alpha_j = jnp.float64(ema_alpha)
        next_local_mean = (1.0 - ema_alpha_j) * local_mean + ema_alpha_j * next_filtered

        return (
            next_filtered,
            next_damping,
            new_vol,
            next_local_mean,
        ), next_filtered

    def _process_single_scenario(
        self,
        scenario_matrix: jnp.ndarray,
        thr: jnp.ndarray,
        dmp: jnp.ndarray,
        alp: jnp.ndarray,
        n_scalar: jnp.ndarray,
        hard_clamp_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Filter one 2D scenario ``(H, W)`` with the causal 1D scan.

        ``hard_clamp_mask`` has the same shape as ``scenario_matrix`` and is
        ``True`` where the caller (Orca) has already flagged a macro anomaly
        on the original raw data (see :meth:`_step_kernel` for the State
        Flush behaviour).

        Args:
            scenario_matrix: the 2D scenario.
            thr: base threshold.
            dmp: initial damping.
            alp: EMA coefficient for the volatility.
            n_scalar: noise scalar from the macro-context calibration.
            hard_clamp_mask: the boolean mask of macro anomalies.

        Returns:
            The filtered scenario, same shape as ``scenario_matrix``.
        """
        h_dim, w_dim = scenario_matrix.shape
        is_1d_flag = h_dim == 1

        flat_scen = scenario_matrix.flatten()

        left_neighbors = jnp.roll(scenario_matrix, shift=1, axis=1)
        left_neighbors = left_neighbors.at[:, 0].set(scenario_matrix[:, 0]).flatten()

        upper_neighbors = jnp.roll(scenario_matrix, shift=1, axis=0)
        upper_neighbors = upper_neighbors.at[0, :].set(scenario_matrix[0, :]).flatten()

        start_val = jnp.float64(flat_scen[0])
        init_state = (
            start_val,
            jnp.float64(dmp),
            jnp.float64(0.0),
            start_val,
        )

        thrs = jnp.full_like(flat_scen, thr)
        dmps = jnp.full_like(flat_scen, dmp)
        alps = jnp.full_like(flat_scen, alp)
        n_scalars = jnp.full_like(flat_scen, n_scalar)
        is_1d_array = jnp.full_like(flat_scen, is_1d_flag, dtype=jnp.bool_)
        hard_clamp_flags = hard_clamp_mask.flatten()

        scan_inputs = (
            flat_scen,
            thrs,
            dmps,
            alps,
            n_scalars,
            left_neighbors,
            upper_neighbors,
            is_1d_array,
            hard_clamp_flags,
        )

        def _wrapped_step(carry: tuple, step_inputs: tuple) -> tuple:
            """Adapt :meth:`_step_kernel` to the ``jax.lax.scan`` signature."""
            return self._step_kernel(carry, step_inputs)

        scan_out: tuple = jax.lax.scan(
            _wrapped_step,
            init_state,
            scan_inputs,
        )
        filtered_flat: jnp.ndarray = scan_out[1]

        return filtered_flat.reshape(h_dim, w_dim)

    def _build_stream_filter(self) -> Callable:
        """Build and compile the legacy single-series 1D filter."""

        def _legacy_step(
            carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
            current_val: jnp.ndarray,
        ) -> tuple:
            """One step of the legacy scan (historical path)."""
            prev, dmp_state, vol = carry

            diff = jnp.abs(current_val - prev)
            alpha_j = jnp.float64(self.alpha)
            new_vol = (1.0 - alpha_j) * vol + alpha_j * diff

            coherence = 1.0 - (diff / 2.0)
            coherence_legacy = jnp.clip(coherence, 0.0, 1.0)

            safe_prev = jnp.where(jnp.abs(prev) > 1e-12, prev, 1e-12)
            ratio = jnp.abs(current_val / safe_prev)
            log_ratio = jnp.clip(
                jnp.log(jnp.where(ratio > 0, ratio, 1.0)),
                -5.0,
                5.0,
            )

            c_lyap = jnp.float64((float(_PHI_128) ** 3) + 0.23606797)
            dynamic_shift_legacy = c_lyap * log_ratio * coherence_legacy

            is_anomaly_legacy = jnp.abs(dynamic_shift_legacy) > jnp.float64(
                self.threshold
            )

            c_gate = jnp.float64(_ALPHA + _SIGMA)
            K_coherent = jnp.clip(
                coherence_legacy * c_gate, jnp.float64(_SIGMA), c_gate
            )

            c_anom_scale = jnp.float64(_ALPHA / 2.0)
            K_anomalous = jnp.clip(
                c_anom_scale / (c_anom_scale + jnp.abs(dynamic_shift_legacy)),
                jnp.float64(_ALPHA),
                c_gate,
            )
            K = jnp.where(is_anomaly_legacy, K_anomalous, K_coherent)

            next_f = (1.0 - K) * prev + K * current_val

            calculated_damping = dynamic_damping_gain(diff)
            next_damping = jnp.where(is_anomaly_legacy, calculated_damping, dmp_state)

            return (next_f, next_damping, new_vol), next_f

        @jax.jit
        def _run(init_state: tuple, data: jnp.ndarray) -> tuple:
            """Run the compiled legacy scan over the whole series."""
            return jax.lax.scan(_legacy_step, init_state, data)

        return _run

    def filter_data_stream(self, raw_data: np.ndarray) -> np.ndarray:
        """Filter a single 1D time series.

        Args:
            raw_data: the input series.

        Returns:
            The filtered series, same length as ``raw_data``.
        """
        if raw_data.size == 0:
            return np.zeros_like(raw_data)

        j_raw = jnp.array(raw_data, dtype=jnp.float64)
        init_val = jnp.float64(j_raw[0])

        init_state = (
            init_val,
            jnp.float64(self.damping),
            jnp.float64(0.0),
            init_val,
        )

        _, gated_stream = self._compiled_single_stream_filter(
            init_state,
            j_raw[1:],
            jnp.float64(self.dyn_thr),
            jnp.float64(self.dyn_dmp),
            jnp.float64(self.dyn_alp),
            jnp.float64(self.noise_scalar),
            init_val,
        )

        gated_np = np.asarray(gated_stream, dtype=np.float64)
        final_stream = np.insert(gated_np, 0, np.asarray(init_val, dtype=np.float64))
        return final_stream

    def filter_batch_scenarios(
        self, raw_batch: np.ndarray, hard_clamp_mask: np.ndarray | None = None
    ) -> np.ndarray:
        """Filter a batch of structured scenarios in parallel (``vmap``).

        Accepts native 2D tensors or multidimensional arrays (3D/4D).

        Args:
            raw_batch: the batch of raw signals, shape ``(N, ...)``.
            hard_clamp_mask: optional boolean mask, same shape as
                ``raw_batch``. Where ``True``, activates the State Flush
                inside :meth:`_step_kernel` for that element. Meant to be
                the same signal already computed upstream by Orca
                (``raw_noise`` on the original raw data above the critical
                threshold), not an approximation computed here.

        Returns:
            The filtered batch, same shape as ``raw_batch``.
        """
        if raw_batch.size == 0:
            return np.zeros_like(raw_batch)
        if hard_clamp_mask is None:
            hard_clamp_mask = np.zeros_like(raw_batch, dtype=bool)

        original_shape = raw_batch.shape
        n_scenarios = int(original_shape[0])

        flat_analysis_batch = raw_batch.reshape(n_scenarios, -1)
        self.calibrate_macro_context(flat_analysis_batch)

        if len(original_shape) == 2:
            h_dim = 1
            w_dim = int(original_shape[1])
            structured_batch = raw_batch.reshape(n_scenarios, h_dim, w_dim)
            structured_mask = np.asarray(hard_clamp_mask).reshape(
                n_scenarios, h_dim, w_dim
            )
        elif len(original_shape) == 3:
            h_dim, w_dim = int(original_shape[1]), int(original_shape[2])
            structured_batch = raw_batch
            structured_mask = np.asarray(hard_clamp_mask)
        elif len(original_shape) == 4:
            c_channels, h_dim, w_dim = (
                int(original_shape[1]),
                int(original_shape[2]),
                int(original_shape[3]),
            )
            structured_batch = raw_batch.reshape(n_scenarios * c_channels, h_dim, w_dim)
            structured_mask = np.asarray(hard_clamp_mask).reshape(
                n_scenarios * c_channels, h_dim, w_dim
            )
        else:
            raise ValueError(
                f"tensor geometry not supported by the engine: {original_shape}"
            )

        j_batch = jnp.array(structured_batch, dtype=jnp.float64)
        j_mask = jnp.array(structured_mask, dtype=jnp.bool_)

        filtered_structured = self._compiled_batch_filter(
            j_batch,
            jnp.float64(self.dyn_thr),
            jnp.float64(self.dyn_dmp),
            jnp.float64(self.dyn_alp),
            jnp.float64(self.noise_scalar),
            j_mask,
        )

        filtered_np = np.array(filtered_structured, dtype=jnp.float64)

        return filtered_np.reshape(original_shape)
