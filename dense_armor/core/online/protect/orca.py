"""
core/online/protect/orca.py
Universal AI Orchestrator Shield (ORCA) -- dynamic, selective,
context-aware orchestrator in 4 phases.
"""

import logging
import time
from collections import Counter, deque
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from dense_armor._dtype import as_default_array, get_default_dtype
from dense_armor.core.online.anomaly.curvature import curvature
from dense_armor.core.online.anomaly.resonance_search import apply_fast_resonance
from dense_armor.core.online.protect.arbiter import route_and_correct
from dense_armor.utility.damping_operator import apply_damping_blend
from dense_armor.utility.engine import AdaptiveSignalStabilizer
from dense_armor.utility.memory import UniversalMemoryGuard
from dense_armor.utility.noise import AIHardwareProfiler

logger = logging.getLogger(__name__)


class Orca:
    # reference chunk_threshold (the old fixed default) for a "base"
    # host according to AIHardwareProfiler: RAM < 12GB, no GPU/TPU
    # (max_tensor_dim=2048, see utility/noise.py). The auto-derived chunk_threshold
    # scales PROPORTIONALLY to this baseline via max_tensor_dim, it does not
    # replace it with the raw RAM heuristic taken alone (which utility/noise.py
    # already declares arbitrary, not empirically calibrated, see CHANGELOG
    # 1.1.8) -- so a "base" host gets the same value as ever
    # (no silent regression on the common case), while a host with
    # more RAM or a GPU/TPU gets larger chunks in proportion.
    _CHUNK_THRESHOLD_BASELINE = 1_000_000
    _CHUNK_THRESHOLD_BASELINE_TENSOR_DIM = 2048
    # Maximum multiple of the target scale (10^val_e) beyond which co_chunk
    # is clipped, see _execute_4_phase_input_shield for the reason.
    _CO_CHUNK_CLIP_MULT = 20.0

    def __init__(
        self,
        static_threshold: float = 0.15,
        initial_damping: float = 0.85,
        alpha: float = 0.05,
        val_e: float = -4.0,
        chunk_threshold: int | None = None,
        min_free_ram_percentage: float = 0.15,
        reference_memory_size: int = 32,
        reference_recall_min_score: float = 0.90,
        hard_clamp_threshold: float = 0.05,
    ) -> None:
        """val_e — target scale exponent for the input log10 compression;
        hard_clamp_threshold — noise |corrupted - reference|, in the units of the
        RAW data (before the log10 compression), above which the emergency
        hard clamp is used. The default 0.05 suits data of order 1; for data of
        a very different scale set it accordingly;
        chunk_threshold — size beyond which a batch is processed in blocks;
        None (default) auto-derives it from AIHardwareProfiler (RAM/backend
        of the current host, see _CHUNK_THRESHOLD_BASELINE* above) instead of the
        old fixed value, equal for any machine;
        reference_memory_size — how many past clean references (per shape) to keep in
        memory for the resonance recall in blind mode (see _recall_reference);
        reference_recall_min_score — minimum apply_fast_resonance score below
        which a reference in memory is considered not similar enough and
        _blind_reference is used instead (empirically calibrated: on correlated vectors the
        score stays >0.95 even with moderate noise, on uncorrelated vectors it does not exceed
        ~0.75 — 0.90 leaves margin on both sides)."""
        self.static_threshold = static_threshold
        self.initial_damping = initial_damping
        self.alpha = alpha
        if chunk_threshold is None:
            profiler = AIHardwareProfiler()
            scale = profiler.max_tensor_dim / self._CHUNK_THRESHOLD_BASELINE_TENSOR_DIM
            chunk_threshold = int(self._CHUNK_THRESHOLD_BASELINE * scale)
            logger.info(
                "chunk_threshold auto-derivato da AIHardwareProfiler: %d (%s)",
                chunk_threshold,
                profiler.get_profile_summary(),
            )
        self.chunk_threshold = int(chunk_threshold)
        self.val_e = float(val_e)
        self.min_free_ram = min_free_ram_percentage
        self.reference_recall_min_score = float(reference_recall_min_score)
        self.hard_clamp_threshold = float(hard_clamp_threshold)
        self.stabilizer = AdaptiveSignalStabilizer(
            static_threshold, initial_damping, alpha
        )
        self.last_kappa = 0.0
        # Shared memory guard (RAM+VRAM), no longer an ad-hoc psutil check
        # duplicated in here (see _gc_se_ram_bassa).
        self._memory_guard = UniversalMemoryGuard(
            min_free_ram_percentage=min_free_ram_percentage
        )
        # Bank of clean references seen in the past, one deque per shape
        # (different shapes are not comparable with each other): when the caller
        # provides x_reference, its rows are remembered here; in blind
        # mode (x_reference=None) a future request with a corrupted input
        # similar to one already seen can reuse the true reference instead of
        # starting from scratch with the local estimate alone (see _recall_reference).
        self._reference_bank: dict = {}
        self._reference_memory_size = int(reference_memory_size)
        # JAX kernels precompiled only once (same principle as the fix
        # to the scan in AdaptiveSignalStabilizer.filter_data_stream): without
        # this, the eager pipeline of _execute_4_phase_*_shield re-dispatches
        # and recompiles every single JAX operation at every call.
        self._compiled_input_shield_kernel = jax.jit(self._run_input_shield_kernel)
        self._compiled_output_shield_kernel = jax.jit(self._run_output_shield_kernel)
        # error margin (like the Kalman covariance, but defined simply
        # as |received_value - corrected_value|: the more the shield had to
        # move a value, the less the result at that point should be trusted)
        self.margine_ingresso = (
            None  # array, same shape as the input: uncertainty in INPUT SPACE
        )
        self.margine_ingresso_medio = 0.0
        self.margine_ingresso_max = 0.0
        self.margine_uscita = (
            None  # array, same shape as the output: uncertainty in OUTPUT SPACE
        )
        self.margine_uscita_medio = 0.0
        self.margine_uscita_max = 0.0

        # Arbiter labels/uncertainty (only if use_arbiter=True in
        # protect_and_forward) -- margine_ingresso already says how much a value
        # was moved; incertezza_arbitro says how unsure the arbiter itself
        # is of having chosen the right classification (near the
        # deviating/clean or spike/regime boundary), not how large the
        # correction was. The two signals are complementary, not redundant:
        # a small correction with high uncertainty is an ambiguous case that
        # went well by chance, not one that is truly safe.
        self.etichette_arbitro = (
            None  # array of strings 'clean'/'spike'/'regime', same shape as the input
        )
        self.incertezza_arbitro = None  # array 0..1, same shape as the input
        self.incertezza_arbitro_media = 0.0
        # Memory of the corruption TYPES already seen for each shape (not just
        # the clean references per shape, already in _reference_bank) -- a
        # Counter of the arbiter labels observed when x_reference
        # was known, so a future call in blind mode on the same
        # shape can know whether that sensor/stream has already shown isolated
        # spikes, regime changes, or both, instead of starting from scratch.
        self._corruption_type_memory: dict = {}

    def _gc_se_ram_bassa(self) -> None:
        """Delegates to UniversalMemoryGuard.check_memory_safety() the RAM/VRAM
        check before the next heavy chunk (same soft-GC + hard
        jax.clear_caches() that this method did before, but shared with the
        rest of the library instead of an ad-hoc psutil check duplicated here).

        Difference in behavior compared with before: under the hard threshold
        (min_free_ram) the Guard raises MemoryPressureError instead of
        continuing silently -- fail-fast with a readable error now,
        rather than a process OOM a few chunks later."""
        self._memory_guard.check_memory_safety()

    def _recall_reference(
        self, row_corrupted_flat: np.ndarray, slice_shape: tuple
    ) -> np.ndarray | None:
        """Searches the bank of clean references (same shape) for the one most
        similar to the current corrupted input via apply_fast_resonance. Returns
        None (no recall) if the bank for this shape is empty or if the
        best score stays below reference_recall_min_score -- in
        both cases the caller falls back to _blind_reference, the
        original behavior unchanged."""
        bank = self._reference_bank.get(slice_shape)
        if not bank:
            return None
        candidates = np.stack(list(bank))
        scores = apply_fast_resonance(candidates, row_corrupted_flat)
        best_idx = int(np.argmax(scores))
        if float(scores[best_idx]) < self.reference_recall_min_score:
            return None
        return candidates[best_idx]

    def _remember_reference(
        self, x_reference_np: np.ndarray, slice_shape: tuple
    ) -> None:
        """Records each row of a clean reference provided by the caller
        in the bank (a fixed-size deque per shape, FIFO -- the oldest
        leave when new ones are added beyond reference_memory_size)."""
        bank = self._reference_bank.setdefault(
            slice_shape, deque(maxlen=self._reference_memory_size)
        )
        for b in range(x_reference_np.shape[0]):
            bank.append(as_default_array(x_reference_np[b].flatten()))

    def tipi_corruzione_visti(self, slice_shape: tuple) -> Counter:
        """Which arbiter labels ('clean'/'spike'/'regime') have already been
        observed for this shape, and how many times -- empty Counter if
        this shape has never been seen with use_arbiter=True and a known
        x_reference (the only way the memory is populated,
        see _aggiorna_memoria_corruzione)."""
        return self._corruption_type_memory.get(slice_shape, Counter())

    def _aggiorna_memoria_corruzione(
        self, etichette_riga: np.ndarray, slice_shape: tuple
    ) -> None:
        """Updates the Counter of corruption types for this shape with the
        labels observed in a row -- called only when x_reference
        is known (the only case in which the arbiter has a true target to
        classify against, not a blind estimate)."""
        counter = self._corruption_type_memory.setdefault(slice_shape, Counter())
        counter.update(etichette_riga.tolist())

    def _applica_arbitro(
        self,
        co_row_flat: np.ndarray,
        purified_row_flat: np.ndarray,
        radius: int = 10,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Routes each point to the right corrector instead of a single one
        for the whole row (see core/online/protect/arbiter.py and
        test/test_arbiter_orca_integration.py for the real comparison that
        motivated this):
          'spike'  -> corrected with the median of the arbiter's wide
                      reference window (rejects an isolated impulse even
                      when the standard 4-phase pipeline would let it
                      partly through).
          'regime' -> RAW value (a sustained level change is
                      treated as genuine, not as an anomaly to
                      reject -- same philosophy as Orca's constant gate,
                      but applied only where the run really settles
                      on a new value).
          'clean'  -> WHATEVER the standard 4-phase pipeline has
                      already produced (purified_row_flat) -- not the raw:
                      a continuous non-anomalous signal (e.g. a real
                      structured oscillation) still needs the soft damping
                      of AdaptiveSignalStabilizer, which the arbiter is
                      not designed to do (see its own docstring).

        Returns (corretto, etichette, incertezza), same shape as
        co_row_flat."""
        co_safe = np.where(np.isfinite(co_row_flat), co_row_flat, purified_row_flat)
        arbiter_out, etichette, incertezza = route_and_correct(co_safe, radius=radius)
        corretto = np.where(
            etichette == "spike",
            arbiter_out,
            np.where(etichette == "regime", co_safe, purified_row_flat),
        )
        return corretto, etichette, incertezza

    def _blind_reference(self, co_row_flat: np.ndarray) -> np.ndarray:
        """BLIND clean reference for a batch row, used when x_reference
        was not provided: same "treatment" given to Kalman in the
        comparison -- real causal memory instead of a blind fixed window.

        Reuses the recursive engine of Stage 1 (AdaptiveSignalStabilizer.
        filter_data_stream): a causal scan that carries forward state + adaptive
        gain + rolling volatility over the WHOLE history of the series (same
        principle as the recursive carry of a Kalman filter).

        The engine alone is NOT enough: its anomalous branch has a minimum gain
        (k_anom_min=0.25) that always lets at least 25% of an even giant
        outlier through, and that contamination accumulates step after step in the
        carry (verified: without pre-cleaning the MSE worsens by ~10000x). First
        the severe statistical outliers are rejected with a local median (almost
        total rejection, unlike the 25% floor), THEN the cleaned series is passed
        into the recursive engine for the causal memory over the whole history."""
        v = as_default_array(co_row_flat)
        finiti = v[np.isfinite(v)]
        fallback = float(np.median(finiti)) if finiti.size else 0.0
        v_safe = np.where(np.isfinite(v), v, fallback)
        n = v_safe.size
        local_med = np.empty(n, dtype=get_default_dtype())
        for i in range(n):
            a, b = max(0, i - 3), min(n, i + 4)
            local_med[i] = np.median(v_safe[a:b])
        mad = np.median(np.abs(v_safe - local_med)) or 1e-12
        resid_std = 1.4826 * mad
        is_gross_outlier = np.abs(v_safe - local_med) > 6.0 * resid_std
        v_pre_cleaned = np.where(is_gross_outlier, local_med, v_safe)
        rif = self.stabilizer.filter_data_stream(v_pre_cleaned)
        return as_default_array(rif)

    def _run_input_shield_kernel(
        self,
        f1: jnp.ndarray,
        c_chunk: jnp.ndarray,
        gate: jnp.ndarray,
        initial_damping: jnp.ndarray,
        hard_clamp_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Only the final elementwise combination of the input shield
        (jnp.where/pure arithmetic), precompiled only once. f1 and gate
        remain computed with separate EAGER calls (filter_batch_scenarios/
        the calibration and the internal numpy
        conversions -- they are not composable inside an outer jax.jit, tried:
        TracerArrayConversionError).

        hard_clamp_mask (computed by the caller on cl_chunk_raw/co_chunk_raw,
        BEFORE any log10 compression): where True, the dynamic gate (removed in v1.1.0)
        (potentially fooled by an already damped signal) is
        bypassed and forced to 0.99 -- emergency protocol for the
        macro-anomalies detected on the original data (not on the already
        compressed ones). Deliberately HIGHER than the maximum gate 0.85 of the
        normal fluid design (which only applies below the raw threshold):
        the emergency scenario must not inherit the 25% gain floor
        designed for ordinary disturbances -- here the goal is to
        knock the residual outlier down to infinitesimal fractions."""
        gt = jnp.where(jnp.isnan(gate), initial_damping, gate)
        gt = jnp.where(hard_clamp_mask, 0.99, gt)
        diff = f1 - c_chunk
        candidate = f1 - (gt * diff)
        return jnp.where(jnp.isnan(candidate), c_chunk, candidate)

    def _run_output_shield_kernel(
        self, ai_output_flat: jnp.ndarray, ref_flat: jnp.ndarray
    ) -> tuple:
        """Pure JAX pipeline of the output shield, isolated to be
        precompiled only once (see _compiled_output_shield_kernel)."""
        final_hardened_flat = apply_damping_blend(ai_output_flat, ref_flat)
        final_hardened_flat = jnp.where(
            jnp.isnan(final_hardened_flat), 0.0, final_hardened_flat
        )
        raw_out_noto = jnp.isfinite(ai_output_flat)
        margine = jnp.where(
            raw_out_noto,
            jnp.abs(ai_output_flat - final_hardened_flat),
            jnp.abs(final_hardened_flat),
        )
        return final_hardened_flat, margine

    def _execute_4_phase_input_shield(
        self, cl_chunk_raw: np.ndarray, co_chunk_raw: np.ndarray
    ) -> tuple:
        """The 4 phases of the input shield on an already prepared flat chunk; returns (decompressed, margin)."""
        v64_cl, v64_co = get_default_dtype().type(cl_chunk_raw), get_default_dtype().type(co_chunk_raw)
        # Safety barrier: noise intensity on the original RAW data,
        # before any log10 compression -- the compression
        # renormalizes each value individually to the same target scale,
        # so computed on the compressed data the true intensity would be lost.
        raw_noise = np.abs(v64_co - v64_cl)
        # SHARED SCALE FACTOR (no longer one per side): one factor per
        # element, derived ONLY from the clean value, applied to BOTH
        # clean and corrupted. Before there were two independent factors (one from
        # v64_cl, one from v64_co): since each renormalizes its own
        # value to the SAME target magnitude (10^val_e) regardless of
        # how large it was originally, any two values with the same
        # sign collapsed to the IDENTICAL compressed value -- destroying
        # every relative difference, hence the dynamics of the anomaly, even before
        # Stage 1/Stage 2 saw it (verified: a total pass-
        # through, without any real protection, reconstructed the
        # exact clean in the same way). With a shared factor based
        # on the clean, the compressed difference stays proportional to the true
        # difference (co_chunk - c_chunk = fact * (v64_co - v64_cl)) --
        # sign preserved (fact always positive), same numerical
        # robustness for the normal case (target scale ~10^val_e when co is
        # near cl), but without cancelling the macro-anomalies that the
        # barrier on raw_noise must be able to see upstream.
        mask_cl = v64_cl != 0.0
        exp10_cl = np.where(mask_cl, np.log10(np.abs(v64_cl) + 1e-15), 0.0)
        # FLOOR ON exp10_cl: do not amplify beyond the target scale (10^val_e).
        # Without this, a near-zero clean value (even just a
        # floating point residual not exactly 0.0 -- e.g. sin(4*pi) ~= -4.9e-16 in
        # a reference that crosses zero) makes fact_shared explode
        # (10^(val_e - exp10_cl) with strongly negative exp10_cl), and since
        # the SAME shared factor also applies to the corrupted (see above),
        # its compressed counterpart shoots to huge values even if the
        # raw noise was small (verified: a single such point
        # brought fact_shared to ~10^10 and the compressed corrupted to ~10^9). That
        # single out-of-scale element then poisons calibrate_macro_context
        # (AdaptiveSignalStabilizer), which sees a huge jump in the global diff
        # and activates the "panic mode" for the WHOLE batch, not just for that point
        # -- degrading the reconstruction everywhere, not only at the transition.
        # With the floor, a value already at the target scale or smaller is not
        # amplified further (fact_shared stops at 1.0): there is in any case
        # no way to precisely reconstruct a clean
        # indistinguishable from zero, but the rest of the batch is no longer
        # dragged down with it.
        exp10_cl = np.maximum(exp10_cl, self.val_e)
        fact_shared = np.where(mask_cl, 10 ** (self.val_e - exp10_cl), 1.0)
        c_chunk_np = v64_cl * fact_shared
        co_chunk_np = v64_co * fact_shared
        # SECOND SAFETY LEVEL: the floor above keeps fact_shared from
        # exploding, but is NOT enough on its own. When the clean is exactly zero
        # (mask_cl=False, fact_shared=1.0, no compression), its
        # corrupted counterpart stays at RAW scale (e.g. ~0.04) while every
        # other point of the batch is compressed to the target scale (~10^val_e,
        # e.g. 1e-4) -- a jump of ~400x between an element and its neighbors,
        # enough to restart the causal memory of
        # AdaptiveSignalStabilizer from a wrong initial state and degrade
        # the reconstruction of the following points too (not only of the
        # offending one -- verified). Direct clip on the ALREADY compressed value
        # to a wide multiple of the target scale: the normal variation
        # (co near cl) stays well within the margin, the pathological jump
        # is switched off whatever its exact cause.
        clip_bound = self._CO_CHUNK_CLIP_MULT * (10.0**self.val_e)
        co_chunk_np = np.clip(co_chunk_np, -clip_bound, clip_bound)
        c_chunk = jnp.array(c_chunk_np).reshape(1, -1)
        co_chunk = jnp.array(co_chunk_np).reshape(1, -1)
        hard_clamp_mask = jnp.array(raw_noise > self.hard_clamp_threshold).reshape(1, -1)

        # State Flush: the same authoritative signal of point 2 passed
        # to Stage 1 as well -- otherwise its minimum gain floor
        # (25% by default) always lets a residual of a
        # macro-anomaly through into the recursive state, which then decays slowly
        # contaminating the following samples (verified: without this,
        # 3 normal samples after a huge peak stayed contaminated
        # by hundreds/thousands of units).
        ref = self.stabilizer.filter_batch_scenarios(
            co_chunk, hard_clamp_mask=hard_clamp_mask
        )
        # isfinite (not just isnan): with the shared factor co_chunk is no
        # longer guaranteed to stay near 10^val_e for extreme corruptions --
        # an overflow (Inf, not NaN) must be intercepted here otherwise it
        # would survive the blending even under hard-clamp.
        f1 = jnp.where(jnp.isfinite(ref), ref, c_chunk)
        # Stage 2 evaluated on the compressed signal BEFORE the Damping (co_chunk),
        # not on f1 (already damped by Stage 1) -- otherwise the ABC
        # gating judges how anomalous a signal is that has already been
        # partly corrected, underestimating the original anomaly.
        # Constant gate, declared. The history is in the CHANGELOG
        # (v1.1.0) and in the commit that removed collatz.py: the dynamic
        # gate based on the Collatz radical was not
        # discriminating (the radical of an integer has no monotonic
        # relation with its magnitude, so it saturated at 0.85
        # even with zero noise), and the continuous variant did
        # worse than the constant on 6 scenarios out of 7 (RMSE).
        gate = jnp.full_like(co_chunk, 0.85)
        fh = self._compiled_input_shield_kernel(
            f1, c_chunk, gate, jnp.float32(self.initial_damping), hard_clamp_mask
        )

        self.last_kappa = float(curvature(fh.flatten(), c_chunk.flatten()))
        jax.block_until_ready(fh)

        # no clamp to zero here: with the sign preserved above, fh can legitimately
        # be negative -- zeroing it would cancel valid clean data (see tests)
        filtered_enc_np = np.array(fh).flatten()
        dec_chunk = filtered_enc_np / fact_shared
        # error margin in original units: how much the received value (raw,
        # corrupted) was moved to get to the corrected value -- small
        # corrections = value already reliable, large corrections = trust little.
        # Where the raw was not even known (NaN/Inf) there is no "how much I
        # moved it": the margin is the whole magnitude of the reconstruction itself
        # (maximum honest uncertainty, NEVER NaN -- otherwise the mean/max collapse
        # to NaN exactly in the cases where the shield should be most useful).
        raw_noto = np.isfinite(v64_co)
        margine_chunk = np.where(
            raw_noto, np.abs(v64_co - dec_chunk), np.abs(dec_chunk)
        )
        del c_chunk, co_chunk, fh, filtered_enc_np
        return dec_chunk, margine_chunk

    def _execute_4_phase_input_shield_batch(
        self, cl_batch_raw: np.ndarray, co_batch_raw: np.ndarray
    ) -> tuple:
        """Vectorized version of _execute_4_phase_input_shield: cl_batch_raw/
        co_batch_raw have shape (B, F) -- B independent rows processed in
        ONE call instead of B separate calls (it was the documented latency
        bottleneck: ~87% JAX dispatch, not real computation, same
        pattern already solved in utility/hybrid_engine.hybrid_shield for Armor).
        Uses filter_batch_scenarios_independent (per-row calibration) instead
        of filter_batch_scenarios (calibration shared over the whole batch)
        -- otherwise rows with different scale/noise would contaminate each
        other, a silently different result from the loop version.
        Verified bit by bit against _execute_4_phase_input_shield called
        row by row, see test_orca_vectorized_batch_matches_loop."""
        # np.float64(arr) on a single-element array collapses it into a
        # 0-d scalar instead of casting the dtype (hence the Deprecation
        # Warning: "Conversion of an array with ndim > 0 to a scalar") --
        # it broke the B=1,F=1 case (shape (1,1) -> () -> IndexError further
        # downstream in filter_batch_scenarios_independent on original_shape[0]).
        # astype always preserves the shape, for any number of elements.
        v64_cl, v64_co = (
            cl_batch_raw.astype(get_default_dtype()),
            co_batch_raw.astype(get_default_dtype()),
        )
        raw_noise = np.abs(v64_co - v64_cl)
        mask_cl = v64_cl != 0.0
        exp10_cl = np.where(mask_cl, np.log10(np.abs(v64_cl) + 1e-15), 0.0)
        exp10_cl = np.maximum(exp10_cl, self.val_e)
        fact_shared = np.where(mask_cl, 10 ** (self.val_e - exp10_cl), 1.0)
        c_chunk_np = v64_cl * fact_shared
        co_chunk_np = v64_co * fact_shared
        clip_bound = self._CO_CHUNK_CLIP_MULT * (10.0**self.val_e)
        co_chunk_np = np.clip(co_chunk_np, -clip_bound, clip_bound)
        c_chunk = jnp.array(c_chunk_np)
        co_chunk = jnp.array(co_chunk_np)
        hard_clamp_mask = jnp.array(raw_noise > self.hard_clamp_threshold)

        ref = self.stabilizer.filter_batch_scenarios_independent(
            co_chunk, hard_clamp_mask=hard_clamp_mask
        )
        f1 = jnp.where(jnp.isfinite(ref), ref, c_chunk)
        # Constant gate, declared. The history is in the CHANGELOG
        # (v1.1.0) and in the commit that removed collatz.py: the dynamic
        # gate based on the Collatz radical was not
        # discriminating (the radical of an integer has no monotonic
        # relation with its magnitude, so it saturated at 0.85
        # even with zero noise), and the continuous variant did
        # worse than the constant on 6 scenarios out of 7 (RMSE).
        gate = jnp.full_like(co_chunk, 0.85)
        fh = self._compiled_input_shield_kernel(
            f1, c_chunk, gate, jnp.float32(self.initial_damping), hard_clamp_mask
        )

        # on a multi-row batch this is the curvature over the WHOLE batch
        # flattened together, not only on the last row as in fact happened
        # in the loop path (self.last_kappa was overwritten at each
        # iteration) -- a diagnostic value, not used to correct
        # the output, so the change does not affect numerical correctness.
        self.last_kappa = float(curvature(fh.flatten(), c_chunk.flatten()))
        jax.block_until_ready(fh)

        filtered_enc_np = np.array(fh)
        dec_batch = filtered_enc_np / fact_shared
        raw_noto = np.isfinite(v64_co)
        margine_batch = np.where(
            raw_noto, np.abs(v64_co - dec_batch), np.abs(dec_batch)
        )
        del c_chunk, co_chunk, fh, filtered_enc_np
        return dec_batch, margine_batch

    def _execute_4_phase_output_shield(
        self, ai_output: jnp.ndarray, output_reference: jnp.ndarray
    ) -> tuple:
        """The 4 phases of the output shield; returns (corrected_output, margin), same shape as ai_output."""
        orig_shape = ai_output.shape
        B = orig_shape[0]
        ai_output_flat, ref_flat = (
            ai_output.reshape(B, -1),
            output_reference.reshape(B, -1),
        )
        if ai_output_flat.shape != ref_flat.shape:
            # incompatible reference: blind self-consistency instead of zeroing everything
            ref_flat = self.stabilizer.filter_batch_scenarios(ai_output_flat)
        # error margin in OUTPUT SPACE: how much the model's raw answer
        # was corrected by the blending -- same logic as the input shield (never
        # NaN even if the model produces a non-finite output)
        final_hardened_flat, margine_flat = self._compiled_output_shield_kernel(
            ai_output_flat, ref_flat
        )
        x_final = final_hardened_flat.reshape(orig_shape)
        margine = margine_flat.reshape(orig_shape)
        jax.block_until_ready(x_final)
        del ai_output_flat, ref_flat, final_hardened_flat
        return x_final, margine

    def protect_and_forward(
        self,
        ai_model_callable: Callable | None,
        x_corrupted: np.ndarray,
        x_reference: np.ndarray | None = None,
        use_input_shield: bool = True,
        use_model_injection: bool = True,
        use_output_shield: bool = True,
        use_arbiter: bool = False,
    ) -> np.ndarray:
        """Runs the 4 phases (input shield -> model -> output shield) and returns the protected output.

        use_arbiter — if True, after the standard input shield it routes each
        point to the right corrector (core/online/protect/arbiter.py): isolated
        impulse -> hard rejection, sustained regime change -> passes
        raw, no point anomaly -> the result already
        produced by the standard shield remains (its soft damping, not a
        pass-through). Default False: behavior and results identical to
        before this option on all the existing tests. It populates
        etichette_arbitro/incertezza_arbitro; if x_reference is known,
        it also updates tipi_corruzione_visti(slice_shape)."""
        self.etichette_arbitro = None
        self.incertezza_arbitro = None
        self.incertezza_arbitro_media = 0.0
        is_simple_data_test = ai_model_callable is None or not use_model_injection
        if is_simple_data_test:
            logger.info(
                "CONTRAZIONE LOGICA DETECTED: Riconosciuto Test di Protezione Dati Semplice (No IA Model)."
            )
            use_model_injection = False

        # A 1D array (e.g. a single series from a sensor/pipeline) is NOT a
        # batch of N independent scalars: it is ONE instance with N samples
        # correlated in time. Without this promotion, B=N and each sample
        # was processed alone (slice_shape=()), zeroing the context
        # on which blind-mode outlier detection relies (without
        # x_reference) -- the main use case documented in the README.
        was_1d = x_corrupted.ndim == 1
        if was_1d:
            x_corrupted = np.asarray(x_corrupted).reshape(1, -1)
            if x_reference is not None:
                x_reference = np.asarray(x_reference).reshape(1, -1)

        orig_shape = x_corrupted.shape
        B = orig_shape[0]
        slice_shape = orig_shape[1:]
        t_start = time.time()

        if use_input_shield:
            logger.info(
                "Attivazione SCUDO ENTRATA (4 Fasi) su Ipervolume: %s", orig_shape
            )
            x_corrupted_np = np.array(x_corrupted)
            if x_reference is None:
                x_reference_np = np.zeros_like(x_corrupted_np)
                for b in range(B):
                    row_flat = x_corrupted_np[b].flatten()
                    recalled = self._recall_reference(row_flat, slice_shape)
                    if recalled is not None:
                        x_reference_np[b] = recalled.reshape(slice_shape)
                    else:
                        x_reference_np[b] = self._blind_reference(row_flat).reshape(
                            slice_shape
                        )
            else:
                x_reference_np = np.array(x_reference)
                self._remember_reference(x_reference_np, slice_shape)
            purified_batch = np.zeros(orig_shape, dtype=get_default_dtype())
            margine_batch = np.zeros(orig_shape, dtype=get_default_dtype())
            flat_cl_all = x_reference_np.reshape(B, -1)
            flat_co_all = x_corrupted_np.reshape(B, -1)
            total_elements = flat_cl_all.shape[1]
            if total_elements <= self.chunk_threshold:
                # Vectorized path: the B rows are processed in ONE
                # call (jax.vmap below, independent calibration per
                # row) instead of B separate JAX dispatches -- see
                # _execute_4_phase_input_shield_batch and CHANGELOG for the
                # measured gain. Each row is completely independent
                # of the others at this stage (no state shared between
                # rows here: _reference_bank/_corruption_type_memory are
                # not touched in this block).
                out_batch, marg_batch = self._execute_4_phase_input_shield_batch(
                    flat_cl_all, flat_co_all
                )
                purified_batch = out_batch.reshape(orig_shape)
                margine_batch = marg_batch.reshape(orig_shape)
                self._gc_se_ram_bassa()
            else:
                # Unchanged fallback: a single row exceeds on its own
                # chunk_threshold (rare), it still has to be split into blocks.
                for b in range(B):
                    flat_cl, flat_co = flat_cl_all[b], flat_co_all[b]
                    out_flat = np.zeros_like(flat_cl)
                    margine_flat = np.zeros_like(flat_cl)
                    c_size = self.chunk_threshold
                    for start_idx in range(0, total_elements, c_size):
                        end_idx = min(start_idx + c_size, total_elements)
                        purified_chunk, margine_chunk = (
                            self._execute_4_phase_input_shield(
                                flat_cl[start_idx:end_idx], flat_co[start_idx:end_idx]
                            )
                        )
                        out_flat[start_idx:end_idx] = purified_chunk
                        margine_flat[start_idx:end_idx] = margine_chunk
                        self._gc_se_ram_bassa()
                    purified_batch[b] = out_flat.reshape(slice_shape)
                    margine_batch[b] = margine_flat.reshape(slice_shape)

            if use_arbiter:
                etichette_batch = np.empty(orig_shape, dtype=object)
                incertezza_batch = np.zeros(orig_shape, dtype=get_default_dtype())
                for b in range(B):
                    co_flat = x_corrupted_np[b].flatten()
                    pur_flat = purified_batch[b].flatten()
                    corretto_flat, etichette_flat, incertezza_flat = (
                        self._applica_arbitro(co_flat, pur_flat)
                    )
                    purified_batch[b] = corretto_flat.reshape(slice_shape)
                    etichette_batch[b] = etichette_flat.reshape(slice_shape)
                    incertezza_batch[b] = incertezza_flat.reshape(slice_shape)
                    if x_reference is not None:
                        self._aggiorna_memoria_corruzione(etichette_flat, slice_shape)
                self.etichette_arbitro = etichette_batch
                self.incertezza_arbitro = incertezza_batch
                self.incertezza_arbitro_media = float(np.mean(incertezza_batch))

            x_for_model = jnp.array(purified_batch)
            self.margine_ingresso = margine_batch
            self.margine_ingresso_medio = float(np.mean(margine_batch))
            self.margine_ingresso_max = float(np.max(margine_batch))
            logger.info(
                "Input purificato in %.3fs. Margine d'errore: medio=%.4g, max=%.4g",
                time.time() - t_start,
                self.margine_ingresso_medio,
                self.margine_ingresso_max,
            )
        else:
            logger.info(
                "SCUDO ENTRATA disattivato. I dati transitano senza pre-filtri."
            )
            x_for_model = jnp.array(x_corrupted)
            self.margine_ingresso = None
            self.margine_ingresso_medio = self.margine_ingresso_max = 0.0

        if use_model_injection:
            t_ia = time.time()
            ai_output = ai_model_callable(x_for_model)
            jax.block_until_ready(ai_output)
            logger.info(
                "Risposta IA ottenuta in %.3fs. Shape Output: %s",
                time.time() - t_ia,
                ai_output.shape,
            )
        else:
            logger.info(
                "INIEZIONE MODELLO bypassata. I dati purificati procedono verso la barriera spettrale."
            )
            ai_output = x_for_model

        if use_output_shield:
            t_out = time.time()
            logger.info("Attivazione SCUDO USCITA (4 Fasi) su Spettro Terminale...")
            if use_model_injection and x_reference is not None:
                # reference in OUTPUT SPACE: the model's response to the clean
                # data, not the purified input (different space if the model is
                # transformative, e.g. classifiers/embeddings/non-linear networks)
                output_reference = ai_model_callable(jnp.array(x_reference))
                jax.block_until_ready(output_reference)
            else:
                # no known clean reference: blind self-consistency on the output
                # itself (same stabilizer as Phase 1, applied here to the output)
                output_reference = self.stabilizer.filter_batch_scenarios(
                    ai_output.reshape(ai_output.shape[0], -1)
                ).reshape(ai_output.shape)
            x_final, margine_out = self._execute_4_phase_output_shield(
                ai_output, output_reference
            )
            self.margine_uscita = np.array(margine_out)
            self.margine_uscita_medio = float(jnp.mean(margine_out))
            self.margine_uscita_max = float(jnp.max(margine_out))
            logger.info(
                "Output rinormalizzato in %.3fs. Margine d'errore: medio=%.4g, max=%.4g",
                time.time() - t_out,
                self.margine_uscita_medio,
                self.margine_uscita_max,
            )
        else:
            logger.info("SCUDO USCITA disattivato. Emissione del flusso lineare.")
            x_final = ai_output
            self.margine_uscita = None
            self.margine_uscita_medio = self.margine_uscita_max = 0.0

        if was_1d:
            x_final = x_final.reshape(-1)
            if self.margine_ingresso is not None:
                self.margine_ingresso = self.margine_ingresso.reshape(-1)
            if self.margine_uscita is not None:
                self.margine_uscita = np.asarray(self.margine_uscita).reshape(-1)
            if self.etichette_arbitro is not None:
                self.etichette_arbitro = self.etichette_arbitro.reshape(-1)
            if self.incertezza_arbitro is not None:
                self.incertezza_arbitro = self.incertezza_arbitro.reshape(-1)

        logger.info(
            "Transito concluso. Sistema sigillato in %.3f secondi totali.",
            time.time() - t_start,
        )
        return x_final
