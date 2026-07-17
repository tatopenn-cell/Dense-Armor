# -*- coding: utf-8 -*-
"""
core/engine.py
=========================
AdaptiveSignalStabilizer — filtro adattivo per serie temporali e gradienti IA.

Caratteristiche
---------------
- Rilevazione di anomalie tramite soglia dinamica basata su volatilità rolling.
- Damping selettivo con preservazione della norma L2 per evitare esplosioni locali.
- Calibrazione macro-contesto su batch 2D (N_scenari × T_steps) con parametri dinamici.
- Implementazione interamente JAX-compatibile, con kernel vmap + scan precompilati.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

# =========================================================================
# COSTANTI NUMERICHE FISSE (XLA BINARY SAFE) -- iperparametri di damping,
# non hanno un significato fisico/matematico oltre a definire la curva di gain
# =========================================================================
_PHI_128        = np.longdouble(1.0 + np.sqrt(5.0)) / 2.0
_ZETA_TOTAL_128 = np.longdouble(60.16762236065194)

# Cast a float primitivo per non rompere la compilazione dei file binari (.bin)
_PHI        = float(_PHI_128)
_ALPHA      = float(0.25)                           # Base simmetrica (1/4)
_SIGMA      = float(np.longdouble(1.0) / (_PHI_128 ** 4)) # ~0.1458980337
_K_MAX_ZETA = float(_ZETA_TOTAL_128 / (_PHI_128 ** 4))  # Limite di damping superiore (~8.778)

@jax.jit
def dynamic_damping_gain(local_noise: jnp.ndarray,
                         k_min: float = 0.0,         # <-- gain minimo (massima trasparenza)
                         k_max: float = _K_MAX_ZETA  # <-- gain massimo (costante fissa, vedi sopra)
                         ) -> jnp.ndarray:
    """
    Curva di damping a sigmoide, funzione del rumore locale:
    - local_noise basso → gain vicino al minimo (massima trasparenza)
    - local_noise alto  → gain vicino al massimo (massima purificazione)
    """
    x = jnp.clip(local_noise, 0.0, 5.0)
    steering = 1.0 / (1.0 + jnp.exp(-(x - 2.0)))
    return k_min + steering * (k_max - k_min)


class AdaptiveSignalStabilizer:
    """
    Filtro antipanico adattivo per segnali IA caotici.
    """

    def __init__(
        self,
        static_threshold: float = 1e-5,
        initial_damping: float = 0.1,
        alpha: float = _SIGMA,               # <-- Agganciato a PHI
        anomaly_sigma_mult: float = 2.0,
        k_anom_min: float = _ALPHA,          # <-- Agganciato a PHI
        k_anom_max: float = _K_MAX_ZETA,     # <-- costante fissa (vedi sopra); sostituisce il vecchio _ALPHA+_SIGMA
        window_radius: int = 1,
        smooth_l2_blend: float = 0.5,
    ) -> None:
        # Parametri di configurazione (CPU side)
        self.threshold: float = float(static_threshold)
        self.damping: float = float(initial_damping)
        self.alpha: float = float(alpha)

        # Iperparametri per la gestione delle anomalie
        self.anomaly_sigma_mult: float = float(anomaly_sigma_mult)
        self.k_anom_min: float = float(k_anom_min)
        self.k_anom_max: float = float(k_anom_max)
        self.window_radius: int = int(max(window_radius, 0))
        self.smooth_l2_blend: float = float(np.clip(smooth_l2_blend, 0.0, 1.0))

        # Parametri dinamici (calibrati per-batch a runtime)
        self.dyn_thr: float = self.threshold
        self.dyn_dmp: float = self.damping
        self.dyn_alp: float = self.alpha
        self.noise_scalar: float = 1.0

        # Kernel JAX precompilati
        self._compiled_stream_filter = self._build_stream_filter()
        self._compiled_batch_filter = jax.jit(
            jax.vmap(
                self._process_single_scenario,
                in_axes=(0, None, None, None, None),
            )
        )
        # Kernel 1D usato da filter_data_stream: thr/dmp/alp/noise_scalar
        # passati come ARGOMENTI jit (come gia' fa _compiled_batch_filter
        # sopra), non chiusi su self.* dentro la funzione -- altrimenti
        # ogni jax.jit(lambda...) creato inline ad ogni chiamata sarebbe
        # un oggetto Python nuovo, cache-miss garantito, ricompilazione
        # XLA completa ad OGNI singola chiamata (~80ms fissi anche a
        # regime, misurato: la cache non si scalda mai).
        self._compiled_single_stream_filter = jax.jit(self._run_single_stream_scan)

    def _run_single_stream_scan(self, init_state, rest, thr, dmp, alp, noise_scalar, init_val):
        n = rest.shape[0]
        thrs = jnp.full((n,), thr)
        dmps = jnp.full((n,), dmp)
        alps = jnp.full((n,), alp)
        n_scalars = jnp.full((n,), noise_scalar)
        left_neighbors = jnp.full((n,), init_val)
        upper_neighbors = jnp.full((n,), init_val)
        is_1d_array = jnp.full((n,), True)

        def _wrapped_step(carry, step_inputs):
            return self._step_kernel(carry, step_inputs)

        scan_inputs = (rest, thrs, dmps, alps, n_scalars, left_neighbors, upper_neighbors, is_1d_array)
        return jax.lax.scan(_wrapped_step, init_state, scan_inputs)

    def calibrate_macro_context(self, raw_batch: np.ndarray) -> None:
        """
        Calibra le soglie dinamiche sul contesto globale del batch.

        Analizza la distribuzione dei salti fra campioni per individuare
        ambienti estremamente turbolenti (shock) rispetto a rumore fisiologico.
        """
        if raw_batch.size == 0:
            return

        sample_diff = np.abs(np.diff(raw_batch))
        global_std = float(np.std(raw_batch))

        # Più lo std è alto, più riduciamo l'aggressività del filtro.
        self.noise_scalar = float(1.0 / (1.0 + global_std))

        # Con un solo campione per riga (es. batch di forma (1, 1)) il diff
        # è vuoto: non c'è un "salto" da misurare, quindi non possiamo
        # decidere il panic mode su questa base. Usciamo qui mantenendo
        # noise_scalar già calibrato sopra, invece di far esplodere
        # np.max/np.mean su un array vuoto.
        if sample_diff.size == 0:
            return

        max_jump = float(np.max(sample_diff))
        mean_jump = float(np.mean(sample_diff))

        # Regime “panic mode” per shock estremi
        if max_jump > 3.0 and (max_jump / (mean_jump + 1e-5)) > 5.0:
            self.dyn_thr = 1e-5
            self.dyn_dmp = 1e-3
            self.dyn_alp = 0.999
            return

        # Regime standard: i parametri dinamici scalano con la volatilità globale e la proporzione aurea
        self.dyn_thr = float(0.10 * global_std)
        self.dyn_dmp = float(_SIGMA * (1.0 + self.noise_scalar))
        self.dyn_alp = float(0.75 + (_SIGMA * self.noise_scalar))

    # ------------------------------------------------------------------ #
    # Kernel di step interno (scan) -- filtro causale ricorsivo con gain adattivo
    # ------------------------------------------------------------------ #
    def _step_kernel(
        self,
        carry: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        inputs: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ):
        prev_filtered, current_damping, rolling_volatility, local_mean = carry
        current_val, thr, dmp, alp, n_scalar, left_neighbor, upper_neighbor, is_1d = inputs

        eps = 1e-7

        # =========================================================================
        # NATIVE HARDWARE FRACTAL SHIELD FOR RUNTIME NaN CANCELLATION
        # =========================================================================
        is_nan_input = jnp.isnan(current_val)
        
        # Ancoraggio dinamico ai registri immuni precedenti per bloccare la propagazione
        current_val_safe = jnp.where(is_nan_input, prev_filtered, current_val)
        left_neighbor_safe = jnp.where(jnp.isnan(left_neighbor), prev_filtered, left_neighbor)
        upper_neighbor_safe = jnp.where(jnp.isnan(upper_neighbor), prev_filtered, upper_neighbor)
        # =========================================================================

        diff_decay = jnp.abs(current_val_safe - prev_filtered)
        diff_spatial_x = jnp.abs(current_val_safe - left_neighbor_safe)
        diff_spatial_y = jnp.abs(current_val_safe - upper_neighbor_safe)
        
        diff = jnp.where(
            is_1d,
            diff_decay,
            0.5 * diff_decay + 0.25 * diff_spatial_x + 0.25 * diff_spatial_y
        )

        local_ref = local_mean
        local_diff = jnp.abs(current_val_safe - local_ref)
        local_scale = jnp.maximum(jnp.abs(local_ref), 1e-3)
        local_coherence = jnp.clip(1.0 - (local_diff / (2.0 * local_scale)), 0.0, 1.0)

        alpha_j = jnp.float32(self.alpha)
        new_vol = (1.0 - alpha_j) * rolling_volatility + alpha_j * diff

        sigma_mult = jnp.float32(self.anomaly_sigma_mult)
        dyn_thr = thr + (new_vol * sigma_mult)
        
        # Se l'elemento originario era NaN, viene forzato come anomalia critica
        is_anomaly = jnp.logical_or(diff > dyn_thr, is_nan_input)

        coherence_t = 1.0 - (diff_decay / 2.0)
        coherence_x = 1.0 - (diff_spatial_x / 2.0)
        coherence_y = 1.0 - (diff_spatial_y / 2.0)
        
        phi_ab_2d = jnp.clip(0.4 * coherence_t + 0.3 * coherence_x + 0.3 * coherence_y, 0.0, 1.0)
        phi_ab_1d = jnp.clip(coherence_t, 0.0, 1.0)
        phi_ab = jnp.where(is_1d, phi_ab_1d, phi_ab_2d)

        phi_mix = 0.5 * phi_ab + 0.5 * local_coherence

        # --- UNIFICAZIONE TOPOLOGICA DEI POLI DI COERENZA ---
        c_gate = jnp.float32(_ALPHA + _SIGMA)  # Configurazione di banda aurea (~0.39589)
        K_coherent = jnp.clip(
            phi_mix * (c_gate / (c_gate + new_vol + eps)),
            jnp.float32(_ALPHA - _SIGMA),       # Soglia minima dinamica (~0.1041)
            jnp.float32(1.0 - (_ALPHA - _SIGMA)), # Soglia massima dinamica (~0.8958)
        )

        k_anom_min = jnp.float32(self.k_anom_min)
        k_anom_max = jnp.float32(self.k_anom_max)
        
        # Sostituzione del polo di anomalia grezza 0.30 con la contrazione aurea di banda
        c_anom = jnp.float32(1.0 / (float(_PHI_128) ** 2)) # costante fissa (~0.3819), XLA-safe
        K_anomalous_raw = c_anom / (c_anom + diff + eps)
        K_anomalous = jnp.clip(K_anomalous_raw, k_anom_min, k_anom_max)

        anomaly_strength = jnp.clip((diff - dyn_thr) / (dyn_thr + eps), 0.0, 1.0)
        
        # Se l'input era NaN, forziamo il guadagno sul ramo anomalo massimo
        anomaly_strength_final = jnp.where(is_nan_input, 1.0, anomaly_strength)
        K = (1.0 - anomaly_strength_final) * K_coherent + anomaly_strength_final * K_anomalous

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

        g = jnp.float32(self.smooth_l2_blend)
        next_filtered = g * raw_smooth + (1.0 - g) * local_blend

        # Aggiornamento del gain di damping (curva sigmoide, vedi dynamic_damping_gain)
        calculated_damping = dynamic_damping_gain(diff)
        next_damping = jnp.where(is_anomaly, calculated_damping, current_damping)

        w_rad = max(self.window_radius, 1)
        ema_alpha = 1.0 / float(w_rad + 1)
        ema_alpha_j = jnp.float32(ema_alpha)
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
    ) -> jnp.ndarray:
        h_dim, w_dim = scenario_matrix.shape
        is_1d_flag = h_dim == 1
        
        flat_scen = scenario_matrix.flatten()
        
        left_neighbors = jnp.roll(scenario_matrix, shift=1, axis=1)
        left_neighbors = left_neighbors.at[:, 0].set(scenario_matrix[:, 0]).flatten()
        
        upper_neighbors = jnp.roll(scenario_matrix, shift=1, axis=0)
        upper_neighbors = upper_neighbors.at[0, :].set(scenario_matrix[0, :]).flatten()

        start_val = jnp.float32(flat_scen[0])
        init_state = (
            start_val,
            jnp.float32(dmp),
            jnp.float32(0.0),
            start_val,
        )

        thrs = jnp.full_like(flat_scen, thr)
        dmps = jnp.full_like(flat_scen, dmp)
        alps = jnp.full_like(flat_scen, alp)
        n_scalars = jnp.full_like(flat_scen, n_scalar)
        is_1d_array = jnp.full_like(flat_scen, is_1d_flag, dtype=jnp.bool_)

        scan_inputs = (flat_scen, thrs, dmps, alps, n_scalars, left_neighbors, upper_neighbors, is_1d_array)

        def _wrapped_step(carry, step_inputs):
            return self._step_kernel(carry, step_inputs)

        _, filtered_flat = jax.lax.scan(
            _wrapped_step,
            init_state,
            scan_inputs,
        )
        
        return filtered_flat.reshape(h_dim, w_dim)

    # ------------------------------------------------------------------ #
    # Builder del filtro streaming legacy (single-vector) - ALLINEATO XLA
    # ------------------------------------------------------------------ #
    def _build_stream_filter(self):
        """
        Costruisce e compila il filtro per una singola serie temporale 1D.
        Fissato per compatibilità di esportazione binaria nativa.
        """

        def _legacy_step(carry, current_val):
            prev, dmp_state, vol = carry

            diff = jnp.abs(current_val - prev)
            alpha_j = jnp.float32(self.alpha)
            new_vol = (1.0 - alpha_j) * vol + alpha_j * diff

            coherence = 1.0 - (diff / 2.0)
            phi_ab = jnp.clip(coherence, 0.0, 1.0)

            safe_prev = jnp.where(jnp.abs(prev) > 1e-12, prev, 1e-12)
            ratio = jnp.abs(current_val / safe_prev)
            log_ratio = jnp.clip(
                jnp.log(jnp.where(ratio > 0, ratio, 1.0)),
                -5.0,
                5.0,
            )
            
            # HARDENING DELLO STATO: cast a float32 per non corrompere la precompilazione XLA
            c_lyap = jnp.float32((float(_PHI_128) ** 3) + 0.23606797)
            v_dyn = c_lyap * log_ratio * phi_ab

            trigger = jnp.abs(v_dyn) > jnp.float32(self.threshold)

            c_gate = jnp.float32(_ALPHA + _SIGMA)
            K_coherent = jnp.clip(phi_ab * c_gate, jnp.float32(_SIGMA), c_gate)
            
            c_anom_scale = jnp.float32(_ALPHA / 2.0)
            K_anomalous = jnp.clip(
                c_anom_scale / (c_anom_scale + jnp.abs(v_dyn)),
                jnp.float32(_ALPHA),
                c_gate,
            )
            K = jnp.where(trigger, K_anomalous, K_coherent)

            next_f = (1.0 - K) * prev + K * current_val
            
            # AGGANCIO AL SERVOSTERZO DI DAMPING ANCHE NEL FLUSSO LEGACY STREAMING
            calculated_damping = dynamic_damping_gain(diff)
            next_damping = jnp.where(trigger, calculated_damping, dmp_state)

            return (next_f, next_damping, new_vol), next_f

        @jax.jit
        def _run(init_state, data):
            return jax.lax.scan(_legacy_step, init_state, data)

        return _run

    # ------------------------------------------------------------------ #
    # API pubblica - Sincronizzata sui 4 canali di carry del Kernel
    # ------------------------------------------------------------------ #
    def filter_data_stream(self, raw_data: np.ndarray) -> np.ndarray:
        """
        Filtra una singola serie temporale 1D garantendo la simmetria strutturale.
        """
        if raw_data.size == 0:
            return np.zeros_like(raw_data)

        j_raw = jnp.array(raw_data, dtype=jnp.float32)
        init_val = jnp.float32(j_raw[0])
        
        # Allineamento definitivo del carry a 4 elementi identico al batch engine
        init_state = (
            init_val,
            jnp.float32(self.damping),
            jnp.float32(0.0),
            init_val,
        )

        _, gated_stream = self._compiled_single_stream_filter(
            init_state,
            j_raw[1:],
            jnp.float32(self.dyn_thr),
            jnp.float32(self.dyn_dmp),
            jnp.float32(self.dyn_alp),
            jnp.float32(self.noise_scalar),
            init_val,
        )
        
        # Reinserimento del punto fisso iniziale preservando la topologia
        final_stream = jnp.insert(gated_stream, 0, init_val)
        return np.array(final_stream, dtype=np.float32)

    def filter_batch_scenarios(self, raw_batch: np.ndarray) -> np.ndarray:
        """
        Filtra in parallelo (vmap) un batch di scenari strutturati.
        Accetta tensori 2D nativi o array multidimensionali (3D/4D).
        Garantito XLA-Safe per l'esportazione binaria.
        """
        if raw_batch.size == 0:
            return np.zeros_like(raw_batch)

        original_shape = raw_batch.shape
        n_scenarios = int(original_shape[0])

        # FIX DEFINITIVO SINTASSI: Estrarre il contesto in modo compatibile con l'ALU
        flat_analysis_batch = raw_batch.reshape(n_scenarios, -1)
        self.calibrate_macro_context(flat_analysis_batch)

        # Adattamento geometrico statico degli assi per non rompere la compilazione AOT/XLA
        if len(original_shape) == 2:
            h_dim = 1
            w_dim = int(original_shape[1])
            structured_batch = raw_batch.reshape(n_scenarios, h_dim, w_dim)
        elif len(original_shape) == 3:
            h_dim, w_dim = int(original_shape[1]), int(original_shape[2])
            structured_batch = raw_batch
        elif len(original_shape) == 4:
            c_channels, h_dim, w_dim = int(original_shape[1]), int(original_shape[2]), int(original_shape[3])
            structured_batch = raw_batch.reshape(n_scenarios * c_channels, h_dim, w_dim)
        else:
            raise ValueError(f"Geometria del tensore non supportata dall'engine: {original_shape}")

        # Conversione in array JAX esplicito a precisione singola standard per registri GPU/CPU
        j_batch = jnp.array(structured_batch, dtype=jnp.float32)

        # Esecuzione del kernel vettorizzato precompilato JAX
        filtered_structured = self._compiled_batch_filter(
            j_batch,
            jnp.float32(self.dyn_thr),
            jnp.float32(self.dyn_dmp),
            jnp.float32(self.dyn_alp),
            jnp.float32(self.noise_scalar),
        )

        # Conversione sicura in NumPy preservando la topologia
        filtered_np = np.array(filtered_structured, dtype=np.float32)

        # Ripristino esatto della shape geometrica originale del chiamante
        return filtered_np.reshape(original_shape)