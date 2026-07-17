# -*- coding: utf-8 -*-
"""
Sentinel Metrology Framework - Universal AI Orchestrator Shield (ORCA)
Sottosistema: Orchestratore Dinamico Selettivo Context-Aware a 4 Fasi
Autore del Framework: Salvatore Pennacchio (Napoli, 2026)
Percorso: shield_/utility/orca.py
"""
import os, sys, time, gc, logging
from typing import Callable, Optional
import numpy as np
import psutil
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ..core.engine import AdaptiveSignalStabilizer
from ..utility.collatz import ABCollatz
from ..core.damping_operator import apply_damping_blend
from ..utility.curvature import curvature

logger = logging.getLogger(__name__)


class Orca:
    def __init__(self, static_threshold: float = 0.15, initial_damping: float = 0.85,
                 alpha: float = 0.05, val_e: float = -4.0,
                 chunk_threshold: int = 1000000, min_free_ram_percentage: float = 0.15) -> None:
        """val_e — esponente di scala target per la compressione log10 in ingresso;
        chunk_threshold — dimensione oltre la quale un batch viene processato a blocchi."""
        self.static_threshold = static_threshold
        self.initial_damping = initial_damping
        self.alpha = alpha
        self.chunk_threshold = chunk_threshold
        self.val_e = float(val_e)
        self.min_free_ram = min_free_ram_percentage
        self.stabilizer = AdaptiveSignalStabilizer(static_threshold, initial_damping, alpha)
        self.shield = ABCollatz(epsilon_target=1.0)
        self.last_kappa = 0.0
        # Kernel JAX precompilati una sola volta (stesso principio del fix
        # allo scan in AdaptiveSignalStabilizer.filter_data_stream): senza
        # questo, la pipeline eager di _execute_4_phase_*_shield ridispaccia
        # e ricompila ogni singola operazione JAX ad ogni chiamata.
        self._compiled_input_shield_kernel = jax.jit(self._run_input_shield_kernel)
        self._compiled_output_shield_kernel = jax.jit(self._run_output_shield_kernel)
        # margine d'errore (come la covarianza di Kalman, ma definito semplicemente
        # come |valore_ricevuto - valore_corretto|: quanto piu' lo scudo ha dovuto
        # spostare un valore, tanto meno ci si deve fidare del risultato in quel punto)
        self.margine_ingresso = None       # array, stessa forma dell'input: incertezza in SPAZIO INPUT
        self.margine_ingresso_medio = 0.0
        self.margine_ingresso_max = 0.0
        self.margine_uscita = None         # array, stessa forma dell'output: incertezza in SPAZIO OUTPUT
        self.margine_uscita_medio = 0.0
        self.margine_uscita_max = 0.0

    def _gc_se_ram_bassa(self) -> None:
        """gc.collect() se la RAM libera si avvicina alla soglia;
        jax.clear_caches() SOLO se la si supera davvero (emergenza reale).
        jax.clear_caches() svuota la cache di compilazione JIT di TUTTO il
        processo (non solo di Orca, non solo questo kernel): usarlo alla
        stessa soglia morbida di gc.collect() vanifica la precompilazione
        fatta una tantum in __init__ ad ogni volta che la RAM libera scende
        anche di poco sotto il margine di sicurezza -- ogni chiamata
        successiva ricompilerebbe XLA da zero, silenziosamente. Riservato
        quindi al limite duro (min_free_ram), non al margine preventivo
        (+0.10) usato solo per il gc.collect() piu' economico."""
        vm = psutil.virtual_memory()
        free_pct = vm.available / vm.total
        if free_pct < (self.min_free_ram + 0.10):
            gc.collect()
        if free_pct < self.min_free_ram:
            jax.clear_caches()

    def _blind_reference(self, co_row_flat: np.ndarray) -> np.ndarray:
        """Riferimento pulito CIECO per una riga di batch, usato quando non e'
        stato fornito x_reference: stesso "trattamento" dato a Kalman nel
        confronto -- memoria causale reale invece di una finestra fissa cieca.

        Riusa il motore ricorsivo dello Stadio 1 (AdaptiveSignalStabilizer.
        filter_data_stream): uno scan causale che porta avanti stato + gain
        adattivo + volatilita' rolling su TUTTA la storia della serie (stesso
        principio del carry ricorsivo di un Kalman filter).

        Il motore da solo NON basta: il suo ramo anomalo ha un guadagno minimo
        (k_anom_min=0.25) che lascia sempre passare almeno un 25% di un outlier
        anche gigantesco, e quella contaminazione si accumula step dopo step nel
        carry (verificato: senza pre-pulizia l'MSE peggiora di ~10000x). Prima si
        rigettano gli outlier statistici gravi con una mediana locale (rigetto
        quasi totale, a differenza del floor del 25%), POI si passa la serie
        ripulita nel motore ricorsivo per la memoria causale sull'intera storia."""
        v = np.asarray(co_row_flat, dtype=np.float64)
        finiti = v[np.isfinite(v)]
        fallback = float(np.median(finiti)) if finiti.size else 0.0
        v_safe = np.where(np.isfinite(v), v, fallback)
        n = v_safe.size
        local_med = np.empty(n)
        for i in range(n):
            a, b = max(0, i - 3), min(n, i + 4)
            local_med[i] = np.median(v_safe[a:b])
        mad = np.median(np.abs(v_safe - local_med)) or 1e-12
        resid_std = 1.4826 * mad
        is_gross_outlier = np.abs(v_safe - local_med) > 6.0 * resid_std
        v_pre_cleaned = np.where(is_gross_outlier, local_med, v_safe)
        rif = self.stabilizer.filter_data_stream(v_pre_cleaned)
        return np.asarray(rif, dtype=np.float64)

    def _run_input_shield_kernel(
        self, f1: jnp.ndarray, c_chunk: jnp.ndarray, gate: jnp.ndarray, initial_damping: jnp.ndarray
    ) -> jnp.ndarray:
        """Solo la combinazione elementwise finale dello scudo entrata
        (jnp.where/aritmetica pura), precompilata una sola volta. f1 e gate
        restano calcolati con chiamate EAGER separate (filter_batch_scenarios/
        compute_damping_gating fanno calibrazione e conversioni numpy
        interne -- non sono componibili dentro un jax.jit esterno, provato:
        TracerArrayConversionError)."""
        gt = jnp.where(jnp.isnan(gate), initial_damping, gate)
        diff = f1 - c_chunk
        candidate = f1 - (gt * diff)
        return jnp.where(jnp.isnan(candidate), c_chunk, candidate)

    def _run_output_shield_kernel(self, ai_output_flat: jnp.ndarray, ref_flat: jnp.ndarray) -> tuple:
        """Pipeline JAX pura dello scudo uscita, isolata per essere
        precompilata una sola volta (vedi _compiled_output_shield_kernel)."""
        final_hardened_flat = apply_damping_blend(ai_output_flat, ref_flat)
        final_hardened_flat = jnp.where(jnp.isnan(final_hardened_flat), 0.0, final_hardened_flat)
        raw_out_noto = jnp.isfinite(ai_output_flat)
        margine = jnp.where(raw_out_noto, jnp.abs(ai_output_flat - final_hardened_flat),
                             jnp.abs(final_hardened_flat))
        return final_hardened_flat, margine

    def _execute_4_phase_input_shield(self, cl_chunk_raw: np.ndarray, co_chunk_raw: np.ndarray) -> tuple:
        """Le 4 fasi dello scudo entrata su un chunk flat gia' pronto; ritorna (decompresso, margine)."""
        v64_cl, v64_co = np.float64(cl_chunk_raw), np.float64(co_chunk_raw)
        # segno preservato: fact_* e' sempre positivo (10**x), quindi v64_* * fact_*
        # mantiene il segno originale -- la maschera deve includere anche i negativi,
        # altrimenti restano non compressi (scala incoerente col resto del batch)
        mask_cl, mask_co = v64_cl != 0.0, v64_co != 0.0
        exp10_cl = np.where(mask_cl, np.log10(np.abs(v64_cl) + 1e-15), 0.0)
        exp10_co = np.where(mask_co, np.log10(np.abs(v64_co) + 1e-15), 0.0)
        fact_cl = np.where(mask_cl, 10 ** (self.val_e - exp10_cl), 1.0)
        fact_co = np.where(mask_co, 10 ** (self.val_e - exp10_co), 1.0)
        c_chunk = jnp.array(v64_cl * fact_cl).reshape(1, -1)
        co_chunk = jnp.array(v64_co * fact_co).reshape(1, -1)

        ref = self.stabilizer.filter_batch_scenarios(co_chunk)
        f1 = jnp.where(jnp.isnan(ref), c_chunk, ref)
        gate = self.shield.compute_damping_gating(f1, c_chunk)
        fh = self._compiled_input_shield_kernel(f1, c_chunk, gate, jnp.float32(self.initial_damping))

        self.last_kappa = float(curvature(fh.flatten(), c_chunk.flatten()))
        jax.block_until_ready(fh)
        
        # niente clamp a zero qui: col segno preservato sopra, fh puo' legittimamente
        # essere negativo -- azzerarlo cancellerebbe dati puliti validi (vedi test)
        filtered_enc_np = np.array(fh).flatten()
        dec_chunk = filtered_enc_np / fact_cl
        # margine d'errore in unita' originali: quanto il valore ricevuto (grezzo,
        # corrotto) e' stato spostato per arrivare al valore corretto -- correzioni
        # piccole = valore gia' affidabile, correzioni grandi = fidarsi poco.
        # Dove il grezzo non era nemmeno noto (NaN/Inf) non esiste un "quanto l'ho
        # spostato": il margine e' l'intera magnitudine della ricostruzione stessa
        # (incertezza massima onesta, MAI NaN -- altrimenti la media/max collassano
        # a NaN proprio nei casi in cui lo scudo dovrebbe essere piu' utile).
        raw_noto = np.isfinite(v64_co)
        margine_chunk = np.where(raw_noto, np.abs(v64_co - dec_chunk), np.abs(dec_chunk))
        del c_chunk, co_chunk, fh, filtered_enc_np
        return dec_chunk, margine_chunk

    def _execute_4_phase_output_shield(self, ai_output: jnp.ndarray, output_reference: jnp.ndarray) -> tuple:
        """Le 4 fasi dello scudo uscita; ritorna (output_corretto, margine), stessa shape di ai_output."""
        orig_shape = ai_output.shape
        B = orig_shape[0]
        ai_output_flat, ref_flat = ai_output.reshape(B, -1), output_reference.reshape(B, -1)
        if ai_output_flat.shape != ref_flat.shape:
            # riferimento incompatibile: auto-consistenza cieca invece di azzerare tutto
            ref_flat = self.stabilizer.filter_batch_scenarios(ai_output_flat)
        # margine d'errore in SPAZIO OUTPUT: quanto la risposta grezza del modello
        # e' stata corretta dal blending -- stessa logica dello scudo entrata (mai
        # NaN anche se il modello produce output non finito)
        final_hardened_flat, margine_flat = self._compiled_output_shield_kernel(ai_output_flat, ref_flat)
        x_final = final_hardened_flat.reshape(orig_shape)
        margine = margine_flat.reshape(orig_shape)
        jax.block_until_ready(x_final)
        del ai_output_flat, ref_flat, final_hardened_flat
        return x_final, margine

    def protect_and_forward(
        self,
        ai_model_callable: Optional[Callable],
        x_corrupted: np.ndarray,
        x_reference: Optional[np.ndarray] = None,
        use_input_shield: bool = True,
        use_model_injection: bool = True,
        use_output_shield: bool = True,
    ) -> np.ndarray:
        """Esegue le 4 fasi (scudo entrata -> modello -> scudo uscita) e ritorna l'output protetto."""
        is_simple_data_test = ai_model_callable is None or not use_model_injection
        if is_simple_data_test:
            logger.info("CONTRAZIONE LOGICA DETECTED: Riconosciuto Test di Protezione Dati Semplice (No IA Model).")
            use_model_injection = False

        # Un array 1D (es. una singola serie da sensore/pipeline) NON e' un
        # batch di N scalari indipendenti: e' UNA istanza con N campioni
        # correlati nel tempo. Senza questa promozione, B=N e ogni campione
        # veniva processato da solo (slice_shape=()), azzerando il contesto
        # su cui si basa il rilevamento outlier in modalita' cieca (senza
        # x_reference) -- il caso d'uso principale documentato nel README.
        was_1d = (x_corrupted.ndim == 1)
        if was_1d:
            x_corrupted = np.asarray(x_corrupted).reshape(1, -1)
            if x_reference is not None:
                x_reference = np.asarray(x_reference).reshape(1, -1)

        orig_shape = x_corrupted.shape
        B = orig_shape[0]
        slice_shape = orig_shape[1:]
        t_start = time.time()
        
        if use_input_shield:
            logger.info("Attivazione SCUDO ENTRATA (4 Fasi) su Ipervolume: %s", orig_shape)
            x_corrupted_np = np.array(x_corrupted)
            if x_reference is None:
                x_reference_np = np.zeros_like(x_corrupted_np)
                for b in range(B):
                    x_reference_np[b] = self._blind_reference(x_corrupted_np[b].flatten()).reshape(slice_shape)
            else:
                x_reference_np = np.array(x_reference)
            purified_batch = np.zeros(orig_shape, dtype=np.float64)
            margine_batch = np.zeros(orig_shape, dtype=np.float64)
            for b in range(B):
                flat_cl, flat_co = x_reference_np[b].flatten(), x_corrupted_np[b].flatten()
                total_elements = flat_cl.size
                out_flat = np.zeros_like(flat_cl)
                margine_flat = np.zeros_like(flat_cl)
                c_size = self.chunk_threshold if total_elements > self.chunk_threshold else total_elements
                for start_idx in range(0, total_elements, c_size):
                    end_idx = min(start_idx + c_size, total_elements)
                    purified_chunk, margine_chunk = self._execute_4_phase_input_shield(flat_cl[start_idx:end_idx], flat_co[start_idx:end_idx])
                    out_flat[start_idx:end_idx] = purified_chunk
                    margine_flat[start_idx:end_idx] = margine_chunk
                    self._gc_se_ram_bassa()
                purified_batch[b] = out_flat.reshape(slice_shape)
                margine_batch[b] = margine_flat.reshape(slice_shape)
            x_for_model = jnp.array(purified_batch)
            self.margine_ingresso = margine_batch
            self.margine_ingresso_medio = float(np.mean(margine_batch))
            self.margine_ingresso_max = float(np.max(margine_batch))
            logger.info("Input purificato in %.3fs. Margine d'errore: medio=%.4g, max=%.4g",
                        time.time() - t_start, self.margine_ingresso_medio, self.margine_ingresso_max)
        else:
            logger.info("SCUDO ENTRATA disattivato. I dati transitano senza pre-filtri.")
            x_for_model = jnp.array(x_corrupted)
            self.margine_ingresso = None
            self.margine_ingresso_medio = self.margine_ingresso_max = 0.0

        if use_model_injection:
            t_ia = time.time()
            ai_output = ai_model_callable(x_for_model)
            jax.block_until_ready(ai_output)
            logger.info("Risposta IA ottenuta in %.3fs. Shape Output: %s", time.time() - t_ia, ai_output.shape)
        else:
            logger.info("INIEZIONE MODELLO bypassata. I dati purificati procedono verso la barriera spettrale.")
            ai_output = x_for_model

        if use_output_shield:
            t_out = time.time()
            logger.info("Attivazione SCUDO USCITA (4 Fasi) su Spettro Terminale...")
            if use_model_injection and x_reference is not None:
                # riferimento nello SPAZIO DI OUTPUT: risposta del modello al dato
                # pulito, non l'input purificato (spazio diverso se il modello e'
                # trasformativo, es. classificatori/embedding/reti non-lineari)
                output_reference = ai_model_callable(jnp.array(x_reference))
                jax.block_until_ready(output_reference)
            else:
                # nessun riferimento pulito noto: auto-consistenza cieca sull'output
                # stesso (stesso stabilizzatore della Fase 1, applicato qui all'uscita)
                output_reference = self.stabilizer.filter_batch_scenarios(
                    ai_output.reshape(ai_output.shape[0], -1)).reshape(ai_output.shape)
            x_final, margine_out = self._execute_4_phase_output_shield(ai_output, output_reference)
            self.margine_uscita = np.array(margine_out)
            self.margine_uscita_medio = float(jnp.mean(margine_out))
            self.margine_uscita_max = float(jnp.max(margine_out))
            logger.info("Output rinormalizzato in %.3fs. Margine d'errore: medio=%.4g, max=%.4g",
                        time.time() - t_out, self.margine_uscita_medio, self.margine_uscita_max)
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

        logger.info("Transito concluso. Sistema sigillato in %.3f secondi totali.", time.time() - t_start)
        return x_final