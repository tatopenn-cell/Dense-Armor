# -*- coding: utf-8 -*-
"""
utility/orca.py
Universal AI Orchestrator Shield (ORCA) -- orchestratore dinamico
selettivo context-aware a 4 fasi.
"""
import time, logging
from collections import deque
from typing import Callable, Optional
import numpy as np
import jax
import jax.numpy as jnp

from ..core.engine import AdaptiveSignalStabilizer
from ..core.memory import UniversalMemoryGuard
from ..core.noise import AIHardwareProfiler
from ..utility.collatz import ABCollatz
from ..core.damping_operator import apply_damping_blend
from ..utility.curvature import curvature
from ..utility.resonance_search import apply_fast_resonance

logger = logging.getLogger(__name__)


class Orca:
    # chunk_threshold di riferimento (il vecchio default fisso) per un host
    # "base" secondo AIHardwareProfiler: RAM < 12GB, nessuna GPU/TPU
    # (max_tensor_dim=2048, vedi core/noise.py). chunk_threshold auto-derivato
    # scala PROPORZIONALMENTE a questa baseline via max_tensor_dim, non lo
    # rimpiazza con l'euristica RAM grezza presa da sola (che core/noise.py
    # dichiara gia' arbitraria, non calibrata empiricamente, vedi CHANGELOG
    # 1.1.8) -- cosi' un host "base" ottiene lo stesso valore di sempre
    # (nessuna regressione silenziosa sul caso comune), mentre un host con
    # piu' RAM o una GPU/TPU ottiene chunk piu' grandi in proporzione.
    _CHUNK_THRESHOLD_BASELINE = 1_000_000
    _CHUNK_THRESHOLD_BASELINE_TENSOR_DIM = 2048
    # Multiplo massimo della scala target (10^val_e) oltre il quale co_chunk
    # viene tagliato, vedi _execute_4_phase_input_shield per il perche'.
    _CO_CHUNK_CLIP_MULT = 20.0

    def __init__(self, static_threshold: float = 0.15, initial_damping: float = 0.85,
                 alpha: float = 0.05, val_e: float = -4.0,
                 chunk_threshold: Optional[int] = None, min_free_ram_percentage: float = 0.15,
                 reference_memory_size: int = 32, reference_recall_min_score: float = 0.90) -> None:
        """val_e — esponente di scala target per la compressione log10 in ingresso;
        chunk_threshold — dimensione oltre la quale un batch viene processato a blocchi;
        None (default) lo auto-deriva da AIHardwareProfiler (RAM/backend
        dell'host corrente, vedi _CHUNK_THRESHOLD_BASELINE* sopra) invece del
        vecchio valore fisso uguale per qualunque macchina;
        reference_memory_size — quanti riferimenti puliti passati (per shape) tenere in
        memoria per il richiamo via risonanza in modalita' cieca (vedi _recall_reference);
        reference_recall_min_score — punteggio minimo di apply_fast_resonance sotto il
        quale un riferimento in memoria viene considerato non abbastanza simile e si
        ricade su _blind_reference (calibrato empiricamente: su vettori correlati lo
        score resta >0.95 anche con rumore moderato, su vettori scorrelati non supera
        ~0.75 — 0.90 lascia margine da entrambi i lati)."""
        self.static_threshold = static_threshold
        self.initial_damping = initial_damping
        self.alpha = alpha
        if chunk_threshold is None:
            profiler = AIHardwareProfiler()
            scale = profiler.max_tensor_dim / self._CHUNK_THRESHOLD_BASELINE_TENSOR_DIM
            chunk_threshold = int(self._CHUNK_THRESHOLD_BASELINE * scale)
            logger.info(
                "chunk_threshold auto-derivato da AIHardwareProfiler: %d (%s)",
                chunk_threshold, profiler.get_profile_summary(),
            )
        self.chunk_threshold = int(chunk_threshold)
        self.val_e = float(val_e)
        self.min_free_ram = min_free_ram_percentage
        self.reference_recall_min_score = float(reference_recall_min_score)
        self.stabilizer = AdaptiveSignalStabilizer(static_threshold, initial_damping, alpha)
        self.shield = ABCollatz(epsilon_target=1.0)
        self.last_kappa = 0.0
        # Guardia di memoria condivisa (RAM+VRAM), non piu' un check psutil
        # ad-hoc duplicato qui dentro (vedi _gc_se_ram_bassa).
        self._memory_guard = UniversalMemoryGuard(min_free_ram_percentage=min_free_ram_percentage)
        # Banca di riferimenti puliti visti in passato, una deque per shape
        # (le shape diverse non sono comparabili tra loro): quando il chiamante
        # fornisce x_reference, le sue righe vengono ricordate qui; in modalita'
        # cieca (x_reference=None) una richiesta futura con un input corrotto
        # simile a uno gia' visto puo' riusare il riferimento vero invece di
        # ripartire da zero con la sola stima locale (vedi _recall_reference).
        self._reference_bank: dict = {}
        self._reference_memory_size = int(reference_memory_size)
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
        """Delega a UniversalMemoryGuard.check_memory_safety() il controllo
        RAM/VRAM prima del prossimo chunk pesante (stesso soft-GC + hard
        jax.clear_caches() che faceva prima questo metodo, ma condiviso col
        resto della libreria invece di un check psutil ad-hoc duplicato qui).

        Differenza di comportamento rispetto a prima: sotto la soglia dura
        (min_free_ram) il Guard solleva MemoryPressureError invece di
        continuare in silenzio -- fail-fast con un errore leggibile ora,
        piuttosto che un OOM del processo qualche chunk piu' avanti."""
        self._memory_guard.check_memory_safety()

    def _recall_reference(self, row_corrupted_flat: np.ndarray, slice_shape: tuple) -> Optional[np.ndarray]:
        """Cerca nella banca dei riferimenti puliti (stessa shape) quello piu'
        simile all'input corrotto corrente via apply_fast_resonance. Ritorna
        None (nessun richiamo) se la banca per questa shape e' vuota o se il
        punteggio migliore resta sotto reference_recall_min_score -- in
        entrambi i casi il chiamante ricade su _blind_reference, il
        comportamento originale invariato."""
        bank = self._reference_bank.get(slice_shape)
        if not bank:
            return None
        candidates = np.stack(list(bank))
        scores = apply_fast_resonance(candidates, row_corrupted_flat)
        best_idx = int(np.argmax(scores))
        if float(scores[best_idx]) < self.reference_recall_min_score:
            return None
        return candidates[best_idx]

    def _remember_reference(self, x_reference_np: np.ndarray, slice_shape: tuple) -> None:
        """Registra ogni riga di un riferimento pulito fornito dal chiamante
        nella banca (una deque a dimensione fissa per shape, FIFO -- i piu'
        vecchi escono quando se ne aggiungono di nuovi oltre reference_memory_size)."""
        bank = self._reference_bank.setdefault(slice_shape, deque(maxlen=self._reference_memory_size))
        for b in range(x_reference_np.shape[0]):
            bank.append(np.asarray(x_reference_np[b].flatten(), dtype=np.float64))

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
        self,
        f1: jnp.ndarray,
        c_chunk: jnp.ndarray,
        gate: jnp.ndarray,
        initial_damping: jnp.ndarray,
        hard_clamp_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Solo la combinazione elementwise finale dello scudo entrata
        (jnp.where/aritmetica pura), precompilata una sola volta. f1 e gate
        restano calcolati con chiamate EAGER separate (filter_batch_scenarios/
        compute_damping_gating fanno calibrazione e conversioni numpy
        interne -- non sono componibili dentro un jax.jit esterno, provato:
        TracerArrayConversionError).

        hard_clamp_mask (calcolato dal chiamante su cl_chunk_raw/co_chunk_raw,
        PRIMA di qualunque compressione log10): dove True, il gate ABC/Collatz
        (potenzialmente ingannato da un segnale gia' ammortizzato) viene
        bypassato e forzato a 0.99 -- protocollo di emergenza per le
        macro-anomalie individuate sui dati originali (non su quelli gia'
        compressi). Deliberatamente PIU' alto del gate massimo 0.85 del
        design fluido normale (che si applica solo sotto la soglia grezza):
        lo scenario di emergenza non deve ereditare il pavimento di guadagno
        del 25% pensato per i disturbi ordinari -- qui l'obiettivo e'
        abbattere l'outlier residuo a frazioni infinitesimali."""
        gt = jnp.where(jnp.isnan(gate), initial_damping, gate)
        gt = jnp.where(hard_clamp_mask, 0.99, gt)
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
        # Sbarramento di sicurezza: intensita' del rumore sui dati GREZZI
        # originali, prima di qualunque compressione log10 -- la compressione
        # rinormalizza ogni valore individualmente alla stessa scala target,
        # quindi calcolata sui dati compressi la vera intensita' andrebbe persa.
        raw_noise = np.abs(v64_co - v64_cl)
        # FATTORE DI SCALA CONDIVISO (non piu' uno per lato): un fattore per
        # elemento, derivato SOLO dal valore pulito, applicato a ENTRAMBI
        # pulito e corrotto. Prima c'erano due fattori indipendenti (uno da
        # v64_cl, uno da v64_co): siccome ciascuno rinormalizza il proprio
        # valore alla STESSA magnitudine target (10^val_e) a prescindere da
        # quanto grande fosse in origine, due valori qualsiasi con lo stesso
        # segno collassavano al valore compresso IDENTICO -- distruggendo
        # ogni differenza relativa, quindi la dinamica dell'anomalia, prima
        # ancora che Stadio 1/Stadio 2 la vedessero (verificato: un pass-
        # through totale, senza nessuna vera protezione, ricostruiva il
        # pulito esatto allo stesso modo). Con un fattore condiviso basato
        # sul pulito, la differenza compressa resta proporzionale alla vera
        # differenza (co_chunk - c_chunk = fact * (v64_co - v64_cl)) --
        # segno preservato (fact sempre positivo), stessa robustezza
        # numerica per il caso normale (scala target ~10^val_e quando co e'
        # vicino a cl), ma senza cancellare le macro-anomalie che lo
        # sbarramento su raw_noise deve poter vedere a monte.
        mask_cl = v64_cl != 0.0
        exp10_cl = np.where(mask_cl, np.log10(np.abs(v64_cl) + 1e-15), 0.0)
        # FLOOR SU exp10_cl: non amplificare oltre la scala target (10^val_e).
        # Senza questo, un valore pulito quasi-zero (anche solo un residuo di
        # floating point non esattamente 0.0 -- es. sin(4*pi) ~= -4.9e-16 in
        # un riferimento che attraversa lo zero) fa esplodere fact_shared
        # (10^(val_e - exp10_cl) con exp10_cl fortemente negativo), e siccome
        # lo STESSO fattore condiviso si applica anche al corrotto (vedi sopra),
        # il suo controvalore compresso schizza a valori enormi anche se il
        # rumore grezzo era piccolo (verificato: un singolo punto del genere
        # portava fact_shared a ~10^10 e il corrotto compresso a ~10^9). Quel
        # singolo elemento fuori scala avvelena poi calibrate_macro_context
        # (AdaptiveSignalStabilizer), che vede un salto enorme nel diff globale
        # e attiva il "panic mode" per l'INTERO batch, non solo per quel punto
        # -- degradando la ricostruzione ovunque, non solo alla transizione.
        # Con il floor, un valore gia' alla scala target o piu' piccolo non
        # viene amplificato ulteriormente (fact_shared si ferma a 1.0): non
        # c'e' comunque modo di ricostruire con precisione un pulito
        # indistinguibile da zero, ma il resto del batch non viene piu'
        # trascinato giu' con lui.
        exp10_cl = np.maximum(exp10_cl, self.val_e)
        fact_shared = np.where(mask_cl, 10 ** (self.val_e - exp10_cl), 1.0)
        c_chunk_np = v64_cl * fact_shared
        co_chunk_np = v64_co * fact_shared
        # SECONDO LIVELLO DI SICUREZZA: il floor sopra impedisce a fact_shared
        # di esplodere, ma NON basta da solo. Quando il pulito e' zero esatto
        # (mask_cl=False, fact_shared=1.0, nessuna compressione), il suo
        # controvalore corrotto resta a scala GREZZA (es. ~0.04) mentre ogni
        # altro punto del batch e' compresso alla scala target (~10^val_e,
        # es. 1e-4) -- un salto di ~400x tra un elemento e i suoi vicini,
        # abbastanza per far ripartire la memoria causale di
        # AdaptiveSignalStabilizer da uno stato iniziale sballato e degradare
        # la ricostruzione anche dei punti successivi (non solo di quello
        # incriminato -- verificato). Clip diretto sul valore GIA' compresso
        # a un multiplo ampio della scala target: la variazione normale
        # (co vicino a cl) resta ben dentro il margine, il salto patologico
        # viene spento qualunque ne sia la causa esatta.
        clip_bound = self._CO_CHUNK_CLIP_MULT * (10.0 ** self.val_e)
        co_chunk_np = np.clip(co_chunk_np, -clip_bound, clip_bound)
        c_chunk = jnp.array(c_chunk_np).reshape(1, -1)
        co_chunk = jnp.array(co_chunk_np).reshape(1, -1)
        hard_clamp_mask = jnp.array(raw_noise > 0.05).reshape(1, -1)

        # State Flush: lo stesso segnale autorevole del punto 2 passato
        # anche allo Stadio 1 -- altrimenti il suo pavimento di guadagno
        # minimo (25% di default) lascia sempre passare un residuo di una
        # macro-anomalia nello stato ricorsivo, che poi decade lentamente
        # contaminando i campioni successivi (verificato: senza questo,
        # 3 campioni normali dopo un picco enorme restavano contaminati
        # per centinaia/migliaia di unita').
        ref = self.stabilizer.filter_batch_scenarios(co_chunk, hard_clamp_mask=hard_clamp_mask)
        # isfinite (non solo isnan): col fattore condiviso co_chunk non e'
        # piu' garantito restare vicino a 10^val_e per corruzioni estreme --
        # un overflow (Inf, non NaN) deve essere intercettato qui altrimenti
        # sopravviverebbe al blending anche sotto hard-clamp.
        f1 = jnp.where(jnp.isfinite(ref), ref, c_chunk)
        # Stadio 2 valutato sul segnale compresso PRIMA del Damping (co_chunk),
        # non su f1 (gia' ammortizzato dallo Stadio 1) -- altrimenti l'ABC
        # gating giudica quanto e' anomalo un segnale che e' gia' stato in
        # parte corretto, sottostimando l'anomalia originale.
        gate = self.shield.compute_damping_gating(co_chunk, c_chunk)
        fh = self._compiled_input_shield_kernel(
            f1, c_chunk, gate, jnp.float32(self.initial_damping), hard_clamp_mask
        )

        self.last_kappa = float(curvature(fh.flatten(), c_chunk.flatten()))
        jax.block_until_ready(fh)
        
        # niente clamp a zero qui: col segno preservato sopra, fh puo' legittimamente
        # essere negativo -- azzerarlo cancellerebbe dati puliti validi (vedi test)
        filtered_enc_np = np.array(fh).flatten()
        dec_chunk = filtered_enc_np / fact_shared
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
                    row_flat = x_corrupted_np[b].flatten()
                    recalled = self._recall_reference(row_flat, slice_shape)
                    if recalled is not None:
                        x_reference_np[b] = recalled.reshape(slice_shape)
                    else:
                        x_reference_np[b] = self._blind_reference(row_flat).reshape(slice_shape)
            else:
                x_reference_np = np.array(x_reference)
                self._remember_reference(x_reference_np, slice_shape)
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