# -*- coding: utf-8 -*-
"""
core/vector.py
======================
ParametricScenarioSimulator — simulazioni Monte Carlo massive via JAX vmap.
BitwisePermutationEngine    — manipolazione vettori combinatori via maschere di bit.

Fix applicati
-------------
- collapse_decision ora NON modifica in-place l'array del chiamante:
  opera su una copia interna e restituisce (result, collapsed_vector).
  Il chiamante può ignorare il vettore collassato se non gli serve.
"""

import numpy as np
import jax
import jax.numpy as jnp


# ─────────────────────────────────────────────────────────────────────────────
# BitwisePermutationEngine
# ─────────────────────────────────────────────────────────────────────────────

class BitwisePermutationEngine:
    """
    Riadattamento di simulator.py per la manipolazione di vettori combinatori.
    Usa maschere di bit per scambiare e mutare stati in array multidimensionali.
    """

    def __init__(self, n_elements: int):
        """n_elements — numero di bit del vettore combinatorio (spazio 2^n_elements)."""
        self.n    = n_elements
        self.size = 1 << n_elements     # 2^N stati possibili

    def apply_bitwise_swap(
        self,
        data:         np.ndarray,
        target_bit:   int,
        control_bit:  int,
    ) -> np.ndarray:
        """Permuta gli elementi del vettore in base a maschere binarie."""
        output   = data.copy()
        t_stride = 1 << (self.n - 1 - target_bit)
        c_stride = 1 << (self.n - 1 - control_bit)

        for i in range(self.size):
            if (i & c_stride) and not (i & t_stride):
                idx_0 = i
                idx_1 = i + t_stride
                output[idx_0], output[idx_1] = data[idx_1], data[idx_0]
        return output


# ─────────────────────────────────────────────────────────────────────────────
# ParametricScenarioSimulator
# ─────────────────────────────────────────────────────────────────────────────

class ParametricScenarioSimulator:
    """
    Simulazioni parallele massive (Monte Carlo) e collasso decisionale
    stocastico condizionato su distribuzione di probabilità reale.
    """

    @staticmethod
    def _simulation_step(carry: jnp.ndarray, single_param: jnp.ndarray) -> tuple:
        """Passo temporale scalare per jax.lax.scan."""
        current_state = carry
        next_state    = current_state * 0.95 + single_param * 0.05
        return next_state, next_state

    def run_parallel_scenarios(
        self,
        base_state:        float,
        parameters_batch:  np.ndarray,
    ) -> np.ndarray:
        """
        Esegue in parallelo (vmap) tutti i batch con jax.lax.scan per l'asse
        temporale.

        Parameters
        ----------
        base_state        — stato scalare iniziale per tutti gli scenari
        parameters_batch  — array (N_scenari, T_steps) di parametri temporali

        Returns
        -------
        np.ndarray di shape (N_scenari, T_steps)
        """
        def simulate_single_instance(params_vector: jnp.ndarray) -> jnp.ndarray:
            """Un singolo scenario (T_steps parametri) portato avanti nel tempo."""
            _, history = jax.lax.scan(
                self._simulation_step, base_state, params_vector
            )
            return history

        parallel_engine = jax.jit(jax.vmap(simulate_single_instance, in_axes=(0,)))
        return np.array(parallel_engine(jnp.array(parameters_batch)))

    def collapse_decision(
        self,
        distribution_vector: np.ndarray,
        target_idx: int,
    ) -> tuple:
        """
        Collasso decisionale stocastico condizionato dalla distribuzione.

        FIX BUG: non modifica più l'array originale in-place.
        Opera su una copia interna.

        Returns
        -------
        (result: int, collapsed_vector: np.ndarray)
            result            — 0 o 1 (scelta stocastica)
            collapsed_vector  — vettore normalizzato post-collasso
        """
        vec = np.array(distribution_vector, copy=True, dtype=np.float64)

        prob_0 = np.sum(np.abs(vec[:target_idx]))
        prob_1 = np.sum(np.abs(vec[target_idx:]))
        total  = prob_0 + prob_1

        if total < 1e-12:
            raise RuntimeError(
                "Vettore decisionale a energia zero — impossibile calcolare la scelta."
            )

        prob_0 /= total
        prob_1 /= total

        result    = int(np.random.choice([0, 1], p=[prob_0, prob_1]))
        zero_slot = 1 - result

        if zero_slot == 0:
            vec[:target_idx] = 0.0
        else:
            vec[target_idx:] = 0.0

        new_total = np.sum(np.abs(vec))
        if new_total > 0:
            vec /= new_total

        return result, vec