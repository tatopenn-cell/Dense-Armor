# -*- coding: utf-8 -*-
# ABCOLLATZ CONJECTURE-DRIVEN SMOOTHING SHIELD
# SYSTEM TRANSFORM CORES FOR DYNAMIC DAMPING REGULATION

import jax
import jax.numpy as jnp
from functools import partial

class ABCollatz:
    """
    Funttore d'asse discreto per la particolarizzazione del rumore binario.
    Combina l'evoluzione di Collatz con il limite olografico ABC 
    per calcolare la sensibilità microscopica della molla di smorzamento.
    """
    def __init__(self, epsilon_target: float = 1.0):
        self.epsilon_target = epsilon_target

    @staticmethod
    @jax.jit
    def check_prime_native(num: jnp.ndarray) -> jnp.ndarray:
        """Isola il comportamento atomico dei suoni primi superiori a 2."""
        divisors = jnp.arange(2, 128, dtype=jnp.float64)
        is_divisible = jnp.where((divisors < num) & (num % divisors == 0.0), 1.0, 0.0)
        return jnp.where((num >= 2.0) & (jnp.sum(is_divisible) == 0.0), 1.0, 0.0)

    @staticmethod
    @jax.jit
    def execute_collatz_step(n: jnp.ndarray) -> jnp.ndarray:
        """Inbinaria la traiettoria d'onda smontando la stringa di bit."""
        is_even = (n % 2.0) == 0.0
        return jnp.where(is_even, n // 2.0, 3.0 * n + 1.0)

    @staticmethod
    @jax.jit
    def calculate_jax_rad(n: jnp.ndarray) -> jnp.ndarray:
        """Pialla i doppioni e le molteplicità isolando il seme primo generatore."""
        # Griglia estesa a 256 per coprire le espansioni iperboliche di Collatz
        divisors = jnp.arange(2, 256, dtype=jnp.float64)
        is_factor = jnp.where((divisors <= n) & (n % divisors == 0.0), 1.0, 0.0)
        is_prime_factor = is_factor * jax.vmap(ABCollatz.check_prime_native)(divisors)
        
        log_factors = jnp.where(is_prime_factor > 0.5, jnp.log(divisors), 0.0)
        rad_value = jnp.exp(jnp.sum(log_factors))
        
        return jnp.where(n == 0.0, 0.0, jnp.where(n == 1.0, 1.0, jnp.round(rad_value)))

    @partial(jax.jit, static_argnums=(0,))
    def evaluate_abc_discrepancy(self, a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
        """Calcola la violazione della barriera geometrica entro i limiti di indefinizione."""
        # =========================================================================
        # FRACTAL PROTECTION: Sanificazione registri interni contro infezione NaN
        # =========================================================================
        a_safe = jnp.where(jnp.isnan(a), 1.0, a)
        b_safe = jnp.where(jnp.isnan(b), 1.0, b)
        c_safe = jnp.where(jnp.isnan(c), 1.0, c)
        
        radical_product = self.calculate_jax_rad(jnp.abs(a_safe * b_safe * c_safe))
        return jnp.abs(radical_product - (jnp.abs(c_safe) ** self.epsilon_target))

    @partial(jax.jit, static_argnums=(0,))
    def compute_damping_gating(self, x_corrupted: jnp.ndarray, x_clean: jnp.ndarray) -> jnp.ndarray:
        """
        Interfaccia di aggancio per il Dynamic Damping.
        Mappa l'universo delle onde ripristinando il flusso pio lineare originario.
        """
        orig_shape = x_corrupted.shape
        
        # =========================================================================
        # INTERCEZIONE RUNTIME: Rilevazione e isolamento dei NaN dello Stadio 2
        # =========================================================================
        is_nan_corrupted = jnp.isnan(x_corrupted)
        x_corrupted_safe = jnp.where(is_nan_corrupted, x_clean, x_corrupted)
        # =========================================================================
        
        noise_b = jnp.abs(x_corrupted_safe - x_clean)
        
        # Convertiamo in indici discreti per attivare Collatz
        n_indices = jnp.round(noise_b * 100.0) + 3.0
        
        # Ripristino del corretto flattening lineare originario per evitare bug di broadcasting spaziale
        collatz_wave = jax.vmap(self.execute_collatz_step)(n_indices.flatten()).reshape(orig_shape)
        
        discrepancy_epsilon = jax.vmap(self.evaluate_abc_discrepancy)(
            x_clean.flatten(), 
            collatz_wave.flatten(), 
            x_corrupted_safe.flatten()
        ).reshape(orig_shape)
        
        steering = 1.0 / (1.0 + jnp.exp(-discrepancy_epsilon))
        
        # Se l'elemento di ingresso era NaN, forza il gating al massimo coefficiente (0.85)
        return jnp.where(is_nan_corrupted, 0.85, 0.10 + steering * (0.85 - 0.10))
