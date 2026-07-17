# -*- coding: utf-8 -*-
"""core/chunk.py.

===============================================================================
SENTINEL ENTERPRISE CHUNKER: SPAZIALE, QUANTISTICO & BEAST MODE OPTIMIZATION
===============================================================================
Modulo per la segmentazione di batch d'immagini e l'esecuzione cached XLA
delle ricette di compilazione senza overhead o ricompilazioni JIT.
"""

import jax
import jax.numpy as jnp
import numpy as np


class ImageChunker:
    """Gestore avanzato di chunking per tensori d'immagine e ricette JIT.

    Risolve i colli di bottiglia della memoria XLA/JAX sia dividendo i grandi
    batch di dati, sia spezzando le liste di istruzioni lunghe in blocchi a
    dimensione fissa.
    """

    def __init__(self, chunk_size: int = 128) -> None:
        """chunk_size — dimensione fissa di ogni blocco (batch o istruzioni)."""
        self.chunk_size = int(chunk_size)

    # ── SEZIONE SPAZIALE FOTO/BATCH (Per test21.py e main.py) ───────────────

    def split_array(self, array: np.ndarray) -> list:
        """Spezza un array multidimensionale in una lista di sotto-chunk."""
        total_samples = array.shape[0]
        num_chunks = int(np.ceil(total_samples / self.chunk_size))
        chunks = []
        for b in range(num_chunks):
            start_idx = b * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, total_samples)
            chunks.append(array[start_idx:end_idx])
        return chunks

    def merge_chunks(self, chunks_list: list) -> np.ndarray:
        """Ricombina una lista di sotto-chunk in un unico array compatto."""
        if not chunks_list:
            return np.array([], dtype=np.float32)
        # Se l'input è monodimensionale flat, usa concatenate invece di vstack
        if chunks_list[0].ndim == 1:
            return np.concatenate(chunks_list)
        return np.vstack(chunks_list)

    # ── SEZIONE BEAST MODE COMPILER (Estratta dalla logica quantistica) ─────

    def execute_pipeline_beast_mode(
        self, codegen_engine, input_vector: np.ndarray, compiled_ops: list
    ) -> np.ndarray:
        """Esegue le istruzioni del compilatore a blocchi fissi (chunk_size).

        Trapianto logico di 'run_circuit_with_chunking'. Impedisce a XLA di
        ricompilare la ricetta IA se cambia il numero di istruzioni.
        """
        output = jnp.array(input_vector, dtype=jnp.float64)

        # Spezza ed esegue la lista di comandi/operazioni
        for i in range(0, len(compiled_ops), self.chunk_size):
            chunk_ops = compiled_ops[i : i + self.chunk_size]
            output = codegen_engine.run_pipeline_with_chunking(
                output, chunk_ops, chunk_size=self.chunk_size
            )

        return np.array(output, dtype=np.float64)

    @staticmethod
    def patch_and_scan_parameters(
        template_ops: jnp.ndarray, dynamic_parameters: jnp.ndarray
    ) -> jnp.ndarray:
        """Iniezione dinamica di parametri e pesi in un ciclo nativo JAX JIT.

        Trapianto logico di 'patch_and_apply' tramite jax.lax.scan. Rileva i
        punti di iniezione contrassegnati dal marcatore -1.0.
        """

        def patch_single_op(carry: jnp.ndarray, op: jnp.ndarray) -> tuple:
            """Un passo di scan: se op e' un marcatore -1.0 lo sostituisce col prossimo parametro dinamico."""
            idx = carry
            # Se rileva lo slot parametrico quantistico/lineare attivo (-1.0)
            is_parametric = op == -1.0
            final_param = jnp.where(is_parametric, dynamic_parameters[idx], op)
            next_idx = jnp.where(is_parametric, idx + jnp.int32(1), idx)

            # Restituisce l'operazione patchata a basso livello XLA
            patched_op = jnp.array(
                [op, op, op, final_param], dtype=jnp.float64
            )
            return next_idx, patched_op

        _, patched_compiled_ops = jax.lax.scan(
            patch_single_op, jnp.int32(0), template_ops
        )
        return patched_compiled_ops
