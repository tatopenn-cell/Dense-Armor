# -*- coding: utf-8 -*-
"""
core/compiler.py
==============================
DynamicAICodegen — compila ricette testuali in pipeline JAX eseguibili.

Fix applicati
-------------
- BILANCIAMENTO AUREO: Integrazione dei poli di accoppiamento basati sulla costante PHI.
- INTEGRITÀ XLA: Tipi forzati lato CPU per prevenire eccezioni sui file binari (.bin).
"""

import logging
import os
import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

# =========================================================================
# ANCORAGGIO GEOMETRICO E BAKING DELLO SPETTRO (XLA BINARY SAFE)
# =========================================================================
_PHI_128   = np.longdouble(1.0 + np.sqrt(5.0)) / 2.0
_PHI       = float(_PHI_128)
_ALPHA     = float(0.25)
_PHI_FOUR  = float(_PHI_128 ** 4)  # Moltiplicatore topologico della barriera (~6.854)

# ── Mappa comandi → branch index (contigui 0-N per lax.switch) ───────────────
CMD_MAP = {
    'identity':    0,
    'relu':        1,
    'sigmoid':     2,
    'tanh':        3,
    'scale':       4,
    'dropout':     5,
    'clip':        6,
    'l2_normalize':7,
}
_N_BRANCHES = 8


# ── step JIT ─────────────────────────────────────────────────────────────────

@jax.jit
def _execute_ai_instruction_step(carry: tuple, instruction: jnp.ndarray) -> tuple:
    """
    Esegue un singolo passo della pipeline AI su JAX.
    carry      = (data_vector, prng_key)
    instruction = [cmd_id, p1, p2, _reserved]
    """
    data, prng_key = carry
    cmd_id = instruction[0].astype(jnp.int32)
    p1, p2 = instruction[1], instruction[2]

    # ── Branch helpers ────────────────────────────────────────────────────────
    def _identity(args: tuple) -> tuple:
        """Nessuna trasformazione: passa il dato invariato."""
        d, _p1, _p2, _k = args
        return d, _k

    def _relu(args: tuple) -> tuple:
        """max(0, d) elemento per elemento."""
        d, _p1, _p2, _k = args
        return jnp.maximum(0.0, d), _k

    def _sigmoid(args: tuple) -> tuple:
        """Sigmoide elemento per elemento."""
        d, _p1, _p2, _k = args
        return jax.nn.sigmoid(d), _k

    def _tanh(args: tuple) -> tuple:
        """Tangente iperbolica elemento per elemento."""
        d, _p1, _p2, _k = args
        return jnp.tanh(d), _k

    def _scale(args: tuple) -> tuple:
        """Trasformazione affine d * p1 + p2."""
        d, _p1, _p2, _k = args
        return d * _p1 + _p2, _k

    def _dropout(args: tuple) -> tuple:
        """Dropout stocastico con probabilita' p1, avanzando la chiave PRNG."""
        d, _p1, _p2, _k = args
        next_k, subkey = jax.random.split(_k)
        safe_p1 = jnp.where(_p1 > 0.0, _p1, 1.0)
        mask    = jax.random.bernoulli(subkey, p=safe_p1, shape=d.shape)
        return jnp.where(mask, d / safe_p1, 0.0), next_k

    def _clip(args: tuple) -> tuple:
        """Clip del dato nell'intervallo [p1, p2]."""
        d, _p1, _p2, _k = args
        return jnp.clip(d, _p1, _p2), _k

    def _l2_normalize(args: tuple) -> tuple:
        """Normalizzazione L2 del dato (norma 1, o invariato se gia' zero)."""
        d, _p1, _p2, _k = args
        norm     = jnp.linalg.norm(d)
        safe_n   = jnp.where(norm > 0.0, norm, 1.0)
        return d / safe_n, _k

    branches = [_identity, _relu, _sigmoid, _tanh, _scale, _dropout, _clip, _l2_normalize]

    safe_cmd = jnp.clip(cmd_id, 0, _N_BRANCHES - 1)
    result, next_key = jax.lax.switch(safe_cmd, branches, (data, p1, p2, prng_key))

    return (result, next_key), None


# ─────────────────────────────────────────────────────────────────────────────
# DynamicAICodegen (Inizio Classe)
# ─────────────────────────────────────────────────────────────────────────────

class DynamicAICodegen:
    """
    Compila ricette testuali in matrici di istruzioni JAX ed esegue pipeline
    con chunking Anti-OOM e calcolo del gradiente via JAX AD.
    """

    def __init__(self, seed: int = 42) -> None:
        """seed — seme iniziale della chiave PRNG usata dalle istruzioni stocastiche (es. dropout)."""
        self.cmd_map  = CMD_MAP
        self.base_key = jax.random.PRNGKey(seed)

    # ── Compilazione ──────────────────────────────────────────────────────────

    def compile_pipeline(self, text_instructions: list) -> np.ndarray:
        """
        Converte una lista di istruzioni testuali/tuple in matrice float64
        di shape (N, 4): [cmd_id, p1, p2, reserved].
        """
        compiled = []
        for cmd in text_instructions:
            if isinstance(cmd, tuple):
                name = str(cmd[0]).lower().strip()
                p1   = float(cmd[1]) if len(cmd) > 1 else 0.0
                p2   = float(cmd[2]) if len(cmd) > 2 else 0.0
            else:
                name = str(cmd).lower().strip()
                if name == 'dropout':
                    p1, p2 = 0.8, 0.0
                elif name == 'clip':
                    p1, p2 = -1.0, 1.0
                elif name == 'scale':
                    p1, p2 = float(_PHI), float(_ALPHA) # <-- Sintonizzazione d'Asse Geometrica
                else:
                    p1, p2 = 0.0, 0.0

            cmd_id = self.cmd_map.get(name, 0)
            compiled.append([float(cmd_id), p1, p2, 0.0])

        return np.array(compiled, dtype=np.float64)

    # ── Esecuzione dinamica ───────────────────────────────────────────────────

    def run_dynamic_pipeline(
        self,
        input_data:   np.ndarray,
        compiled_ops: np.ndarray,
    ) -> np.ndarray:
        """Esegue la pipeline compilata in un singolo lax.scan JIT."""

        @jax.jit
        def _run(d: jnp.ndarray, ops: jnp.ndarray, k: jnp.ndarray) -> tuple:
            """Esegue l'intera pipeline di istruzioni in un unico scan JIT."""
            (f_data, next_k), _ = jax.lax.scan(
                _execute_ai_instruction_step, (d, k), ops
            )
            return f_data, next_k

        res_data, updated_key = _run(
            jnp.array(input_data, dtype=jnp.float64),
            jnp.array(compiled_ops, dtype=jnp.float64),
            self.base_key,
        )
        self.base_key = updated_key
        return np.array(res_data)


    # ── Chunked execution (Anti-OOM) Advanced via Grafo Nativo XLA ───────────

    def run_pipeline_with_chunking(
        self,
        input_data:   np.ndarray,
        compiled_ops: np.ndarray,
        chunk_size:   int = 500,
    ) -> np.ndarray:
        """
        [ADVANCED ENGINE]: Suddivide la pipeline in blocchi ed esegue il chunking 
        interamente all'interno dell'acceleratore hardware senza colli di bottiglia CPU.
        """
        n_ops = len(compiled_ops)
        # Calcolo dei blocchi necessari preservando la conformazione statica
        n_chunks = (n_ops + chunk_size - 1) // chunk_size
        total_slots = n_chunks * chunk_size
        
        # Allocazione della matrice di padding condizionata
        padded_ops = np.zeros((total_slots, 4), dtype=np.float64)
        padded_ops[:n_ops] = compiled_ops
        
        # Riorganizzazione geometrica in tensore 3D (N_Chunks x Chunk_Size x 4)
        structured_chunks = padded_ops.reshape(n_chunks, chunk_size, 4)
        
        j_data = jnp.array(input_data, dtype=jnp.float64)
        j_chunks = jnp.array(structured_chunks, dtype=jnp.float64)

        @jax.jit
        def _chunked_scan_compiler(data_state: jnp.ndarray, chunks_tensor: jnp.ndarray) -> jnp.ndarray:
            """Scan a due livelli sui macro-chunk, mantenendo intatto lo stato tra un chunk e il successivo."""
            # Scan a due livelli: itera sui macro-chunk mantenendo intatto lo stato d'onda JAX
            def _chunk_iterator(carry_state: jnp.ndarray, single_chunk: jnp.ndarray) -> tuple:
                """Esegue un singolo chunk di istruzioni, propagando lo stato al chunk successivo."""
                (f_data, next_k), _ = jax.lax.scan(
                    _execute_ai_instruction_step, (carry_state, self.base_key), single_chunk
                )
                return f_data, None
            
            final_state, _ = jax.lax.scan(_chunk_iterator, data_state, chunks_tensor)
            return final_state

        res_data = _chunked_scan_compiler(j_data, j_chunks)
        return np.array(res_data)

    # ── Gradients con Regolarizzazione Topologica Aurea ───────────────────────

    def compute_gradients(
        self,
        input_data:   np.ndarray,
        compiled_ops: np.ndarray,
    ) -> np.ndarray:
        """Calcola i gradienti AD della loss, normalizzata da una costante fissa (_PHI_FOUR)."""

        def _topological_loss_fn(d: jnp.ndarray, ops: jnp.ndarray, k: jnp.ndarray) -> jnp.ndarray:
            """Loss scalare (errore quadratico normalizzato) differenziata da jax.grad."""
            (f_data, _), _ = jax.lax.scan(
                _execute_ai_instruction_step, (d, k), ops
            )
            # Normalizzazione: l'errore quadratico e' scalato da una costante fissa
            return jnp.sum(jnp.square(f_data)) / jnp.float64(_PHI_FOUR)

        grad_engine = jax.jit(jax.grad(_topological_loss_fn, argnums=0))
        j_input     = jnp.array(input_data, dtype=jnp.float64)
        j_ops       = jnp.array(compiled_ops, dtype=jnp.float64)
        
        grads       = grad_engine(j_input, j_ops, self.base_key)
        self.base_key = jax.random.split(self.base_key)[0]
        return np.array(grads)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_compiled_pipeline(
        self,
        compiled_ops: np.ndarray,
        filename: str = "compiled_recipe.npy",
    ):
        """Salva la matrice delle operazioni compilate in formato binario compresso .npy."""
        np.save(filename, compiled_ops)
        logger.info("Advanced Pipeline salvata con successo: '%s'", filename)

    def load_compiled_pipeline(
        self,
        filename: str = "compiled_recipe.npy",
    ) -> np.ndarray:
        """Carica una matrice di operazioni precedentemente salvata."""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File pipeline assente: '{filename}'")
        ops = np.load(filename)
        logger.info("Advanced Pipeline caricata con successo: '%s'  shape=%s", filename, ops.shape)
        return ops
