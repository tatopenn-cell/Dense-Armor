"""Compile textual recipes into executable JAX pipelines.

The supported operations (relu, sigmoid, tanh, scale, dropout, clip,
l2_normalize) run through ``jax.lax.switch`` inside a single JIT kernel.
The types are forced to float64 on the CPU side before each run, so
saving and reloading the compiled pipeline from a binary file does not
raise.
"""

import logging
import os

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

_PHI_128 = np.longdouble(1.0 + np.sqrt(5.0)) / 2.0
_PHI = float(_PHI_128)
_ALPHA = 0.25
_PHI_FOUR = float(_PHI_128**4)

CMD_MAP = {
    "identity": 0,
    "relu": 1,
    "sigmoid": 2,
    "tanh": 3,
    "scale": 4,
    "dropout": 5,
    "clip": 6,
    "l2_normalize": 7,
}
_N_BRANCHES = 8


@jax.jit
def _execute_ai_instruction_step(carry: tuple, instruction: jnp.ndarray) -> tuple:
    """Execute one step of the AI pipeline.

    Args:
        carry: a tuple ``(data_vector, prng_key)``.
        instruction: a vector ``[cmd_id, p1, p2, _reserved]``.

    Returns:
        The updated carry and ``None`` as the per-step output.
    """
    data, prng_key = carry
    cmd_id = instruction[0].astype(jnp.int32)
    p1, p2 = instruction[1], instruction[2]

    def _identity(args: tuple) -> tuple:
        """Pass the data unchanged."""
        d, _p1, _p2, _k = args
        return d, _k

    def _relu(args: tuple) -> tuple:
        """Element-wise ``max(0, d)``."""
        d, _p1, _p2, _k = args
        return jnp.maximum(0.0, d), _k

    def _sigmoid(args: tuple) -> tuple:
        """Element-wise sigmoid."""
        d, _p1, _p2, _k = args
        return jax.nn.sigmoid(d), _k

    def _tanh(args: tuple) -> tuple:
        """Element-wise hyperbolic tangent."""
        d, _p1, _p2, _k = args
        return jnp.tanh(d), _k

    def _scale(args: tuple) -> tuple:
        """Affine transform ``d * p1 + p2``."""
        d, _p1, _p2, _k = args
        return d * _p1 + _p2, _k

    def _dropout(args: tuple) -> tuple:
        """Stochastic dropout with probability p1, advancing the PRNG key."""
        d, _p1, _p2, _k = args
        next_k, subkey = jax.random.split(_k)
        safe_p1 = jnp.where(_p1 > 0.0, _p1, 1.0)
        mask = jax.random.bernoulli(subkey, p=safe_p1, shape=d.shape)
        return jnp.where(mask, d / safe_p1, 0.0), next_k

    def _clip(args: tuple) -> tuple:
        """Clip the data in ``[p1, p2]``."""
        d, _p1, _p2, _k = args
        return jnp.clip(d, _p1, _p2), _k

    def _l2_normalize(args: tuple) -> tuple:
        """L2-normalize the data (unit norm, unchanged when already zero)."""
        d, _p1, _p2, _k = args
        norm = jnp.linalg.norm(d)
        safe_n = jnp.where(norm > 0.0, norm, 1.0)
        return d / safe_n, _k

    branches = [
        _identity,
        _relu,
        _sigmoid,
        _tanh,
        _scale,
        _dropout,
        _clip,
        _l2_normalize,
    ]

    safe_cmd = jnp.clip(cmd_id, 0, _N_BRANCHES - 1)
    result, next_key = jax.lax.switch(safe_cmd, branches, (data, p1, p2, prng_key))

    return (result, next_key), None


@jax.jit
def _run_pipeline_jit(d: jnp.ndarray, ops: jnp.ndarray, k: jnp.ndarray) -> tuple:
    """Run the whole instruction pipeline inside a single JIT scan."""
    (f_data, next_k), _ = jax.lax.scan(_execute_ai_instruction_step, (d, k), ops)
    return f_data, next_k


@jax.jit
def _run_pipeline_chunked_jit(
    data_state: jnp.ndarray, chunks_tensor: jnp.ndarray, base_key: jnp.ndarray
) -> tuple:
    """Two-level scan over macro-chunks, carrying state and PRNG key across chunks."""

    def _chunk_iterator(carry: tuple, single_chunk: jnp.ndarray) -> tuple:
        """Run one instruction chunk, propagating state and key to the next."""
        return jax.lax.scan(_execute_ai_instruction_step, carry, single_chunk)[0], None

    (final_state, next_k), _ = jax.lax.scan(
        _chunk_iterator, (data_state, base_key), chunks_tensor
    )
    return final_state, next_k


def _loss_fn(d: jnp.ndarray, ops: jnp.ndarray, k: jnp.ndarray) -> jnp.ndarray:
    """Scalar loss (normalized squared error) for :func:`jax.grad`."""
    (f_data, _), _ = jax.lax.scan(_execute_ai_instruction_step, (d, k), ops)
    return jnp.sum(jnp.square(f_data)) / jnp.float64(_PHI_FOUR)


_grad_engine = jax.jit(jax.grad(_loss_fn, argnums=0))


class DynamicAICodegen:
    """Compile textual recipes into JAX instruction matrices and run pipelines.

    Chunked execution keeps the memory bounded; gradients come from JAX
    automatic differentiation.

    Args:
        seed: initial seed of the PRNG key used by stochastic instructions
            (for instance dropout).
    """

    def __init__(self, seed: int = 42) -> None:
        """Store the command map and the initial PRNG key."""
        self.cmd_map = CMD_MAP
        self.base_key = jax.random.PRNGKey(seed)

    def compile_pipeline(self, text_instructions: list) -> np.ndarray:
        """Convert textual instructions into a float64 matrix.

        Args:
            text_instructions: a list of strings or tuples. A tuple is
                ``(name, p1, p2)``; a plain string uses the operation
                defaults.

        Returns:
            An array of shape ``(N, 4)`` with rows ``[cmd_id, p1, p2, 0.0]``.
        """
        compiled = []
        for i, cmd in enumerate(text_instructions):
            if isinstance(cmd, tuple):
                name = str(cmd[0]).lower().strip()
                try:
                    p1 = float(cmd[1]) if len(cmd) > 1 else 0.0
                    p2 = float(cmd[2]) if len(cmd) > 2 else 0.0
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"instruction {i} {cmd!r}: parameters must be numbers"
                    ) from exc
            else:
                name = str(cmd).lower().strip()
                if name == "dropout":
                    p1, p2 = 0.8, 0.0
                elif name == "clip":
                    p1, p2 = -1.0, 1.0
                elif name == "scale":
                    p1, p2 = float(_PHI), float(_ALPHA)
                else:
                    p1, p2 = 0.0, 0.0

            cmd_id = self.cmd_map.get(name, 0)
            compiled.append([float(cmd_id), p1, p2, 0.0])

        return np.array(compiled, dtype=np.float64)

    def run_dynamic_pipeline(
        self,
        input_data: np.ndarray,
        compiled_ops: np.ndarray,
    ) -> np.ndarray:
        """Run the compiled pipeline inside a single JIT scan.

        Args:
            input_data: the input array.
            compiled_ops: the compiled operation matrix.

        Returns:
            The output of the pipeline as a NumPy array.
        """
        res_data, updated_key = _run_pipeline_jit(
            jnp.array(input_data, dtype=jnp.float64),
            jnp.array(compiled_ops, dtype=jnp.float64),
            self.base_key,
        )
        self.base_key = updated_key
        return np.array(res_data)

    def run_pipeline_with_chunking(
        self,
        input_data: np.ndarray,
        compiled_ops: np.ndarray,
        chunk_size: int = 500,
    ) -> np.ndarray:
        """Run the pipeline in fixed-size chunks inside the JIT kernel.

        Args:
            input_data: the input array.
            compiled_ops: the compiled operation matrix.
            chunk_size: number of instructions per chunk.

        Returns:
            The output of the pipeline as a NumPy array.
        """
        n_ops = len(compiled_ops)
        n_chunks = (n_ops + chunk_size - 1) // chunk_size
        total_slots = n_chunks * chunk_size

        padded_ops = np.zeros((total_slots, 4), dtype=np.float64)
        padded_ops[:n_ops] = compiled_ops

        structured_chunks = padded_ops.reshape(n_chunks, chunk_size, 4)

        j_data = jnp.array(input_data, dtype=jnp.float64)
        j_chunks = jnp.array(structured_chunks, dtype=jnp.float64)

        res_data, self.base_key = _run_pipeline_chunked_jit(j_data, j_chunks, self.base_key)
        return np.array(res_data)

    def compute_gradients(
        self,
        input_data: np.ndarray,
        compiled_ops: np.ndarray,
    ) -> np.ndarray:
        """Return the AD gradient of the loss with respect to the input.

        The loss is normalized by the fixed constant ``_PHI_FOUR``.

        Args:
            input_data: the input array.
            compiled_ops: the compiled operation matrix.

        Returns:
            The gradient, same shape as ``input_data``.
        """
        j_input = jnp.array(input_data, dtype=jnp.float64)
        j_ops = jnp.array(compiled_ops, dtype=jnp.float64)

        grads = _grad_engine(j_input, j_ops, self.base_key)
        self.base_key = jax.random.split(self.base_key)[0]
        return np.array(grads)

    def save_compiled_pipeline(
        self,
        compiled_ops: np.ndarray,
        filename: str = "compiled_recipe.npy",
    ):
        """Save the compiled operation matrix to a ``.npy`` file.

        Args:
            compiled_ops: the operation matrix.
            filename: destination path.
        """
        np.save(filename, compiled_ops)
        logger.info("Advanced pipeline saved to %r", filename)

    def load_compiled_pipeline(
        self,
        filename: str = "compiled_recipe.npy",
    ) -> np.ndarray:
        """Load a previously saved operation matrix.

        Args:
            filename: source path.

        Returns:
            The loaded operation matrix.

        Raises:
            FileNotFoundError: if ``filename`` does not exist.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"pipeline file not found: {filename!r}")
        ops = np.load(filename)
        logger.info("Advanced pipeline loaded from %r, shape=%s", filename, ops.shape)
        return ops
