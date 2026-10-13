"""Block chunking for data batches and compiled operation lists.

Chunking keeps the number of items per JIT call constant, so XLA does not
recompile when the batch or the operation list length changes.
"""

import jax
import jax.numpy as jnp
import numpy as np


class ImageChunker:
    """Split and rebuild data batches and compiled operation lists in fixed-size blocks.

    Chunking is useful when a batch is too large to fit in memory in one
    allocation, or when a compiled operation list is long enough that
    changing its length would force XLA to recompile at every call.

    Args:
        chunk_size: fixed size of every block, for both batches and
            operation lists.
    """

    def __init__(self, chunk_size: int = 128) -> None:
        """Initialize the chunker with a fixed block size."""
        self.chunk_size = int(chunk_size)

    def split_array(self, array: np.ndarray) -> list:
        """Split a multidimensional array into a list of sub-chunks.

        Args:
            array: the input array; chunking runs on the first axis.

        Returns:
            A list of arrays, each with ``chunk_size`` rows except the last.
        """
        total_samples = array.shape[0]
        num_chunks = int(np.ceil(total_samples / self.chunk_size))
        chunks = []
        for b in range(num_chunks):
            start_idx = b * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, total_samples)
            chunks.append(array[start_idx:end_idx])
        return chunks

    def merge_chunks(self, chunks_list: list) -> np.ndarray:
        """Rebuild a single array from a list of sub-chunks.

        Flat 1D chunks are concatenated; 2D and higher chunks are stacked
        on the first axis with ``np.vstack``.

        Args:
            chunks_list: the list produced by :meth:`split_array`.

        Returns:
            The rebuilt array, or an empty float32 array when the input
            list is empty.
        """
        if not chunks_list:
            return np.array([], dtype=np.float32)
        if chunks_list[0].ndim == 1:
            return np.concatenate(chunks_list)
        return np.vstack(chunks_list)

    def execute_pipeline_chunked(
        self, codegen_engine, input_vector: np.ndarray, compiled_ops: list
    ) -> np.ndarray:
        """Run the compiled operations in fixed-size blocks of ``chunk_size``.

        Splitting the operation list into fixed blocks prevents XLA from
        recompiling the pipeline when the number of operations changes.
        Each block is delegated to
        ``codegen_engine.run_pipeline_with_chunking``.

        Args:
            codegen_engine: an object exposing ``run_pipeline_with_chunking``.
            input_vector: the input array.
            compiled_ops: the list of compiled operations.

        Returns:
            The output array, in float64.
        """
        output = jnp.array(input_vector, dtype=jnp.float64)

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
        """Replace the -1.0 markers of an operation template with dynamic parameters.

        Every element of ``template_ops`` different from -1.0 is repeated in
        the four columns of the patched operation. Every -1.0 marker is
        replaced, in order, with the next value of ``dynamic_parameters``.
        Implemented with ``jax.lax.scan`` to stay JIT-compatible.

        Args:
            template_ops: the operation template with -1.0 markers.
            dynamic_parameters: the parameter values, in order.

        Returns:
            The patched operation array.
        """

        def patch_single_op(carry: jnp.ndarray, op: jnp.ndarray) -> tuple:
            """Scan step: replace a -1.0 marker with the next dynamic parameter."""
            idx = carry
            is_parametric = op == -1.0
            final_param = jnp.where(is_parametric, dynamic_parameters[idx], op)
            next_idx = jnp.where(is_parametric, idx + jnp.int32(1), idx)
            patched_op = jnp.array([op, op, op, final_param], dtype=jnp.float64)
            return next_idx, patched_op

        scan_out: tuple = jax.lax.scan(patch_single_op, jnp.int32(0), template_ops)
        patched_compiled_ops: jnp.ndarray = scan_out[1]
        return patched_compiled_ops
