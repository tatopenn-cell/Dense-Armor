"""Parallel Monte Carlo simulation and combinatorial bit-mask manipulation.

:class:`ParametricScenarioSimulator` runs massive Monte Carlo scenarios with
JAX ``vmap`` and a stochastic decision collapse. :class:`BitwisePermutationEngine`
swaps elements of a combinatorial vector through bit masks.

The ``_parallel_engine`` is built **once** at module import, not inside the
method: when ``jax.jit(jax.vmap(...))`` is recreated at every call, the JAX
compilation cache can never hit (a new Python closure has a different
identity at every call) and every invocation recompiles the XLA graph from
scratch. The measured cost is dominated by that recompilation (~130 ms per
call), not by the ``np.array`` cast. ``base_state`` is a traced argument
(not static), so different values do not force a new compilation.

``collapse_decision`` does not modify the caller array in place: it works
on an internal copy and returns ``(result, collapsed_vector)``. The caller
can ignore the collapsed vector when it is not needed.
"""

import jax
import jax.numpy as jnp
import numpy as np


class BitwisePermutationEngine:
    """Swap elements of a combinatorial vector (space 2^n) through bit masks.

    Elements are swapped in a multidimensional array according to
    target/control bit pairs.

    Args:
        n_elements: number of bits of the combinatorial vector, so the
            addressable space has ``2 ** n_elements`` states.
    """

    def __init__(self, n_elements: int):
        """Store the number of bits and the derived state count."""
        self.n = n_elements
        self.size = 1 << n_elements

    def apply_bitwise_swap(
        self,
        data: np.ndarray,
        target_bit: int,
        control_bit: int,
    ) -> np.ndarray:
        """Permute the vector elements according to binary masks.

        Args:
            data: the input vector.
            target_bit: the bit whose pairs are swapped.
            control_bit: the bit that gates the swap.

        Returns:
            The permuted vector, with the same shape as ``data``.
        """
        output = data.copy()
        t_stride = 1 << (self.n - 1 - target_bit)
        c_stride = 1 << (self.n - 1 - control_bit)

        for i in range(self.size):
            if (i & c_stride) and not (i & t_stride):
                idx_0 = i
                idx_1 = i + t_stride
                output[idx_0], output[idx_1] = data[idx_1], data[idx_0]
        return output


def _scenario_step(carry: jnp.ndarray, single_param: jnp.ndarray) -> tuple:
    """Scalar time step for ``jax.lax.scan``."""
    current_state = carry
    next_state = current_state * 0.95 + single_param * 0.05
    return next_state, next_state


def _simulate_single_instance(
    base_state: jnp.ndarray, params_vector: jnp.ndarray
) -> jnp.ndarray:
    """Single scenario (T_steps parameters) advanced through time."""
    scan_out: tuple = jax.lax.scan(_scenario_step, base_state, params_vector)
    history: jnp.ndarray = scan_out[1]
    return history


_parallel_engine = jax.jit(jax.vmap(_simulate_single_instance, in_axes=(None, 0)))


class ParametricScenarioSimulator:
    """Massive parallel Monte Carlo simulation with stochastic collapse.

    Args:
        base_state: initial scalar state shared by every scenario.
    """

    _simulation_step = staticmethod(_scenario_step)

    def run_parallel_scenarios(
        self,
        base_state: float,
        parameters_batch: np.ndarray,
    ) -> np.ndarray:
        """Run every batch in parallel through ``jax.lax.scan`` on time.

        Args:
            base_state: initial scalar state for all scenarios.
            parameters_batch: array of shape ``(N_scenarios, T_steps)``.

        Returns:
            An array of shape ``(N_scenarios, T_steps)``.
        """
        result = _parallel_engine(
            jnp.float64(base_state), jnp.asarray(parameters_batch)
        )
        return np.array(result)

    def collapse_decision(
        self,
        distribution_vector: np.ndarray,
        target_idx: int,
    ) -> tuple:
        """Stochastic decision collapse conditioned on the input distribution.

        The input array is not modified in place: the method works on an
        internal copy.

        Args:
            distribution_vector: the probability-like vector to collapse.
            target_idx: the index that splits the two branches.

        Returns:
            A tuple ``(result, collapsed_vector)`` where ``result`` is 0 or
            1 and ``collapsed_vector`` is the normalized vector after the
            collapse.

        Raises:
            RuntimeError: when the input vector has zero energy.
        """
        vec = np.array(distribution_vector, copy=True, dtype=np.float64)

        prob_0 = np.sum(np.abs(vec[:target_idx]))
        prob_1 = np.sum(np.abs(vec[target_idx:]))
        total = prob_0 + prob_1

        if total < 1e-12:
            raise RuntimeError(
                "Zero-energy decision vector: cannot compute the choice."
            )

        prob_0 /= total
        prob_1 /= total

        result = int(np.random.choice([0, 1], p=[prob_0, prob_1]))
        zero_slot = 1 - result

        if zero_slot == 0:
            vec[:target_idx] = 0.0
        else:
            vec[target_idx:] = 0.0

        new_total = np.sum(np.abs(vec))
        if new_total > 0:
            vec /= new_total

        return result, vec
