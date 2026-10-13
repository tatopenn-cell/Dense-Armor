r"""Adaptive damping blend between a raw signal and a reference.

The blend uses a non-linear gain that depends on the distance between the
two inputs. A small distance favors the raw signal; a large distance favors
the reference, with a damped gain that stays bounded.

The constants (``PHI``, ``ZETA``, ...) are fixed numerical hyper-parameters,
not quantum states or operators. No complex arithmetic and no Hilbert space
are involved.
"""

import jax
import jax.numpy as jnp
import numpy as np

_PHI_128 = np.longdouble(1.0 + np.sqrt(5.0)) / 2.0
_ZETA_TOTAL_128 = np.longdouble(60.16762236065194)

_PHI = float(_PHI_128)
_ALPHA = 0.25
_SIGMA = float(np.longdouble(1.0) / (_PHI_128**4))
_K_MAX_ZETA = float(_ZETA_TOTAL_128 / (_PHI_128**4))


@jax.jit
def apply_damping_blend(
    gradient_state: jnp.ndarray, noise_matrix: jnp.ndarray
) -> jnp.ndarray:
    r"""Blend ``gradient_state`` (raw) and ``noise_matrix`` (reference).

    The blend coefficient ``K`` grows with the distance between the two
    signals: a small distance gives low ``K`` (favor the raw signal), a
    large distance gives high ``K`` (favor the reference, damped). NaN
    inputs are replaced before the blend.

    Args:
        gradient_state: the raw signal.
        noise_matrix: the reference signal.

    Returns:
        The blended signal, with no NaN in the output.
    """
    eps = 1e-7

    is_nan_gradient = jnp.isnan(gradient_state)
    is_nan_noise = jnp.isnan(noise_matrix)

    gradient_state_safe = jnp.where(is_nan_gradient, noise_matrix, gradient_state)
    noise_matrix_safe = jnp.where(is_nan_noise, gradient_state_safe, noise_matrix)

    diff = jnp.abs(gradient_state_safe - noise_matrix_safe)

    coherence = jnp.clip(
        1.0 - (diff / (2.0 * jnp.maximum(jnp.abs(gradient_state_safe), 1e-3))),
        0.0,
        1.0,
    )

    x_steer = jnp.clip(diff, 0.0, 5.0)
    steering = 1.0 / (1.0 + jnp.exp(-(x_steer - 2.0)))
    h_damping = steering * jnp.float32(_K_MAX_ZETA)

    c_anom = jnp.float32(1.0 / (_PHI**2))
    K_anomalous_ratio = diff / (c_anom + diff + eps)
    K_anomalous_raw = jnp.float32(_ALPHA) + K_anomalous_ratio * (
        jnp.float32(_K_MAX_ZETA) - jnp.float32(_ALPHA)
    )
    K_anomalous = jnp.clip(
        K_anomalous_raw, jnp.float32(_ALPHA), jnp.float32(_K_MAX_ZETA)
    )

    K_operator = (1.0 - coherence) * h_damping + coherence * K_anomalous

    K_operator = jnp.where(
        jnp.logical_or(is_nan_gradient, is_nan_noise),
        jnp.float32(1.0 - (_ALPHA - _SIGMA)),
        K_operator,
    )

    k_min_bound = jnp.float32(_ALPHA - _SIGMA)
    k_max_bound = jnp.float32(1.0 - (_ALPHA - _SIGMA))
    K_operator = jnp.clip(K_operator, k_min_bound, k_max_bound)

    blended = (1.0 - K_operator) * gradient_state_safe + K_operator * noise_matrix_safe

    blended = jnp.where(jnp.isnan(blended), noise_matrix_safe, blended)

    return blended
