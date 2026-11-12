"""Blend of a raw signal with a reference signal.

``apply_damping_blend(raw, reference)`` returns ``(1 - K) * raw + K *
reference`` for every element. The weight ``K`` of the reference depends on
the distance ``|raw - reference|`` and is kept between ``ALPHA - SIGMA``
(about 0.104) and ``1 - (ALPHA - SIGMA)`` (about 0.896).

In practice ``K`` rises from about 0.25 at distance 0 to its upper bound
within a distance of about 0.05 (for a raw value of 1) and stays there: a
reference that differs from the raw signal by more than that is followed at
about 90 %. The tests in ``test/core/test_damping_operator.py`` measure this.

The constants (``PHI``, ``ALPHA``, ``SIGMA``, ``K_MAX_ZETA``) are fixed
numerical hyper-parameters. This repository gives no derivation for them.
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
    """Return ``(1 - K) * gradient_state + K * noise_matrix`` for every element.

    The names are historical: ``gradient_state`` is the raw signal and
    ``noise_matrix`` is the reference. ``K`` is bounded and grows with the
    distance between the two (see the module docstring for its range). A NaN
    in either input is replaced by the other input before the blend.

    Args:
        gradient_state: the raw signal.
        noise_matrix: the reference signal.

    Returns:
        The blended signal, with no NaN.
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
