import jax
import jax.numpy as jnp

# Versione ottimizzata ad altissima densità matematica (XLA-Fused)
@jax.jit
def curvature(x_current: jnp.ndarray, x_reference: jnp.ndarray, delta: float = 1e-6) -> jnp.ndarray:
    """
    Calcola la curvatura geometrica (κ) calcolando direttamente il gradiente analitico.
    Elimina l'overhead di tracciamento di jax.grad mantenendo l'invarianza numerica.
    """
    # Derivata analitica esplicita di sum(square(x - ref)) -> 2 * (x - ref)
    dy_dx = 2.0 * (x_current - x_reference)
    
    # Calcolo geometrico dello stress spettrale
    dy_dx_sq = jnp.sum(jnp.square(dy_dx))
    
    numerator = jnp.sqrt(dy_dx_sq + delta)
    denominator = jnp.sqrt(1.0 + dy_dx_sq + delta)
    
    return numerator / denominator

