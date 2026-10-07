import sys, os
import pytest, numpy as np, jax, jax.numpy as jnp
from dense_armor.core.engine import AdaptiveSignalStabilizer

jax.config.update("jax_enable_x64", True)
H, W = 32, 32
x_star = jnp.ones((H, W), dtype=jnp.float64)

def run_filtered_scan(attack_seq):
    params = {
        "static_threshold": 0.10, "initial_damping": 0.1, "alpha": 0.25,
        "anomaly_sigma_mult": 2.0, "k_anom_min": 0.20, "k_anom_max": 0.80,
        "window_radius": 1, "smooth_l2_blend": 0.5
    }
    stabilizer = AdaptiveSignalStabilizer(**params)
    corrupted_batch = x_star[None, :, :] + attack_seq
    filtered_batch = stabilizer.filter_batch_scenarios(corrupted_batch)
    return jnp.mean((filtered_batch - x_star[None, :, :]) ** 2, axis=(1, 2))

# n_iter=100, n_restarts=10 -> 1000 passi totali effettivi.
def build_pgd_sequence(key, eps=0.30, alpha_step=0.04, n_iter=100, n_restarts=10):
    def pgd_restart(carry, key_i):
        delta0 = jax.random.uniform(key_i, (H, W), minval=-eps, maxval=eps, dtype=jnp.float64)
        def pgd_step(delta, _):
            g = jax.grad(lambda d: -jnp.mean(jnp.sin(jnp.pi * (x_star + d) / eps) ** 2))(delta)
            delta_new = jnp.clip(delta + alpha_step * jnp.sign(g + 1e-12), -eps, eps)
            return delta_new, delta_new
        _, traj = jax.lax.scan(pgd_step, delta0, None, length=n_iter)
        return carry, traj
    keys = jax.random.split(key, n_restarts)
    _, all_traj = jax.lax.scan(pgd_restart, None, keys)
    return all_traj.reshape(n_restarts * n_iter, H, W)

def build_bim_sequence(eps=0.25, alpha_step=0.03, n_steps=1000):
    def bim_step(delta, _):
        g = jax.grad(lambda d: -jnp.mean((x_star + d) ** 2))(delta)
        delta_new = jnp.clip(delta + alpha_step * jnp.sign(g + 1e-12), -eps, eps)
        return delta_new, delta_new
    _, traj = jax.lax.scan(bim_step, jnp.zeros((H, W), dtype=jnp.float64), None, length=n_steps)
    return traj

def build_mifgsm_sequence(eps=0.28, alpha_step=0.035, mu=0.9, n_steps=1000):
    def mifgsm_step(carry, _):
        delta, momentum = carry
        g = jax.grad(lambda d: -jnp.mean((x_star + d) ** 2))(delta)
        g_norm = g / (jnp.mean(jnp.abs(g)) + 1e-12)
        momentum_new = mu * momentum + g_norm
        delta_new = jnp.clip(delta + alpha_step * jnp.sign(momentum_new + 1e-12), -eps, eps)
        return (delta_new, momentum_new), delta_new
    init = (jnp.zeros((H, W), dtype=jnp.float64), jnp.zeros((H, W), dtype=jnp.float64))
    _, traj = jax.lax.scan(mifgsm_step, init, None, length=n_steps)
    return traj

def test_gradient_based_attacks():
    """PGD, BIM, MI-FGSM: 1000 passi ciascuno contro AdaptiveSignalStabilizer."""
    key = jax.random.PRNGKey(42)
    print("\nGenerazione sequenze avversariali basate su gradiente...")
    pgd_seq = build_pgd_sequence(key)
    bim_seq = build_bim_sequence()
    mifgsm_seq = build_mifgsm_sequence()

    v_pgd = run_filtered_scan(pgd_seq)
    v_bim = run_filtered_scan(bim_seq)
    v_mifgsm = run_filtered_scan(mifgsm_seq)

    print("\nTraiettoria dell'errore (V = MSE rispetto al riferimento):")
    for label, v in (("PGD", v_pgd), ("BIM", v_bim), ("MI-FGSM", v_mifgsm)):
        print(f"  {label}:")
        for step in range(0, 1000, 200):
            print(f"    passo {step:04d} | V = {float(v[step]):.5f}")

    print("\nSintesi:")
    print(f"  PGD      -> V_max: {float(jnp.max(v_pgd)):.4f} | V_finale: {float(v_pgd[-1]):.4f}")
    print(f"  BIM      -> V_max: {float(jnp.max(v_bim)):.4f} | V_finale: {float(v_bim[-1]):.4f}")
    print(f"  MI-FGSM  -> V_max: {float(jnp.max(v_mifgsm)):.4f} | V_finale: {float(v_mifgsm[-1]):.4f}")

    assert jnp.max(v_pgd) < 1.20, "Divergenza rilevata sotto attacco PGD"

if __name__ == "__main__":
    test_gradient_based_attacks()
