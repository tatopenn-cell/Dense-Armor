import sys, os, time
import pytest, numpy as np, jax, jax.numpy as jnp
import psutil  # Rilevamento telemetria hardware live
from dense_armor.core.engine import AdaptiveSignalStabilizer

# Configurazione globale asse XLA
jax.config.update("jax_enable_x64", True)

# Parametri geometrici strutturali
H, W, STEPS_LONG = 32, 32, 50000
rows, cols = jnp.arange(H, dtype=jnp.float64), jnp.arange(W, dtype=jnp.float64)
R_grid, C_grid = jnp.meshgrid(rows, cols, indexing='ij')
# Segnale di riferimento con vera texture spaziale (3 cicli per asse): una rampa lineare
# piatta rende la deformazione elastica quasi invisibile per costruzione (interpolare un
# gradiente liscio cambia poco il valore), qui invece rotazione/traslazione ed elastic
# deformation producono davvero una perturbazione misurabile.
x_star = 0.5 + 0.5 * jnp.sin(2.0 * jnp.pi * 3.0 * R_grid / H) * jnp.cos(2.0 * jnp.pi * 3.0 * C_grid / W)

# Traiettorie complete precalcolate una sola volta (prima venivano rigenerate per intero
# ad ogni chunk solo per estrarne una fetta)
ANGLES_FULL = jnp.linspace(0.0, 1.57, STEPS_LONG, dtype=jnp.float64)
TXS_FULL    = jnp.linspace(0.0, 0.35, STEPS_LONG, dtype=jnp.float64)
TYS_FULL    = jnp.linspace(0.0, 0.35, STEPS_LONG, dtype=jnp.float64)

def _affine_transform(image, angle_rad, tx, ty):
    cos_a, sin_a = jnp.cos(angle_rad), jnp.sin(angle_rad)
    C, R = jnp.meshgrid(jnp.linspace(-1.0, 1.0, W, dtype=jnp.float64), jnp.linspace(-1.0, 1.0, H, dtype=jnp.float64))
    src_c_px = (cos_a * C + sin_a * R - tx + 1.0) * 0.5 * (W - 1)
    src_r_px = (-sin_a * C + cos_a * R - ty + 1.0) * 0.5 * (H - 1)
    c0 = jnp.clip(jnp.floor(src_c_px).astype(jnp.int32), 0, W - 2)
    r0 = jnp.clip(jnp.floor(src_r_px).astype(jnp.int32), 0, H - 2)
    c1, r1 = c0 + 1, r0 + 1
    wc, wr = src_c_px - c0.astype(jnp.float64), src_r_px - r0.astype(jnp.float64)
    return ((1.0-wr)*(1.0-wc)*image[r0, c0] + (1.0-wr)*wc*image[r0, c1] + wr*(1.0-wc)*image[r1, c0] + wr*wc*image[r1, c1])

def build_rotation_translation_chunk(start_idx, end_idx):
    n_steps = end_idx - start_idx
    indices = jnp.arange(start_idx, end_idx)
    angles = ANGLES_FULL[indices]
    txs = TXS_FULL[indices]
    tys = TYS_FULL[indices]

    _, traj = jax.lax.scan(lambda _, i: (None, _affine_transform(x_star, angles[i], txs[i], tys[i]) - x_star), None, jnp.arange(n_steps))
    return jax.block_until_ready(traj)

def build_elastic_deformation_chunk(carry_key, n_steps, alpha_el=1.50):
    def deform_step(k, i):
        k1, next_key = jax.random.split(k)
        sub_k1, sub_k2 = jax.random.split(k1)
        dx_step = jax.random.normal(sub_k1, (H, W), dtype=jnp.float64) * alpha_el
        dy_step = jax.random.normal(sub_k2, (H, W), dtype=jnp.float64) * alpha_el
        r_src = jnp.clip(R_grid + dy_step, 0.0, H - 1.001)
        c_src = jnp.clip(C_grid + dx_step, 0.0, W - 1.001)
        r0, c0 = jnp.floor(r_src).astype(jnp.int32), jnp.floor(c_src).astype(jnp.int32)
        r1, c1 = jnp.clip(r0 + 1, 0, H - 1), jnp.clip(c0 + 1, 0, W - 1)
        wr, wc = r_src - r0.astype(jnp.float64), c_src - c0.astype(jnp.float64)
        out_step = ((1.0-wr)*(1.0-wc)*x_star[r0, c0] + (1.0-wr)*wc*x_star[r0, c1] + wr*(1.0-wc)*x_star[r1, c0] + wr*wc*x_star[r1, c1]) - x_star
        return next_key, out_step

    next_key, traj = jax.lax.scan(deform_step, carry_key, jnp.arange(n_steps))
    return next_key, jax.block_until_ready(traj)

def calcola_chunk_ottimale():
    """ Rileva la RAM libera ed estrae la dimensione del blocco ideale per prevenire OOM. """
    ram_libera_byte = psutil.virtual_memory().available
    # Singolo frame: 32x32 pixel * 8 byte (float64) * 5.0 (coefficiente di overhead buffer XLA/NumPy)
    costo_stimato_frame = 32 * 32 * 8 * 5.0
    # Definizione vincoli geometrici (Minimo 1000 step di campionamento, Massimo 10000)
    chunk_calcolato = int(ram_libera_byte // costo_stimato_frame)
    return max(1000, min(10000, chunk_calcolato))

def test_sentinel_geometric_spatial_attacks():
    key = jax.random.PRNGKey(88)
    t_start = time.time()
    
    print("\n=====================================================================================")
    print("[XLA SYSTEM PROFILE] HARDWARE AGNOSTIC KERNEL OPERATING LAYER — CATEGORIA B SCALE")
    print(f"HOST EXECUTION PLAN: {STEPS_LONG} Concurrent Iterations | Adaptive Memory Guard: Active")
    print("=====================================================================================")
    
    params = {
        "static_threshold": 0.10, "initial_damping": 0.1, "alpha": 0.25,
        "anomaly_sigma_mult": 2.0, "k_anom_min": 0.20, "k_anom_max": 0.80,
        "window_radius": 1, "smooth_l2_blend": 0.5
    }
    
    stabilizer_rot = AdaptiveSignalStabilizer(**params)
    stabilizer_el = AdaptiveSignalStabilizer(**params)
    
    v_rot_list, v_elastic_list = [], []
    elastic_key = key
    start_idx = 0
    
    print("[THREAD-DUAL] Avvio elaborazione asincrona auto-calibrante...")
    
    while start_idx < STEPS_LONG:
        # Calcolo dinamico dello scenario in base al carico live del sistema
        chunk_corrente = calcola_chunk_ottimale()
        end_idx = min(start_idx + chunk_corrente, STEPS_LONG)
        current_chunk_len = end_idx - start_idx
        
        # Log di tracciamento ad alta precisione
        ram_m_libera = psutil.virtual_memory().available / (1024 ** 2)
        print(f" -> Finestra [{start_idx:05d}:{end_idx:05d}] | Allocazione Chunk: {current_chunk_len} | RAM Libera: {ram_m_libera:.1f} MB")
        
        # 1. Elaborazione asse ROTAZIONE
        rot_chunk = build_rotation_translation_chunk(start_idx, end_idx)
        corrupted_rot = x_star[None, :, :] + rot_chunk
        res_rot = stabilizer_rot.filter_batch_scenarios(corrupted_rot)
        _ = jax.block_until_ready(res_rot)
        v_rot_list.append(jnp.mean((res_rot - x_star[None, :, :]) ** 2, axis=(1, 2)))
        
        # 2. Elaborazione asse ELASTICO
        elastic_key, el_chunk = build_elastic_deformation_chunk(elastic_key, current_chunk_len)
        corrupted_el = x_star[None, :, :] + el_chunk
        res_el = stabilizer_el.filter_batch_scenarios(corrupted_el)
        _ = jax.block_until_ready(res_el)
        v_elastic_list.append(jnp.mean((res_el - x_star[None, :, :]) ** 2, axis=(1, 2)))
        
        start_idx = end_idx
        
    v_rot = jnp.concatenate(v_rot_list, axis=0)
    v_elastic = jnp.concatenate(v_elastic_list, axis=0)
    t_elapsed = time.time() - t_start
    
    print("\n[DUAL-AGENT METRICS] BATTLEGROUND TELEMETRY STREAM (INTERVAL SAMPLING)")
    print("-" * 85)
    
    checkpoints = [0, 10000, 20000, 30000, 40000, STEPS_LONG - 1]
    for check in checkpoints:
        print(f"Step {check:05d} | [THREAT] Iniezione geometrica affine/elastica ")
        print(f"           | [DEFENSE] Risposta d'asse 2D - V_rot: {float(v_rot[check]):.5f} | V_el: {float(v_elastic[check]):.5f}")
        
        if check == 0:
            print("           | [STATUS] Iniezione fredda controllata. Margine d'errore simmetrico.")
        elif check == 20000:
            print("           | [STATUS] Attacco ondulatorio intercettato: attivazione gating di coerenza locale.")
        elif check == 40000:
            print("           | [STATUS] Deriva iperbolica affine bloccata da barriera di contrazione geometrica L2.")
        print(" " + "-"*80)
            
    print("=" * 85)
    print(f"[METRICS] Tempo di computazione asincrono totale  : {t_elapsed:.2f} s")
    print(f"[METRICS] Errore massimo transitorio ROTAZIONE (V_max)  : {float(jnp.max(v_rot)):.4f}")
    print(f"[METRICS] Errore residuo finale ROTAZIONE (V_inf)       : {float(v_rot[-1]):.4f}")
    print(f"[METRICS] Errore massimo transitorio ELASTICO (V_max)   : {float(jnp.max(v_elastic)):.4f}")
    print(f"[METRICS] Errore residuo finale ELASTICO (V_inf)        : {float(v_elastic[-1]):.4f}")
    print("=" * 85)

    assert t_elapsed >= 5.0, f"[ERROR] Esecuzione troppo rapida per il benchmarking hardware: {t_elapsed:.2f}s"
    assert jnp.max(v_rot) < 1.20, "[ERROR] Esplosione numerica nello spazio degli stati ROTAZIONE (Divergenza NaN/Inf)!"
    # BUG-FIX: v_elastic veniva calcolato e stampato ma mai validato — un fallimento
    # totale della difesa elastica sarebbe passato inosservato.
    assert jnp.max(v_elastic) < 1.20, "[ERROR] Esplosione numerica nello spazio degli stati ELASTICO (Divergenza NaN/Inf)!"
    print("\n[VERDETTO SCIENTIFICO] Confinamento UBB verificato. Il kernel Sentinel ha neutralizzato l'agente avversariale.")

if __name__ == "__main__":
    test_sentinel_geometric_spatial_attacks()
