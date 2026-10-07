import sys, os, time
import pytest, numpy as np, jax, jax.numpy as jnp
import psutil  # Telemetria hardware live della RAM
from dense_armor.core.engine import AdaptiveSignalStabilizer

# Forzatura float64 su asse XLA
jax.config.update("jax_enable_x64", True)

# Parametri strutturali del reticolo
H, W, STEPS_LONG = 32, 32, 50000
rows, cols = jnp.arange(H, dtype=jnp.float64), jnp.arange(W, dtype=jnp.float64)
R_grid, C_grid = jnp.meshgrid(rows, cols, indexing='ij')
# Stesso fix di test_boundB.py: una rampa lineare piatta rende l'anomaly detection dello
# stabilizzatore artificiosamente facile (varianza locale nulla in assenza di attacco).
x_star = 0.5 + 0.5 * jnp.sin(2.0 * jnp.pi * 3.0 * R_grid / H) * jnp.cos(2.0 * jnp.pi * 3.0 * C_grid / W)

# BUG-FIX: le tre funzioni sotto reinizializzavano lo stato dell'ottimizzatore a zero ad
# ogni chunk, quindi l'attacco ripartiva da capo ogni 10.000 passi invece di proseguire
# con continuità per i 50.000 passi dichiarati. Ora ricevono e restituiscono lo stato
# (carry_state) da incatenare tra un chunk e il successivo, come già fa build_elastic_
# deformation_chunk in test_boundB.py.

# Generazione a blocchi: asse C&W L2
# BUG-FIX (root cause vera, trovata isolando l'attacco senza scudo): `loss` veniva
# calcolata usando `w` (variabile esterna), poi differenziata rispetto all'argomento
# della lambda `wv` MAI USATO nel corpo -> jax.grad restituiva un gradiente
# identicamente zero ad ogni passo. Con gradiente zero il momentum resta a zero e
# delta non si muove mai dal suo valore iniziale (0.000000 esatto, verificato).
# Ora la loss e' una vera funzione dell'argomento differenziato, e il coefficiente
# del termine adversariale (attack_strength) e' sempre >= 0 cosi' la discesa del
# gradiente spinge davvero la perturbazione a crescere invece di annullarla.
def build_cw_l2_chunk(start_idx, end_idx, carry_state=None, lr=0.15, c_const=5.0):
    def cw_step(carry, step_idx):
        w, vel = carry
        attack_strength = 0.5 + 0.5 * jnp.sin(0.01 * step_idx)  # sempre in [0, 1]

        def loss_fn(wv):
            delta = jnp.tanh(wv) * 0.5
            return -c_const * jnp.mean(delta ** 2) * attack_strength + jnp.mean(delta ** 2)

        g = jax.grad(loss_fn)(w)
        vel_new = 0.9 * vel - lr * g
        return (w + vel_new, vel_new), jnp.tanh(w + vel_new) * 0.5

    # BUG-FIX #2: partire da delta ESATTAMENTE zero e' un punto di equilibrio instabile
    # (la loss e' una funzione pari di delta -> gradiente nullo proprio in quel punto,
    # verificato isolando l'attacco: restava a 0.000000 anche col gradiente corretto).
    # Un piccolo innesco deterministico non-zero rompe la simmetria e permette alla
    # discesa del gradiente di muoversi davvero.
    init = carry_state if carry_state is not None else (jnp.full((H, W), 1e-3, dtype=jnp.float64), jnp.zeros((H, W), dtype=jnp.float64))
    final_state, traj = jax.lax.scan(cw_step, init, jnp.arange(start_idx, end_idx))
    return jax.block_until_ready(traj), final_state

# Generazione a blocchi: asse C&W Linf
# Stesso BUG-FIX: `jax.grad(lambda d: loss)(delta)` differenziava una lambda il cui
# argomento `d` non compariva nel corpo (usava `loss`, gia' calcolata con `delta`
# esterna) -> gradiente zero, delta bloccata a 0 per l'intero run.
def build_cw_linf_chunk(start_idx, end_idx, carry_state=None, lr=0.12, eps=0.65, c_const=3.0):
    def step(carry, step_idx):
        delta, vel = carry
        attack_strength = 0.5 + 0.5 * jnp.cos(0.01 * step_idx)  # sempre in [0, 1]

        # BUG-FIX #3: con coefficiente simmetrico (0.5 - attack_strength, che oscilla
        # in modo bilanciato tra positivo e negativo) l'integrale su un ciclo e' ~0 e
        # il momentum si annulla in media -> delta resta incollata al seed iniziale
        # (verificato: mean|delta| restava a ~0.001 anche dopo 5000 passi). Serve la
        # stessa asimmetria gia' usata in C&W L2 (c_const) perche' l'escursione
        # negativa domini su quella positiva e il momentum accumuli una spinta netta.
        def loss_fn(d):
            return -c_const * jnp.mean(d ** 2) * attack_strength + 0.5 * jnp.mean(d ** 2)

        g = jax.grad(loss_fn)(delta)
        vel_new = 0.9 * vel - lr * g
        delta_new = jnp.clip(delta + vel_new, -eps, eps)
        return (delta_new, vel_new), delta_new

    # BUG-FIX #2: stesso punto di equilibrio instabile in delta=0, stesso innesco.
    init = carry_state if carry_state is not None else (jnp.full((H, W), 1e-3, dtype=jnp.float64), jnp.zeros((H, W), dtype=jnp.float64))
    final_state, traj = jax.lax.scan(step, init, jnp.arange(start_idx, end_idx))
    return jax.block_until_ready(traj), final_state

# Generazione a blocchi: asse DeepFool
# BUG-FIX: g_norm_sq era quasi zero (gradiente di una funzione quasi piatta su un
# segnale normalizzato), quindi r_i = |f_val|/g_norm_sq esplodeva e saturava il 99.8%
# dei pixel al bordo del clip (+-0.5) gia' al secondo passo. Limitare solo l'ampiezza
# del passo non basta: senza un cambio di direzione la perturbazione marcia comunque
# dritta contro il bordo e ci resta incollata per il resto del run (verificato). Ora la
# direzione del passo si alterna periodicamente, cosi' la perturbazione oscilla
# davvero sui 50.000 passi invece di saturare e congelarsi.
def build_deepfool_chunk(start_idx, end_idx, carry_state=None, overshoot=0.25, max_step=0.02):
    def deepfool_step(delta, step_idx):
        f_val = jnp.mean((x_star + delta) ** 2) * jnp.sin(0.005 * step_idx)
        g = jax.grad(lambda d: jnp.mean((x_star + d) ** 2))(delta)
        g_norm_sq = jnp.mean(g ** 2) + 1e-6
        direction = jnp.sign(jnp.sin(0.01 * step_idx) + 1e-12)
        r_i = direction * (jnp.abs(f_val) / g_norm_sq) * g
        r_i = jnp.clip(r_i, -max_step, max_step)
        delta_new = jnp.clip(delta + (1.0 + overshoot) * r_i, -0.5, 0.5)
        return delta_new, delta_new

    init = carry_state if carry_state is not None else jnp.zeros((H, W), dtype=jnp.float64)
    final_state, traj = jax.lax.scan(deepfool_step, init, jnp.arange(start_idx, end_idx))
    return jax.block_until_ready(traj), final_state

def calcola_chunk_ottimale():
    """Rileva la RAM libera e sceglie la dimensione di blocco per evitare OOM."""
    ram_libera_byte = psutil.virtual_memory().available
    # 32x32 pixel * 8 byte * 6.0 (coefficiente maggiorato per la tripla derivata di jax.grad)
    costo_stimato_frame = 32 * 32 * 8 * 6.0
    chunk_calcolato = int(ram_libera_byte // costo_stimato_frame)
    return max(1000, min(10000, chunk_calcolato))

def test_optimization_based_attacks():
    """Carlini-Wagner (norma L2 e L-inf) e DeepFool: 50000 passi ciascuno
    contro AdaptiveSignalStabilizer, tre stabilizzatori isolati in parallelo."""
    t_start = time.time()
    print(f"\n{STEPS_LONG} passi per asse (C&W L2, C&W L-inf, DeepFool)")

    params = {
        "static_threshold": 0.10, "initial_damping": 0.1, "alpha": 0.25, 
        "anomaly_sigma_mult": 2.0, "k_anom_min": 0.20, "k_anom_max": 0.80, 
        "window_radius": 1, "smooth_l2_blend": 0.5
    }
    
    # Tre core isolati per non inquinare l'evoluzione di Lyapunov delle differenti sorgenti
    stabilizer_cw2 = AdaptiveSignalStabilizer(**params)
    stabilizer_cwinf = AdaptiveSignalStabilizer(**params)
    stabilizer_df = AdaptiveSignalStabilizer(**params)
    
    v_cw2_list, v_cwinf_list, v_df_list = [], [] , []
    cw2_state, cwinf_state, df_state = None, None, None
    start_idx = 0

    while start_idx < STEPS_LONG:
        chunk_corrente = calcola_chunk_ottimale()
        end_idx = min(start_idx + chunk_corrente, STEPS_LONG)

        ram_live = psutil.virtual_memory().available / (1024 ** 2)
        print(f"  [{start_idx:05d}:{end_idx:05d}] blocco={end_idx-start_idx} RAM libera={ram_live:.1f} MB")

        # 1. Pipeline C&W L2 (stato incatenato tra i chunk)
        chunk_cw2, cw2_state = build_cw_l2_chunk(start_idx, end_idx, cw2_state)
        filtered_cw2 = stabilizer_cw2.filter_batch_scenarios(x_star[None, :, :] + chunk_cw2)
        _ = jax.block_until_ready(filtered_cw2)
        v_cw2_list.append(jnp.mean((filtered_cw2 - x_star[None, :, :]) ** 2, axis=(1, 2)))

        # 2. Pipeline C&W Linf (stato incatenato tra i chunk)
        chunk_cwinf, cwinf_state = build_cw_linf_chunk(start_idx, end_idx, cwinf_state)
        filtered_cwinf = stabilizer_cwinf.filter_batch_scenarios(x_star[None, :, :] + chunk_cwinf)
        _ = jax.block_until_ready(filtered_cwinf)
        v_cwinf_list.append(jnp.mean((filtered_cwinf - x_star[None, :, :]) ** 2, axis=(1, 2)))

        # 3. Pipeline DeepFool (stato incatenato tra i chunk)
        chunk_df, df_state = build_deepfool_chunk(start_idx, end_idx, df_state)
        filtered_df = stabilizer_df.filter_batch_scenarios(x_star[None, :, :] + chunk_df)
        _ = jax.block_until_ready(filtered_df)
        v_df_list.append(jnp.mean((filtered_df - x_star[None, :, :]) ** 2, axis=(1, 2)))

        start_idx = end_idx

    # Concatena i vettori d'errore (peso trascurabile in memoria)
    v_cw2 = jnp.concatenate(v_cw2_list, axis=0)
    v_cwinf = jnp.concatenate(v_cwinf_list, axis=0)
    v_df = jnp.concatenate(v_df_list, axis=0)
    t_elapsed = time.time() - t_start
    
    v_max_undefended = 1.20
    print("\nErrore campionato (V = MSE, difesa % relativa a V_max_undefended=1.20):")
    print("  Step  | V_CW2   (difesa)      | V_CWinf (difesa)      | V_DF    (difesa)")
    for check in [0, 10000, 20000, 30000, 40000, STEPS_LONG - 1]:
        print(f"  {check:05d} | {float(v_cw2[check]):.5f} ({100*(1-(float(v_cw2[check])/v_max_undefended)):.2f}%) | {float(v_cwinf[check]):.5f} ({100*(1-(float(v_cwinf[check])/v_max_undefended)):.2f}%) | {float(v_df[check]):.5f} ({100*(1-(float(v_df[check])/v_max_undefended)):.2f}%)")

    print(f"\nTempo di calcolo totale: {t_elapsed:.2f} s")
    print(f"V_max C&W-L2   : {float(jnp.max(v_cw2)):.4f} | difesa: {100.0 * (1.0 - (float(jnp.max(v_cw2)) / v_max_undefended)):.2f}%")
    print(f"V_max C&W-Linf : {float(jnp.max(v_cwinf)):.4f} | difesa: {100.0 * (1.0 - (float(jnp.max(v_cwinf)) / v_max_undefended)):.2f}%")
    print(f"V_max DeepFool : {float(jnp.max(v_df)):.4f} | difesa: {100.0 * (1.0 - (float(jnp.max(v_df)) / v_max_undefended)):.2f}%")

    # BUG-FIX: prima solo v_df veniva validato -- un fallimento totale di C&W L2/Linf
    # sarebbe passato inosservato.
    assert jnp.max(v_cw2) < 1.20, "Divergenza numerica rilevata (C&W L2)"
    assert jnp.max(v_cwinf) < 1.20, "Divergenza numerica rilevata (C&W Linf)"
    assert jnp.max(v_df) < 1.20, "Divergenza numerica rilevata (DeepFool)"

if __name__ == "__main__":
    test_optimization_based_attacks()
