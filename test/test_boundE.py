import sys
import os
import time
import jax

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Alias con underscore iniziale (non "test_*"): altrimenti pytest li
# raccoglie ANCHE come test a se' stanti (qualunque nome "test_*" a livello
# di modulo, importato o no) -- ogni categoria girerebbe due volte, una
# volta qui direttamente e una dentro test_adversarial_suite_summary sotto
# (verificato: raddoppiava il tempo del file, 91s invece di 45s).
from test_boundA import test_gradient_based_attacks as _test_A
from test_boundB import test_geometric_spatial_attacks as _test_B
from test_boundC import test_optimization_based_attacks as _test_C
from test_boundD import test_fourier_domain_attacks as _test_D

def test_adversarial_suite_summary():
    """Esegue le 4 categorie di attacco (CAT-A/B/C/D) in sequenza e stampa un
    riepilogo consolidato -- stessa logica di test_boundA-D.py, eseguite una
    per una qui sotto invece che separatamente."""
    jax.config.update("jax_enable_x64", True)

    print("\nSuite di difesa adversarial -- riepilogo consolidato")

    validation_pipeline = [
        {"id": "CAT-A", "label": "Attacchi a gradiente (PGD, BIM, MI-FGSM)", "fn": _test_A, "steps": 140},
        {"id": "CAT-B", "label": "Trasformazioni spaziali/geometriche (affine, elastica)", "fn": _test_B, "steps": 50000},
        {"id": "CAT-C", "label": "Attacchi a ottimizzazione (Carlini-Wagner, DeepFool)", "fn": _test_C, "steps": 50000},
        {"id": "CAT-D", "label": "Dominio delle frequenze (Fourier a banda media)", "fn": _test_D, "steps": 50000}
    ]

    execution_reports = {}
    t_global_start = time.time()

    for stage in validation_pipeline:
        print(f"\n[{stage['id']}] {stage['label']}...")
        t_stage_start = time.time()
        try:
            stage["fn"]()
            jax.block_until_ready(None)
            t_stage_elapsed = time.time() - t_stage_start
            execution_reports[stage["id"]] = {
                "label": stage["label"],
                "status": "OK",
                "time": t_stage_elapsed,
                "steps": stage["steps"]
            }
            print(f"[{stage['id']}] completato")
        except AssertionError as exc:
            t_stage_elapsed = time.time() - t_stage_start
            execution_reports[stage["id"]] = {"label": stage["label"], "status": "FALLITO", "time": t_stage_elapsed, "steps": stage["steps"]}
            print(f"[{stage['id']}] fallito: {exc}")
            raise exc

    t_global_elapsed = time.time() - t_global_start

    print("\nRiepilogo")
    print(f"  {'ID':<6} | {'Categoria':<50} | {'Passi':>10} | {'Esito':<7} | {'Tempo':>8}")
    total_steps_executed = 0
    for stage_id, data in execution_reports.items():
        total_steps_executed += data["steps"]
        print(f"  {stage_id:<6} | {data['label'][:50]:<50} | {data['steps']:>10,} | {data['status']:<7} | {data['time']:>6.3f} s")

    print(f"\nPassi totali: {total_steps_executed:,} | tempo totale: {t_global_elapsed:.3f} s")

if __name__ == "__main__":
    test_adversarial_suite_summary()
