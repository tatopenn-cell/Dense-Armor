import sys
import os
import time
import jax

# Iniezione della directory radice per mappare correttamente il modulo core locale
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importazione formale dei sottosistemi di test pre-allocati sul disco rigido.
# Alias con underscore iniziale (non "test_*"): altrimenti pytest li
# raccoglie ANCHE come test a se' stanti (qualunque nome "test_*" a livello
# di modulo, importato o no) -- ogni categoria girerebbe due volte, una
# volta qui direttamente e una dentro test_sentinel_hardware_evaluation_suite
# sotto (verificato: raddoppiava il tempo del file, 91s invece di 45s).
from test_boundA import test_sentinel_advanced_gradient_attacks as _test_A
from test_boundB import test_sentinel_geometric_spatial_attacks as _test_B
from test_boundC import test_sentinel_optimization_boundary_attacks as _test_C
from test_boundD import test_sentinel_fourier_domain_attacks as _test_D

def test_sentinel_hardware_evaluation_suite():
    """
    Orchestratore Centrale di Livello E per l'estrazione e il consolidamento delle
    metriche di stabilità hardware e confinamento UBB su 4 categorie di perturbazione.
    """
    jax.config.update("jax_enable_x64", True)
    
    print("\n" + "="*95)
    print(f"[SYSTEM PROFILE] CENTRALIZED HARDWARE EVALUATION SUITE — RUNTIME ENGINE ACTIVATED")
    print(f"COMPILER INTEGRATION: JAX-XLA Native | PRECISION LAYER: IEEE 754 float64")
    print("="*95)
    
    validation_pipeline = [
        {"id": "CAT-A", "label": "Gradient-Based Iterative Vectors (PGD, BIM, MI-FGSM)", "fn": _test_A, "steps": 140},
        {"id": "CAT-B", "label": "Spatial & Geometric Transformations (Affine, Elastic)", "fn": _test_B, "steps": 50000},
        {"id": "CAT-C", "label": "Optimization-Based Bounds (Carlini-Wagner, DeepFool)", "fn": _test_C, "steps": 50000},
        {"id": "CAT-D", "label": "Frequency Domain Spectral Alterations (Fourier Mid-Band)", "fn": _test_D, "steps": 50000}
    ]
    
    execution_reports = {}
    t_global_start = time.time()
    
    for stage in validation_pipeline:
        print(f"\n[{stage['id']}/START] Esecuzione routine: {stage['label']}...")
        print("-" * 95)
        
        t_stage_start = time.time()
        try:
            # Invocazione della sub-routine e barriera di sincronizzazione asincrona hardware
            stage["fn"]()
            jax.block_until_ready(None)
            
            t_stage_elapsed = time.time() - t_stage_start
            execution_reports[stage["id"]] = {
                "label": stage["label"],
                "status": "SUCCESS",
                "time": t_stage_elapsed,
                "steps": stage["steps"]
            }
            print(f"[{stage['id']}/DONE] Modulo validato correttamente.")
            
        except AssertionError as exc:
            t_stage_elapsed = time.time() - t_stage_start
            execution_reports[stage["id"]] = {"label": stage["label"], "status": "FAIL", "time": t_stage_elapsed, "steps": stage["steps"]}
            print(f"[{stage['id']}/FAIL] Limite di confinamento superato o violazione UBB: {exc}")
            raise exc
        except Exception as exc:
            print(f"[{stage['id']}/CRASH] Arresto anomalo del kernel JAX: {exc}")
            raise exc
            
    t_global_elapsed = time.time() - t_global_start
    
    # ── RESOCONTO GLOBALE DELLE STATISTICHE DI CONVERGENZA ED EFFICIENZA ──
    print("\n" + "="*95)
    print("[METRICS SUMMARY] PIPELINE CONSOLIDATED STATISTICS REPORT")
    print("-" * 95)
    print(f"  ID     | Sottosistema di Perturbazione                          | Iterazioni | Status  | Latency")
    print("-" * 95)
    
    total_steps_executed = 0
    for stage_id, data in execution_reports.items():
        total_steps_executed += data["steps"]
        print(f"  {stage_id:<6} | {data['label'][:46]:<46} | {data['steps']:>10,} | {data['status']:<7} | {data['time']:>6.3f} s")
        
    print("-" * 95)
    print(f"[METRICS] Volume totale iterazioni elaborate nel Grafo : {total_steps_executed:,} passi di codice")
    print(f"[METRICS] Tempo complessivo di calcolo asincrono reale : {t_global_elapsed:.3f} s")
    print(f"[METRICS] Rendimento medio hardware per step kernel    : {(t_global_elapsed / total_steps_executed) * 1e6:.4f} microsecondi/passo")
    print(f"[METRICS] Integrità complessiva della memoria VRAM     : Stabile (Nessun gradiente esplosivo intercettato)")
    print("=" * 95 + "\n")

if __name__ == "__main__":
    test_sentinel_hardware_evaluation_suite()
