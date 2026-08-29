"""
The benchmark collatz.py's own compute_damping_gating_smooth docstring has
referenced since it was written, but that never actually existed until now.

Runs the real Orca.protect_and_forward() pipeline on the same 7 scenarios
testKalman.py uses, twice: once with the current default gate
(compute_damping_gating, a declared constant 0.85), once with the
never-benchmarked smooth variant (compute_damping_gating_smooth) swapped in
via monkeypatch -- same signature, same call site, only the formula differs.
Reports real RMSE for both on every scenario.
"""
import importlib.util
import logging
import pathlib

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np

from dense_armor.utility.collatz import ABCollatz
from dense_armor.utility.orca import Orca

_TEST_DIR = pathlib.Path(__file__).resolve().parent


def _import_kalman_scenarios():
    spec = importlib.util.spec_from_file_location("testKalman", _TEST_DIR / "testKalman.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_km = _import_kalman_scenarios()

logging.getLogger("dense_armor").setLevel(logging.CRITICAL)

X_CLEAN = _km.X_CLEAN
N = _km.N
SCENARI = _km.genera_scenari(X_CLEAN, N)


def _modello_a_valle(x):
    return x


def _rmse_per_scenario(gate_fn) -> dict:
    original = ABCollatz.compute_damping_gating
    ABCollatz.compute_damping_gating = gate_fn
    try:
        orca = Orca()
        _ = orca.protect_and_forward(_modello_a_valle, np.full(N, X_CLEAN), x_reference=None)
        risultati = {}
        for key, (descr, dati, target) in SCENARI.items():
            out = np.array(orca.protect_and_forward(_modello_a_valle, dati, x_reference=None))
            mask = ~np.isnan(dati)
            risultati[key] = float(np.sqrt(np.mean((out[mask] - target[mask]) ** 2)))
        return risultati
    finally:
        ABCollatz.compute_damping_gating = original


rmse_costante = _rmse_per_scenario(ABCollatz.compute_damping_gating)
rmse_smooth = _rmse_per_scenario(ABCollatz.compute_damping_gating_smooth)

vittorie_smooth = sum(1 for k in SCENARI if rmse_smooth[k] < rmse_costante[k])
vittorie_costante = sum(1 for k in SCENARI if rmse_costante[k] < rmse_smooth[k])


def test_smooth_variant_measured_against_constant_default():
    print("\n" + "=" * 78)
    print("  compute_damping_gating (costante 0.85) vs compute_damping_gating_smooth")
    print("=" * 78)
    for key, (descr, _, _) in SCENARI.items():
        print(f"[{key}] {descr}")
        print(f"    RMSE costante: {rmse_costante[key]:8.4f}   |   RMSE smooth: {rmse_smooth[key]:8.4f}")
    print(f"\nSmooth vince in {vittorie_smooth}/{len(SCENARI)}, costante vince in "
          f"{vittorie_costante}/{len(SCENARI)}")
    assert all(np.isfinite(v) for v in rmse_costante.values())
    assert all(np.isfinite(v) for v in rmse_smooth.values())
