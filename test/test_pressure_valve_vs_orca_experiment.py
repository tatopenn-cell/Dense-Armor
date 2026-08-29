"""
Three-way comparison, same 7 scenarios as testKalman.py: Orca's own default
pipeline (constant compute_damping_gating), hybrid_shield (phi_ab, already
vendored for Armatura), and pressure_valve (JSD-adaptive threshold, already
in robust_filters.py but never wired into either pipeline).

pressure_valve is a batch/offline filter (its own docstring: not for
core/hybrid_engine.py's real-time cycle) -- run here on the whole series at
once, same as the other two, for a fair like-for-like RMSE comparison.
"""
import importlib.util
import logging
import pathlib

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np

from dense_armor.core.hybrid_engine import hybrid_shield
from dense_armor.utility.orca import Orca
from dense_armor.utility.robust_filters import pressure_valve

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


orca = Orca()
_ = orca.protect_and_forward(_modello_a_valle, np.full(N, X_CLEAN), x_reference=None)

rmse_orca, rmse_hybrid, rmse_valve = {}, {}, {}
for key, (descr, dati, target) in SCENARI.items():
    out_orca = np.array(orca.protect_and_forward(_modello_a_valle, dati, x_reference=None))
    out_hybrid, _, _ = hybrid_shield(dati, riferimento=None)

    dati_no_nan = np.where(np.isnan(dati), np.nanmedian(dati), dati)
    out_valve, _, _, _ = pressure_valve(dati_no_nan)

    mask = ~np.isnan(dati)
    rmse_orca[key] = float(np.sqrt(np.mean((out_orca[mask] - target[mask]) ** 2)))
    rmse_hybrid[key] = float(np.sqrt(np.mean((out_hybrid[mask] - target[mask]) ** 2)))
    rmse_valve[key] = float(np.sqrt(np.mean((out_valve[mask] - target[mask]) ** 2)))

_vincitori = {}
for k in SCENARI:
    scores = {"Orca": rmse_orca[k], "hybrid_shield": rmse_hybrid[k], "pressure_valve": rmse_valve[k]}
    _vincitori[k] = min(scores, key=scores.get)


def test_three_way_comparison():
    print("\n" + "=" * 92)
    print("  Orca (costante) vs hybrid_shield (phi_ab) vs pressure_valve (JSD-adattivo)")
    print("=" * 92)
    for key, (descr, _, _) in SCENARI.items():
        print(f"[{key}] {descr}")
        print(f"    Orca: {rmse_orca[key]:8.4f}   |   hybrid_shield: {rmse_hybrid[key]:8.4f}"
              f"   |   pressure_valve: {rmse_valve[key]:8.4f}   ->  vince {_vincitori[key]}")
    from collections import Counter
    conteggio = Counter(_vincitori.values())
    print(f"\nVittorie: {dict(conteggio)}")
    assert all(np.isfinite(v) for v in rmse_orca.values())
    assert all(np.isfinite(v) for v in rmse_hybrid.values())
    assert all(np.isfinite(v) for v in rmse_valve.values())
