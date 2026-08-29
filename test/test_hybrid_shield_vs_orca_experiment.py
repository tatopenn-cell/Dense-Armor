"""
Compares dense_armor.core.hybrid_engine.hybrid_shield (the phi_ab-based
engine that already replaced ABCollatz for Armatura) against Orca's own
default pipeline (constant compute_damping_gating), on the same 7 scenarios
testKalman.py uses -- to check whether the already-vendored phi_ab approach
would also help Orca's Stage-2 gate, before adapting it into that exact
call site.

hybrid_shield's own output IS the cleaned series (not a [0,1] gate to
blend with), so this compares full pipelines end to end, not one swapped
function.
"""
import importlib.util
import logging
import pathlib

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np

from dense_armor.core.hybrid_engine import hybrid_shield
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


orca = Orca()
_ = orca.protect_and_forward(_modello_a_valle, np.full(N, X_CLEAN), x_reference=None)

rmse_orca = {}
rmse_hybrid = {}
for key, (descr, dati, target) in SCENARI.items():
    out_orca = np.array(orca.protect_and_forward(_modello_a_valle, dati, x_reference=None))
    out_hybrid, _, _ = hybrid_shield(dati, riferimento=None)

    mask = ~np.isnan(dati)
    rmse_orca[key] = float(np.sqrt(np.mean((out_orca[mask] - target[mask]) ** 2)))
    rmse_hybrid[key] = float(np.sqrt(np.mean((out_hybrid[mask] - target[mask]) ** 2)))

vittorie_hybrid = sum(1 for k in SCENARI if rmse_hybrid[k] < rmse_orca[k])
vittorie_orca = sum(1 for k in SCENARI if rmse_orca[k] < rmse_hybrid[k])


def test_hybrid_shield_measured_against_orca_default():
    print("\n" + "=" * 78)
    print("  Orca (default, gate costante) vs hybrid_shield (phi_ab, gia' in Armatura)")
    print("=" * 78)
    for key, (descr, _, _) in SCENARI.items():
        print(f"[{key}] {descr}")
        print(f"    RMSE Orca: {rmse_orca[key]:8.4f}   |   RMSE hybrid_shield: {rmse_hybrid[key]:8.4f}")
    print(f"\nhybrid_shield vince in {vittorie_hybrid}/{len(SCENARI)}, Orca vince in "
          f"{vittorie_orca}/{len(SCENARI)}")
    assert all(np.isfinite(v) for v in rmse_orca.values())
    assert all(np.isfinite(v) for v in rmse_hybrid.values())
