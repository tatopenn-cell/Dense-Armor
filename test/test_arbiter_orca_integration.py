"""
Verifies Orca(use_arbiter=True) against the same 7 scenarios testKalman.py
uses, compared directly to Orca's own default (use_arbiter=False) output.

Real, honest result (see CHANGELOG and utility/arbiter.py for the full
writeup, including a real bug found and fixed in the same session):
use_arbiter never makes RMSE worse than the default on any of the 7
scenarios, and improves it on 5 (A, B, E, F, G). Getting F required
switching classify_segments' reference window from symmetric to causal
(only points before i) -- a symmetric window straddling a sustained level
change mixed old and new level into its own scale, diluting the very
transition it needed to detect (verified: with the symmetric window, F
showed zero detection at all, deviation exactly 0 throughout).
"""
import importlib.util
import logging
import pathlib

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np

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


def _modello(x):
    return x


def _rmse_per_scenario(use_arbiter: bool) -> dict:
    orca = Orca()
    _ = orca.protect_and_forward(_modello, np.full(N, X_CLEAN), x_reference=None, use_arbiter=use_arbiter)
    risultati = {}
    for key, (descr, dati, target) in SCENARI.items():
        out = np.array(orca.protect_and_forward(_modello, dati, x_reference=None, use_arbiter=use_arbiter))
        mask = ~np.isnan(dati)
        risultati[key] = float(np.sqrt(np.mean((out[mask] - target[mask]) ** 2)))
    return risultati


rmse_default = _rmse_per_scenario(use_arbiter=False)
rmse_arbiter = _rmse_per_scenario(use_arbiter=True)


def test_arbiter_never_worse_than_default():
    print("\n" + "=" * 78)
    print("  Orca default vs Orca(use_arbiter=True)")
    print("=" * 78)
    for key, (descr, _, _) in SCENARI.items():
        print(f"[{key}] {descr}")
        print(f"    default: {rmse_default[key]:8.4f}   |   arbiter: {rmse_arbiter[key]:8.4f}")
    # tolleranza 1e-6 per confronti in virgola mobile a parita' di risultato
    for key in SCENARI:
        assert rmse_arbiter[key] <= rmse_default[key] + 1e-6, (
            f"scenario {key}: arbiter ({rmse_arbiter[key]}) peggiore del default ({rmse_default[key]})"
        )


def test_arbiter_improves_on_isolated_anomaly_scenarios():
    # A (impulsi), B (sub-soglia), E (Cauchy), F (rottura di livello,
    # causal window fix), G (buco NaN): miglioramento reale misurato, non
    # solo parita'.
    for key in ("A", "B", "E", "F", "G"):
        assert rmse_arbiter[key] < rmse_default[key]


def test_arbiter_detects_the_real_regime_transition_in_scenario_f():
    # Verifica diretta della rilevazione (non solo l'effetto sull'RMSE):
    # gli indici etichettati 'regime' devono includere la transizione
    # vera (indice 60) -- prima del fix alla finestra causale, NESSUN
    # punto veniva etichettato 'regime' qui (deviazione sempre 0).
    from dense_armor.utility.arbiter import classify_segments
    _, dati, _ = SCENARI["F"]
    etichette, _, _ = classify_segments(dati)
    regime_idx = np.where(etichette == "regime")[0]
    assert regime_idx.size > 0
    assert 60 in regime_idx


def test_labels_and_uncertainty_are_populated():
    orca = Orca()
    _, _, dati, _ = (None, None, SCENARI["A"][1], None)
    out = orca.protect_and_forward(_modello, dati, x_reference=None, use_arbiter=True)
    assert orca.etichette_arbitro is not None
    assert orca.incertezza_arbitro is not None
    assert orca.etichette_arbitro.shape == out.shape
    assert set(np.unique(orca.etichette_arbitro)) <= {"clean", "spike", "regime"}
    assert 0.0 <= orca.incertezza_arbitro_media <= 1.0


def test_default_path_unaffected_when_arbiter_off():
    orca = Orca()
    dati = SCENARI["A"][1]
    out = orca.protect_and_forward(_modello, dati, x_reference=None, use_arbiter=False)
    assert orca.etichette_arbitro is None
    assert orca.incertezza_arbitro is None
    assert out is not None


def test_corruption_type_memory_updates_when_reference_known():
    orca = Orca()
    dati = SCENARI["A"][1]
    target = SCENARI["A"][2]
    slice_shape = (N,)
    assert sum(orca.tipi_corruzione_visti(slice_shape).values()) == 0
    _ = orca.protect_and_forward(_modello, dati, x_reference=target, use_arbiter=True)
    conteggio = orca.tipi_corruzione_visti(slice_shape)
    assert sum(conteggio.values()) == N
    assert conteggio.get("spike", 0) > 0
