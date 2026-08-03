# -*- coding: utf-8 -*-
"""Regressione per il bug di duplicazione verbatim in core/preset.py
(l'intero dizionario dei preset era definito due volte nello stesso
file)."""
import ast
import pathlib

from dense_armor.core.preset import SIGNAL_STABILIZER_PRESETS

REQUIRED_KEYS = {
    "static_threshold",
    "initial_damping",
    "alpha",
    "anomaly_sigma_mult",
    "k_anom_min",
    "k_anom_max",
}


def test_preset_contiene_esattamente_i_quattro_profili_calibrati():
    assert set(SIGNAL_STABILIZER_PRESETS.keys()) == {
        "balanced_v2",
        "cifar10_best_v1",
        "pure_1d_time_v1",
        "cifar10_hardened_lyapunov",
    }


def test_ogni_preset_ha_tutte_le_chiavi_numeriche_richieste():
    for name, preset in SIGNAL_STABILIZER_PRESETS.items():
        assert set(preset.keys()) == REQUIRED_KEYS, name
        for key, value in preset.items():
            assert isinstance(value, (int, float)), f"{name}.{key}"


def test_il_file_definisce_i_preset_una_sola_volta():
    import dense_armor.core.preset as preset_module

    source = pathlib.Path(preset_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "SIGNAL_STABILIZER_PRESETS"
            for t in node.targets
        )
    ]
    assert len(assignments) == 1
