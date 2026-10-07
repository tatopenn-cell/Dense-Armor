# -*- coding: utf-8 -*-
"""Regressione per il bug di duplicazione verbatim in core/preset.py
(l'intero dizionario dei preset era definito due volte nello stesso
file)."""
import ast
import pathlib

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from dense_armor.core.preset import SIGNAL_STABILIZER_PRESETS
from dense_armor.core.engine import AdaptiveSignalStabilizer

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


def test_preset_diversi_producono_filtraggio_davvero_diverso():
    """L'utilita' dichiarata dei preset non e' che esistano con chiavi valide,
    ma che collegarli a AdaptiveSignalStabilizer cambi davvero il
    comportamento di filtraggio. balanced_v2 (alpha=0.05, damping=0.05) e
    pure_1d_time_v1 (alpha=0.35, damping=0.15) sono calibrati per regimi
    diversi (segnale stabilizzato vs serie 1D piu' reattiva): su una serie
    rumorosa con un outlier, devono produrre una varianza residua
    misurabilmente diversa, non la stessa a meno di rumore numerico."""
    rng = np.random.default_rng(0)
    serie = rng.normal(loc=1.0, scale=0.05, size=(1, 200))
    serie[0, 100] = 20.0  # outlier isolato

    varianze = {}
    for nome in ("balanced_v2", "pure_1d_time_v1"):
        stabilizer = AdaptiveSignalStabilizer(**SIGNAL_STABILIZER_PRESETS[nome])
        out = np.array(stabilizer.filter_batch_scenarios(serie))
        varianze[nome] = float(np.var(out))

    rapporto = varianze["pure_1d_time_v1"] / varianze["balanced_v2"]
    assert rapporto > 2.0, (
        f"i due preset producono varianze residue troppo simili ({varianze}) "
        "per essere davvero configurazioni distinte"
    )
