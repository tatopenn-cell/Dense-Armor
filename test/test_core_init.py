# -*- coding: utf-8 -*-
"""Regressione per il bug core/init.py vs core/__init__.py: il file con
il nome giusto era vuoto, quindi nessuna di queste importazioni funzionava
a livello di pacchetto anche se il codice reale esisteva."""


def test_import_diretto_dal_pacchetto_core():
    from dense_armor.core import (
        AIHardwareProfiler,
        StochasticAdversarialNoise,
        UniversalMemoryGuard,
        MemoryPressureError,
        ParametricScenarioSimulator,
        BitwisePermutationEngine,
        AdaptiveSignalStabilizer,
        DynamicAICodegen,
        CMD_MAP,
        TensorVault,
        AIEngineVisualizer,
        PipelineProfiler,
        apply_damping_blend,
        __version__,
    )

    assert __version__ == "1.0.11"
    assert callable(apply_damping_blend)
    assert isinstance(CMD_MAP, dict) and "relu" in CMD_MAP


def test_init_py_col_nome_sbagliato_non_esiste_piu():
    import pathlib
    import dense_armor.core as core_pkg

    core_dir = pathlib.Path(core_pkg.__file__).parent
    assert not (core_dir / "init.py").exists()
