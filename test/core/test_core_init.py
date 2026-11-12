# -*- coding: utf-8 -*-
"""Regressione per il bug core/init.py vs core/__init__.py: il file con
il nome giusto era vuoto, quindi nessuna di queste importazioni funzionava
a livello di pacchetto anche se il codice reale esisteva."""


def test_import_diretto_dal_pacchetto_core():
    import dense_armor
    from dense_armor.utility import (
        AdaptiveSignalStabilizer,
        AIHardwareProfiler,
        MemoryPressureError,
        UniversalMemoryGuard,
        __version__,
        apply_damping_blend,
    )

    assert __version__ == dense_armor.__version__
    assert callable(apply_damping_blend)


def test_init_py_col_nome_sbagliato_non_esiste_piu():
    import pathlib

    import dense_armor.utility as core_pkg

    core_dir = pathlib.Path(core_pkg.__file__).parent
    assert not (core_dir / "init.py").exists()
