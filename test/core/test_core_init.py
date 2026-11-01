# -*- coding: utf-8 -*-
"""Regressione per il bug core/init.py vs core/__init__.py: il file con
il nome giusto era vuoto, quindi nessuna di queste importazioni funzionava
a livello di pacchetto anche se il codice reale esisteva."""


def test_import_diretto_dal_pacchetto_core():
    from dense_armor.core import (
        AIHardwareProfiler,
        UniversalMemoryGuard,
        MemoryPressureError,
        AdaptiveSignalStabilizer,
        apply_damping_blend,
        __version__,
    )

    import dense_armor

    assert __version__ == dense_armor.__version__
    assert callable(apply_damping_blend)


def test_init_py_col_nome_sbagliato_non_esiste_piu():
    import pathlib
    import dense_armor.core as core_pkg

    core_dir = pathlib.Path(core_pkg.__file__).parent
    assert not (core_dir / "init.py").exists()
