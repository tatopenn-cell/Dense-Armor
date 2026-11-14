"""Dense-Armor - a low-level learning toolkit: online, deep and machine learning, with ontology, audio, video, robot motion control and finance.
Author: Salvatore Pennacchio (Napoli, 2026). Sister project: Dense-Evolution."""
import logging

__version__ = "1.6.0"

from . import _compat

_compat.install()

from ._dtype import get_default_dtype, set_default_dtype
from .armor import Armor

__all__ = ["Armor", "get_default_dtype", "set_default_dtype"]


def __getattr__(name):
    if name == "Armatura":
        import warnings

        warnings.warn(
            "dense_armor.Armatura is the old name of dense_armor.Armor",
            DeprecationWarning,
            stacklevel=2,
        )
        return Armor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Nessun handler configurato di default (convenzione standard per librerie):
# chi integra dense-armor decide dove/se mostrare i log (es.
# logging.basicConfig(level=logging.INFO)), invece di doverli sempre vedere
# su stdout o modificare il sorgente per silenziarli.
logging.getLogger("dense_armor").addHandler(logging.NullHandler())
