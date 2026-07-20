"""Dense-Armor - the wearable Sentinel shield: two-stage anomaly damping for any AI.
Author: Salvatore Pennacchio (Napoli, 2026). Sister project: Dense-Evolution."""
import logging

from .armatura import Armatura

__version__ = "1.1.1"
__all__ = ["Armatura"]

# Nessun handler configurato di default (convenzione standard per librerie):
# chi integra dense-armor decide dove/se mostrare i log (es.
# logging.basicConfig(level=logging.INFO)), invece di doverli sempre vedere
# su stdout o modificare il sorgente per silenziarli.
logging.getLogger("dense_armor").addHandler(logging.NullHandler())
