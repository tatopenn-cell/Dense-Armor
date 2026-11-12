"""Datasets: ready-made streams for tests and experiments."""

from dense_armor.core.signals.datasets.casper import Casper
from dense_armor.core.signals.datasets.drift_stream import DriftStream
from dense_armor.core.signals.datasets.synthetic_arm import SyntheticArm

__all__ = [
    "Casper",
    "DriftStream",
    "SyntheticArm",
]
