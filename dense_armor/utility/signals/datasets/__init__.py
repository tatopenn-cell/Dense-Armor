"""Datasets: ready-made streams for tests and experiments."""

from dense_armor.utility.signals.datasets.casper import Casper
from dense_armor.utility.signals.datasets.drift_stream import DriftStream
from dense_armor.utility.signals.datasets.synthetic_arm import SyntheticArm

__all__ = [
    "Casper",
    "DriftStream",
    "SyntheticArm",
]
