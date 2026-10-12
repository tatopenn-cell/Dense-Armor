"""Online time-series models: autoregression, Kalman filtering, cycle residual.

Three families, one file each. All the models learn one sample at a
time and share the ``dense_armor.roles`` interface.
"""

from dense_armor.utility.timeseries.ar import OnlineAR, OnlineVAR
from dense_armor.utility.timeseries.cycle import CycleResidual
from dense_armor.utility.timeseries.kalman import (
    InnovationAdaptiveKalman,
    KalmanFilter,
    VBAdaptiveKalman,
)

__all__ = [
    "CycleResidual",
    "InnovationAdaptiveKalman",
    "KalmanFilter",
    "OnlineAR",
    "OnlineVAR",
    "VBAdaptiveKalman",
]
