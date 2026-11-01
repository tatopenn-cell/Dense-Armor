"""Online time-series models: autoregression, Kalman filtering, cycle residual.

Three families, one file each. All the models learn one sample at a
time and share the ``dense_armor.roles`` interface.
"""

from dense_armor.utility.online.timeseries.ar import OnlineAR, OnlineVAR
from dense_armor.utility.online.timeseries.cycle import CycleResidual
from dense_armor.utility.online.timeseries.garch import GARCH11
from dense_armor.utility.online.timeseries.kalman import (
    InnovationAdaptiveKalman,
    KalmanFilter,
    VBAdaptiveKalman,
)

__all__ = [
    "CycleResidual",
    "GARCH11",
    "InnovationAdaptiveKalman",
    "KalmanFilter",
    "OnlineAR",
    "OnlineVAR",
    "VBAdaptiveKalman",
]
