"""Dense-Armor online-learning base (Task 01).

General-purpose machine-learning API design (Buitinck et al., arXiv:1309.0238) with
dict-input, one-sample-at-a-time style (Montiel et al.,
arXiv:2012.04740). One file per role.

If the optional online-learning dependency is installed, the role classes are registered as virtual
subclasses of its ABCs, so our models work inside its
pipelines, checks and evaluators without the core depending on it.
"""

from dense_armor.roles.anomaly_detector import AnomalyDetector, AnomalyGate
from dense_armor.roles.classifier import Classifier
from dense_armor.roles.drift_detector import DriftDetector
from dense_armor.roles.estimator import Estimator
from dense_armor.roles.protection import Protected
from dense_armor.roles.regressor import Regressor
from dense_armor.roles.root import InconsistentVersionWarning, Root
from dense_armor.roles.transformer import Transformer, TransformerSupervised
from dense_armor.roles.wrapper import ModelWrapper

_ROLE_MAP = (
    ("Classifier", Classifier),
    ("Regressor", Regressor),
    ("Transformer", Transformer),
    ("AnomalyDetector", AnomalyDetector),
    ("DriftDetector", DriftDetector),
    ("ModelWrapper", ModelWrapper),
)

try:
    from river import base as _river_base
except ModuleNotFoundError:
    _river_base = None  # type: ignore[assignment]

if _river_base is not None:
    for _name, _cls in _ROLE_MAP:
        _rcls = getattr(_river_base, _name, None)
        if _rcls is None:
            continue
        try:
            _rcls.register(_cls)
        except (TypeError, AttributeError):
            pass
    del _name, _cls, _rcls
del _ROLE_MAP

__all__ = [
    "AdaptiveConformalRegressor",
    "AnomalyDetector",
    "AnomalyGate",
    "Classifier",
    "DriftDetector",
    "Estimate",
    "Estimator",
    "InconsistentVersionWarning",
    "ModelWrapper",
    "Protected",
    "PureEW",
    "RealtimePipeline",
    "Regressor",
    "Root",
    "SafeEstimator",
    "Signal",
    "Transformer",
    "TransformerSupervised",
    "UnitCheckedPipeline",
    "UnitSpec",
    "call_json",
    "limits_from_urdf",
    "profile",
    "schema",
]

from dense_armor.roles.agents import call_json, schema
from dense_armor.roles.physics import (
    PhysicalLimitsGuard,
    UnitCheckedPipeline,
    UnitSpec,
    limits_from_urdf,
)
from dense_armor.roles.realtime import PureEW, RealtimePipeline, profile
from dense_armor.roles.safety import (
    Health,
    SafeEstimator,
)
from dense_armor.roles.signal import Signal
from dense_armor.roles.uncertainty import (
    AdaptiveConformalRegressor,
    Estimate,
)
