"""Dense-Armor online-learning base (Task 01).

Scikit-learn API design (Buitinck et al., arXiv:1309.0238) with
river-style dict inputs and _one / _many split (Montiel et al.,
arXiv:2012.04740). One file per role.

If river is installed, the role classes are registered as virtual
subclasses of river's ABCs, so our models work inside river's
pipelines, checks and evaluators without the core depending on river.
"""

from dense_armor.roles.anomaly_detector import AnomalyDetector, AnomalyGate
from dense_armor.roles.root import Root, InconsistentVersionWarning
from dense_armor.roles.classifier import Classifier
from dense_armor.roles.drift_detector import DriftDetector
from dense_armor.roles.estimator import Estimator
from dense_armor.roles.protection import Protected
from dense_armor.roles.regressor import Regressor
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
    "Signal",
    "AdaptiveConformalRegressor",
    "Estimate",
    "PureEW",
    "RealtimePipeline",
    "SafeEstimator",
    "UnitCheckedPipeline",
    "UnitSpec",
    "call_json",
    "limits_from_urdf",
    "profile",
    "schema",

    "AnomalyDetector",
    "AnomalyGate",
    "Root",
    "Classifier",
    "DriftDetector",
    "Estimator",
    "InconsistentVersionWarning",
    "Protected",
    "Regressor",
    "Transformer",
    "TransformerSupervised",
    "ModelWrapper",
]

from dense_armor.roles.signal import Signal  # noqa: E402
from dense_armor.roles.agents import call_json, schema  # noqa: E402
from dense_armor.roles.physics import UnitCheckedPipeline, UnitSpec, limits_from_urdf  # noqa: E402
from dense_armor.roles.realtime import PureEW, RealtimePipeline, profile  # noqa: E402
from dense_armor.roles.safety import SafeEstimator  # noqa: E402
from dense_armor.roles.uncertainty import AdaptiveConformalRegressor, Estimate  # noqa: E402
from dense_armor.roles.safety import Health  # noqa: E402
from dense_armor.roles.physics import PhysicalLimitsGuard  # noqa: E402
