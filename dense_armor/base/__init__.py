"""Dense-Armor online-learning base (Task 01).

Scikit-learn API design (Buitinck et al., arXiv:1309.0238) with
river-style dict inputs and _one / _many split (Montiel et al.,
arXiv:2012.04740). One file per role.

If river is installed, the role classes are registered as virtual
subclasses of river's ABCs, so our models work inside river's
pipelines, checks and evaluators without the core depending on river.
"""

from dense_armor.base.anomaly_detector import AnomalyDetector, AnomalyFilter
from dense_armor.base.base import Base, InconsistentVersionWarning
from dense_armor.base.classifier import Classifier
from dense_armor.base.drift_detector import DriftDetector
from dense_armor.base.estimator import Estimator
from dense_armor.base.protection import Protected
from dense_armor.base.regressor import Regressor
from dense_armor.base.transformer import Transformer, TransformerSupervised
from dense_armor.base.wrapper import Wrapper

_ROLE_MAP = (
    ("Classifier", Classifier),
    ("Regressor", Regressor),
    ("Transformer", Transformer),
    ("AnomalyDetector", AnomalyDetector),
    ("DriftDetector", DriftDetector),
    ("Wrapper", Wrapper),
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
    "AnomalyDetector",
    "AnomalyFilter",
    "Base",
    "Classifier",
    "DriftDetector",
    "Estimator",
    "InconsistentVersionWarning",
    "Protected",
    "Regressor",
    "Transformer",
    "TransformerSupervised",
    "Wrapper",
]
