"""Old import paths, kept working: each name resolves to the same module object."""
import importlib
import importlib.abc
import importlib.util
import sys

ALIASES = {
    "dense_armor.anomaly": "dense_armor.utility.anomaly",
    "dense_armor.anomaly.curvature": "dense_armor.utility.anomaly.curvature",
    "dense_armor.anomaly.deviation": "dense_armor.utility.anomaly.deviation",
    "dense_armor.anomaly.filters": "dense_armor.utility.anomaly.filters",
    "dense_armor.anomaly.mahalanobis": "dense_armor.utility.anomaly.mahalanobis",
    "dense_armor.anomaly.one_sided": "dense_armor.utility.anomaly.one_sided",
    "dense_armor.anomaly.resonance_search": "dense_armor.utility.anomaly.resonance_search",
    "dense_armor.anomaly.robust_filters": "dense_armor.utility.anomaly.robust_filters",
    "dense_armor.anomaly.streaming": "dense_armor.utility.anomaly.streaming",
    "dense_armor.control": "dense_armor.utility.control",
    "dense_armor.control.cbf_filter": "dense_armor.utility.control.cbf_filter",
    "dense_armor.control.kinematic_controller": "dense_armor.utility.control.kinematic_controller",
    "dense_armor.control.rate_limiter": "dense_armor.utility.control.rate_limiter",
    "dense_armor.control.trajectory": "dense_armor.utility.control.trajectory",
    "dense_armor.drift": "dense_armor.utility.drift",
    "dense_armor.drift.cusum": "dense_armor.utility.drift.cusum",
    "dense_armor.drift.detector": "dense_armor.utility.drift.detector",
    "dense_armor.dynamics.online_dynamics": "dense_armor.utility.learn.online_dynamics",
    "dense_armor.learn": "dense_armor.utility.learn",
    "dense_armor.learn.calibration": "dense_armor.utility.learn.calibration",
    "dense_armor.learn.metric_learning": "dense_armor.utility.learn.metric_learning",
    "dense_armor.learn.online_classifiers": "dense_armor.utility.learn.online_classifiers",
    "dense_armor.learn.online_dynamics": "dense_armor.utility.learn.online_dynamics",
    "dense_armor.misc": "dense_armor.utility.misc",
    "dense_armor.misc.anwav": "dense_armor.utility.misc.anwav",
    "dense_armor.misc.collatz": "dense_armor.utility.misc.collatz",
    "dense_armor.misc.diagnostic": "dense_armor.utility.misc.diagnostic",
    "dense_armor.misc.iodat": "dense_armor.utility.misc.iodat",
    "dense_armor.misc.metro": "dense_armor.utility.misc.metro",
    "dense_armor.protect": "dense_armor.utility.protect",
    "dense_armor.protect.arbiter": "dense_armor.utility.protect.arbiter",
    "dense_armor.protect.healing": "dense_armor.utility.protect.healing",
    "dense_armor.protect.orca": "dense_armor.utility.protect.orca",
    "dense_armor.protect.stable_frame_filter": "dense_armor.utility.protect.stable_frame_filter",
    "dense_armor.protect.streaming_arbiter": "dense_armor.utility.protect.streaming_arbiter",
    "dense_armor.utility.anwav": "dense_armor.utility.misc.anwav",
    "dense_armor.utility.arbiter": "dense_armor.utility.protect.arbiter",
    "dense_armor.utility.calibration": "dense_armor.utility.learn.calibration",
    "dense_armor.utility.cbf_filter": "dense_armor.utility.control.cbf_filter",
    "dense_armor.utility.collatz": "dense_armor.utility.misc.collatz",
    "dense_armor.utility.curvature": "dense_armor.utility.anomaly.curvature",
    "dense_armor.utility.cusum": "dense_armor.utility.drift.cusum",
    "dense_armor.utility.diagnostic": "dense_armor.utility.misc.diagnostic",
    "dense_armor.utility.healing": "dense_armor.utility.protect.healing",
    "dense_armor.utility.iodat": "dense_armor.utility.misc.iodat",
    "dense_armor.utility.kinematic_controller": "dense_armor.utility.control.kinematic_controller",
    "dense_armor.utility.metric_learning": "dense_armor.utility.learn.metric_learning",
    "dense_armor.utility.metro": "dense_armor.utility.misc.metro",
    "dense_armor.utility.one_sided": "dense_armor.utility.anomaly.one_sided",
    "dense_armor.utility.online_classifiers": "dense_armor.utility.learn.online_classifiers",
    "dense_armor.utility.orca": "dense_armor.utility.protect.orca",
    "dense_armor.utility.rate_limiter": "dense_armor.utility.control.rate_limiter",
    "dense_armor.utility.resonance_search": "dense_armor.utility.anomaly.resonance_search",
    "dense_armor.utility.river_anomaly": "dense_armor.utility.anomaly.deviation",
    "dense_armor.utility.river_drift": "dense_armor.utility.drift.detector",
    "dense_armor.utility.robust_filters": "dense_armor.utility.anomaly.robust_filters",
    "dense_armor.utility.stable_frame_filter": "dense_armor.utility.protect.stable_frame_filter",
    "dense_armor.utility.streaming": "dense_armor.utility.anomaly.streaming",
    "dense_armor.utility.streaming_arbiter": "dense_armor.utility.protect.streaming_arbiter",
    "dense_armor.utility.streaming_filters": "dense_armor.utility.anomaly.filters",
    "dense_armor.utility.streaming_mahalanobis": "dense_armor.utility.anomaly.mahalanobis",
    "dense_armor.utility.trajectory": "dense_armor.utility.control.trajectory",
}


class _AliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name in ALIASES:
            return importlib.util.spec_from_loader(name, self)
        return None

    def create_module(self, spec):
        module = importlib.import_module(ALIASES[spec.name])
        sys.modules[spec.name] = module
        return module

    def exec_module(self, module):
        pass


def install():
    if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _AliasFinder())
