"""Old import paths, kept working: each name resolves to the same module object."""

import importlib
import importlib.abc
import importlib.util
import sys

ALIASES = {
    "dense_armor.armatura": "dense_armor.armor",
    "dense_armor.anomaly": "dense_armor.utility.online.anomaly",
    "dense_armor.anomaly.curvature": "dense_armor.utility.online.anomaly.curvature",
    "dense_armor.anomaly.deviation": "dense_armor.utility.online.anomaly.deviation",
    "dense_armor.anomaly.filters": "dense_armor.utility.online.anomaly.filters",
    "dense_armor.anomaly.mahalanobis": "dense_armor.utility.online.anomaly.mahalanobis",
    "dense_armor.anomaly.one_sided": "dense_armor.utility.online.anomaly.one_sided",
    "dense_armor.anomaly.resonance_search": "dense_armor.utility.online.anomaly.resonance_search",
    "dense_armor.anomaly.robust_filters": "dense_armor.utility.online.anomaly.robust_filters",
    "dense_armor.anomaly.streaming": "dense_armor.utility.online.anomaly.streaming",
    "dense_armor.control": "dense_armor.utility.robot.control",
    "dense_armor.control.cbf_filter": "dense_armor.utility.robot.control.cbf_filter",
    "dense_armor.control.kinematic_controller": "dense_armor.utility.robot.control.kinematic_controller",
    "dense_armor.control.rate_limiter": "dense_armor.utility.robot.control.rate_limiter",
    "dense_armor.control.trajectory": "dense_armor.utility.robot.control.trajectory",
    "dense_armor.drift": "dense_armor.utility.online.drift",
    "dense_armor.drift.cusum": "dense_armor.utility.online.drift.cusum",
    "dense_armor.drift.detector": "dense_armor.utility.online.drift.detector",
    "dense_armor.dynamics.online_dynamics": "dense_armor.utility.online.learn.online_dynamics",
    "dense_armor.learn": "dense_armor.utility.online.learn",
    "dense_armor.learn.calibration": "dense_armor.utility.online.learn.calibration",
    "dense_armor.learn.metric_learning": "dense_armor.utility.online.learn.metric_learning",
    "dense_armor.learn.online_classifiers": "dense_armor.utility.online.learn.online_classifiers",
    "dense_armor.learn.online_dynamics": "dense_armor.utility.online.learn.online_dynamics",
    "dense_armor.misc": "dense_armor.utility.signals.stats",
    "dense_armor.misc.diagnostic": "dense_armor.utility.evaluation.evaluate.diagnostic",
    "dense_armor.misc.iodat": "dense_armor.utility.signals.stream.iodat",
    "dense_armor.misc.metro": "dense_armor.utility.signals.preprocessing.metro",
    "dense_armor.protect": "dense_armor.utility.online.protect",
    "dense_armor.protect.arbiter": "dense_armor.utility.online.protect.arbiter",
    "dense_armor.protect.healing": "dense_armor.utility.online.protect.healing",
    "dense_armor.protect.orca": "dense_armor.utility.online.protect.orca",
    "dense_armor.protect.stable_frame_filter": "dense_armor.utility.online.protect.stable_frame_filter",
    "dense_armor.protect.streaming_arbiter": "dense_armor.utility.online.protect.streaming_arbiter",
    "dense_armor.utility.arbiter": "dense_armor.utility.online.protect.arbiter",
    "dense_armor.utility.calibration": "dense_armor.utility.online.learn.calibration",
    "dense_armor.utility.cbf_filter": "dense_armor.utility.robot.control.cbf_filter",
    "dense_armor.utility.curvature": "dense_armor.utility.online.anomaly.curvature",
    "dense_armor.utility.cusum": "dense_armor.utility.online.drift.cusum",
    "dense_armor.utility.diagnostic": "dense_armor.utility.evaluation.evaluate.diagnostic",
    "dense_armor.utility.healing": "dense_armor.utility.online.protect.healing",
    "dense_armor.utility.iodat": "dense_armor.utility.signals.stream.iodat",
    "dense_armor.utility.kinematic_controller": "dense_armor.utility.robot.control.kinematic_controller",
    "dense_armor.utility.metric_learning": "dense_armor.utility.online.learn.metric_learning",
    "dense_armor.utility.metro": "dense_armor.utility.signals.preprocessing.metro",
    "dense_armor.utility.one_sided": "dense_armor.utility.online.anomaly.one_sided",
    "dense_armor.utility.online_classifiers": "dense_armor.utility.online.learn.online_classifiers",
    "dense_armor.utility.orca": "dense_armor.utility.online.protect.orca",
    "dense_armor.utility.rate_limiter": "dense_armor.utility.robot.control.rate_limiter",
    "dense_armor.utility.resonance_search": "dense_armor.utility.online.anomaly.resonance_search",
    "dense_armor.utility.river_anomaly": "dense_armor.utility.online.anomaly.deviation",
    "dense_armor.utility.river_drift": "dense_armor.utility.online.drift.detector",
    "dense_armor.utility.robust_filters": "dense_armor.utility.online.anomaly.robust_filters",
    "dense_armor.utility.stable_frame_filter": "dense_armor.utility.online.protect.stable_frame_filter",
    "dense_armor.utility.streaming": "dense_armor.utility.online.anomaly.streaming",
    "dense_armor.utility.streaming_arbiter": "dense_armor.utility.online.protect.streaming_arbiter",
    "dense_armor.utility.streaming_filters": "dense_armor.utility.online.anomaly.filters",
    "dense_armor.utility.streaming_mahalanobis": "dense_armor.utility.online.anomaly.mahalanobis",
    "dense_armor.utility.trajectory": "dense_armor.utility.robot.control.trajectory",
}

_FROM_MISC = {
    "sdft": "dense_armor.utility.signals.stats.sdft",
    "skyline": "dense_armor.utility.signals.stats.skyline",
    "metro": "dense_armor.utility.signals.preprocessing.metro",
    "diagnostic": "dense_armor.utility.evaluation.evaluate.diagnostic",
    "iodat": "dense_armor.utility.signals.stream.iodat",
}
for _prefix in ("dense_armor.utility.misc", "dense_armor.utility.infrastructure.misc"):
    ALIASES[_prefix] = "dense_armor.utility.signals.stats"
    for _name, _new in _FROM_MISC.items():
        ALIASES[f"{_prefix}.{_name}"] = _new


UTILITY_GROUPS = {
    "online": (
        "learn",
        "optim",
        "linear",
        "timeseries",
        "cluster",
        "tree",
        "forest",
        "ensemble",
        "bandit",
        "active",
        "facto",
        "reco",
        "rules",
        "bayes",
        "neighbors",
        "multiclass",
        "multioutput",
        "proba",
        "anomaly",
        "drift",
        "protect",
    ),
    "deep": ("neural",),
    "signals": ("stream", "stats", "preprocessing", "vision", "datasets"),
    "robot": ("control",),
    "evaluation": ("evaluate", "metrics"),
    "infrastructure": ("compose", "data"),
    "finance": (),
    "audio": (),
}
_GROUP_OF = {
    folder: group for group, folders in UTILITY_GROUPS.items() for folder in folders
}


def _target(name):
    if name in ALIASES:
        return ALIASES[name]
    prefix = "dense_armor.utility."
    if name.startswith(prefix):
        folder, dot, rest = name[len(prefix) :].partition(".")
        if folder in _GROUP_OF:
            return prefix + _GROUP_OF[folder] + "." + folder + dot + rest
    return None


class _AliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if _target(name) is not None:
            return importlib.util.spec_from_loader(name, self)
        return None

    def create_module(self, spec):
        module = importlib.import_module(_target(spec.name))
        sys.modules[spec.name] = module
        return module

    def exec_module(self, module):
        pass


def install():
    if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _AliasFinder())
