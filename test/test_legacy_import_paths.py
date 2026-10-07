"""Old import paths (dense_armor.utility.*, dense_armor.dynamics.online_dynamics) stay valid."""
import importlib

import pytest

PAIRS = [
    ("utility.river_anomaly", "anomaly.deviation"),
    ("utility.streaming_filters", "anomaly.filters"),
    ("utility.streaming_mahalanobis", "anomaly.mahalanobis"),
    ("utility.robust_filters", "anomaly.robust_filters"),
    ("utility.streaming", "anomaly.streaming"),
    ("utility.one_sided", "anomaly.one_sided"),
    ("utility.curvature", "anomaly.curvature"),
    ("utility.resonance_search", "anomaly.resonance_search"),
    ("utility.cusum", "drift.cusum"),
    ("utility.river_drift", "drift.detector"),
    ("utility.arbiter", "protect.arbiter"),
    ("utility.streaming_arbiter", "protect.streaming_arbiter"),
    ("utility.orca", "protect.orca"),
    ("utility.healing", "protect.healing"),
    ("utility.stable_frame_filter", "protect.stable_frame_filter"),
    ("utility.calibration", "learn.calibration"),
    ("utility.metric_learning", "learn.metric_learning"),
    ("utility.online_classifiers", "learn.online_classifiers"),
    ("utility.cbf_filter", "control.cbf_filter"),
    ("utility.kinematic_controller", "control.kinematic_controller"),
    ("utility.rate_limiter", "control.rate_limiter"),
    ("utility.trajectory", "control.trajectory"),
    ("utility.anwav", "misc.anwav"),
    ("utility.collatz", "misc.collatz"),
    ("utility.metro", "misc.metro"),
    ("utility.iodat", "misc.iodat"),
    ("utility.diagnostic", "misc.diagnostic"),
    ("dynamics.online_dynamics", "learn.online_dynamics"),
]


@pytest.mark.parametrize("old,new", PAIRS)
def test_old_path_is_the_same_module(old, new):
    try:
        mod_new = importlib.import_module(f"dense_armor.{new}")
    except ModuleNotFoundError as exc:
        pytest.skip(f"optional dependency missing: {exc.name}")
    assert importlib.import_module(f"dense_armor.{old}") is mod_new
