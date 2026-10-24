"""Fixes of the defects found by reading: batch tree, linear models, metric learning, misc."""

import numpy as np
import pytest

from dense_armor.utility.drift import (
    ADWIN,
    KSWIN,
    BayesianOnlineChangepoint,
    CUSUMDriftDetector,
    PageHinkley,
)
from dense_armor.utility.drift.cusum import cusum_detector
from dense_armor.utility.evaluate import evaluate_events, progressive_val_score
from dense_armor.utility.evaluate.model_selection import OnlineModelSelection
from dense_armor.utility.learn.calibration import OnlinePlattScaling
from dense_armor.utility.learn.metric_learning import LEGO
from dense_armor.utility.learn.online_dynamics import RecursiveLeastSquares
from dense_armor.utility.linear.bayes import BayesianLinearRegression
from dense_armor.utility.linear.glm import LogisticRegression
from dense_armor.utility.metrics import Accuracy
from dense_armor.utility.misc.diagnostic import diag
from dense_armor.utility.misc.metro import Metro
from dense_armor.utility.optim import SGD
from dense_armor.utility.tree.batch import BatchDecisionTreeClassifier


class _Const:
    def learn_one(self, x, y, t=None):
        return self

    def predict_one(self, x, t=None):
        return 1


def test_current_best_before_the_run_raises():
    sel = OnlineModelSelection([_Const(), _Const()], Accuracy)
    with pytest.raises(ValueError):
        sel.current_best()
    sel.run([({}, 1)] * 3)
    assert sel.current_best() in (0, 1)


def test_a_negative_delay_raises():
    with pytest.raises(ValueError):
        progressive_val_score([({}, 1)], _Const(), Accuracy(), delay=-5)


def test_a_delay_in_seconds_needs_timestamps():
    with pytest.raises(ValueError):
        progressive_val_score([({}, 1)] * 3, _Const(), Accuracy(), delay=1.5)


def test_an_anomaly_detector_without_t_is_called_without_it():
    class Det:
        score_threshold = 0.5

        def score_one(self, x):
            return 1.0

        def learn_one(self, x):
            return self

    stream = [(float(i), {"a": 1.0}) for i in range(5)]
    assert evaluate_events(stream, Det(), [(1.0, 3.0)])["missed"] == 0


def test_the_drift_package_exports_every_detector():
    names = (ADWIN, KSWIN, BayesianOnlineChangepoint, CUSUMDriftDetector, PageHinkley)
    assert all(callable(c) for c in names)


def test_adwin_range_follows_the_window_after_an_eviction():
    adw = ADWIN(max_buckets=64)
    for v in [0.0] * 4 + [10.0] * 4:
        adw._add(v)
    for _ in range(4):
        adw._remove_oldest()
    assert (adw._min, adw._max) == (10.0, 10.0)


def test_bocpd_counts_missing_from_the_start():
    det = BayesianOnlineChangepoint()
    assert det.n_missing_ == 0
    det.update(float("nan"))
    det.update(float("inf"))
    assert det.n_missing_ == 2


def test_cusum_fixed_reference_refuses_a_nan_in_the_first_window():
    x = np.concatenate([[float("nan")], np.zeros(5000)])
    with pytest.raises(ValueError):
        cusum_detector(x, reference="fixed")


def test_batch_fit_refuses_a_row_with_a_missing_feature():
    with pytest.raises(ValueError):
        BatchDecisionTreeClassifier().fit([{"a": 1.0, "b": 2.0}, {"a": 1.0}], [0, 1])


def test_batch_fit_refuses_nan_and_inf():
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            BatchDecisionTreeClassifier().fit([{"a": 0.0}, {"a": bad}], [0, 1])


def test_batch_fit_refuses_a_value_that_is_not_a_number():
    with pytest.raises(ValueError):
        BatchDecisionTreeClassifier().fit([{"a": 0.0}, {"a": "x"}], [0, 1])


def test_batch_has_no_unused_class_set():
    clf = BatchDecisionTreeClassifier().fit([{"a": 0.0}, {"a": 1.0}], [0, 1])
    assert not hasattr(clf, "_classes_")


def test_logistic_treats_any_nonzero_label_as_one():
    one, two = LogisticRegression(SGD(0.1)), LogisticRegression(SGD(0.1))
    x = {"a": 1.0, "b": -0.5}
    one.learn_one(x, 1)
    two.learn_one(x, 2)
    assert np.array_equal(one._w, two._w)


def test_logistic_and_bayes_skip_an_infinite_value():
    log = LogisticRegression(SGD(0.1))
    log.learn_one({"a": float("inf")}, 1)
    assert log.n_missing == 1 and log._w is None
    bay = BayesianLinearRegression()
    bay.learn_one({"a": 1.0}, float("inf"))
    bay.learn_one({"a": float("inf")}, 1.0)
    assert bay.n_missing == 2 and bay._m is None


def test_rls_counts_a_missing_key_as_zero():
    rls = RecursiveLeastSquares(feature_keys=["a", "b"])
    rls.learn_one({"a": 1.0, "b": 2.0}, 3.0)
    rls.learn_one({"a": 1.0}, 1.0)
    assert np.isfinite(rls.predict_one({"b": 1.0}))


def test_lego_percentile_targets_refuses_one_class():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        LEGO.percentile_targets(X, [1, 1, 1])
    with pytest.raises(ValueError):
        LEGO.percentile_targets(X, [0, 1, 2])
    near, far = LEGO.percentile_targets(X, [0, 0, 1])
    assert near <= far


def test_platt_asks_the_base_once_per_call():
    class Base:
        calls = 0

        def predict_proba_one(self, x, **kw):
            Base.calls += 1
            return {False: 0.3, True: 0.7}

        def learn_one(self, x, y, **kw):
            return self

    cal = OnlinePlattScaling(Base())
    cal.predict_proba_one({"x": 1.0})
    assert Base.calls == 1
    cal.learn_one({"x": 1.0}, 1)
    assert Base.calls == 2


def test_metro_keeps_the_sign():
    m = Metro()
    for v in (-1e-9, 1e-9, -3.5e-12):
        enc, fact = m.enc(v)
        assert np.sign(enc) == np.sign(v)
        assert m.dec(enc, fact) == pytest.approx(v)
    assert m.enc(0.0) == (0.0, 1.0)


def test_diag_scales_by_dtype_and_refuses_mixed_arguments():
    sig = np.sin(np.linspace(0.0, 20.0, 2000)).astype(np.float64)
    as16 = (sig * 32767).astype(np.int16)
    as32 = (sig * 2147483647).astype(np.int32)
    same = diag(as32, as16)
    ref = diag(sig.astype(np.float32), sig.astype(np.float32))
    assert same is not None and ref is not None
    with pytest.raises(TypeError):
        diag("a.wav", sig)


def test_control_validation():
    from dense_armor.utility.control.cbf_filter import (
        cbf_filtered_trajectory,
        cbf_safety_filter_live,
    )
    from dense_armor.utility.control.kinematic_controller import (
        kinematic_tracking_controller,
    )
    from dense_armor.utility.control.trajectory import quintic_trajectory

    with pytest.raises(ValueError):
        kinematic_tracking_controller([0.0, 1.0], [0.0], [0.0, 0.0], 1.0)
    with pytest.raises(ValueError):
        quintic_trajectory([0.0], [[1.0]], 1.0)
    with pytest.raises(ValueError):
        quintic_trajectory([0.0], [1.0], 0.0)
    with pytest.raises(ValueError):
        quintic_trajectory([0.0], [1.0], 1.0, n_samples=0)
    with pytest.raises(ValueError):
        cbf_filtered_trajectory(np.zeros(3), 5.0, 1.0, n_substeps=0)
    with pytest.raises(ValueError):
        cbf_safety_filter_live(0.0, 1.0, 0.1, 5.0, 1.0, n_substeps=0)


def test_func_transformer_whole():
    from dense_armor.roles import Signal
    from dense_armor.utility.compose import FuncTransformer

    s = Signal(values=[1.0, 2.0], names=["a", "b"], units=["m", "s"], t=0.0, missing=[False, True])
    out = FuncTransformer(lambda d: {k: v + 1.0 for k, v in d.items()}, whole=True).transform_one(s)
    assert list(out.units) == ["m", "s"]
    assert list(out.mask) == [False, True]
    with pytest.raises(TypeError):
        FuncTransformer(lambda d: [1.0], whole=True).transform_one(s)
    assert FuncTransformer(lambda v: v + 1.0, keys=()).transform_one({"a": 1.0}) == {"a": 1.0}


def test_kmeans_warmup_shape_mismatch():
    from dense_armor.utility.cluster.kmeans import OnlineKMeans

    km = OnlineKMeans(k=2, warmup=4)
    km.learn_one({"x": [0.0]})
    km.learn_one({"x": [0.0, 1.0]})
    assert km.n_missing_ == 1 and km.n_seen_ == 1


def test_bandit_fixes():
    from dense_armor.utility.bandit import BayesUCB
    from dense_armor.utility.bandit.evaluate import offline_replay

    p = BayesUCB(arms=[0], family="gaussian", estimate_var=True)
    xs = [1e6 + d for d in (-1.0, 0.0, 1.0, 2.0, -2.0)]
    for x in xs:
        p.update(0, x)
    assert abs(p.m2_[0] / (len(xs) - 1) - float(np.var(xs, ddof=1))) < 1e-6
    with pytest.raises(ValueError):
        offline_replay([({"x": 0.0}, 0, 1.0)], lambda c: 0, n_bootstrap=0)


def test_anomaly_missing_key():
    from dense_armor.utility.anomaly.deviation import StreamingDeviationScorer
    from dense_armor.utility.anomaly.filters import HampelFilter, HampelScorer
    from dense_armor.utility.anomaly.streaming import StreamingDeviationDetector

    s = HampelScorer(radius=5, feature="v")
    for v in range(10):
        s.learn_one({"v": float(v)})
    s.learn_one({"w": 1.0})
    s.learn_one({})
    s.learn_one({"v": float("nan")})
    assert s.n_missing == 3 and s.score_one({"w": 1.0}) == 0.0
    d = StreamingDeviationScorer(feature="v")
    d.learn_one({})
    assert d.n_missing == 1 and d.score_one({}) == 0.0
    f = HampelFilter(radius=5, feature="v")
    assert f.transform_one({"w": 2.0}) == {"w": 2.0}
    det = StreamingDeviationDetector(radius=5, ref_mult=2)
    for v in (1.0, 1.1, 0.9, 1.0, float("nan")):
        det.update(v)
    assert len(det._buffer) == 4


def test_stream_checks(tmp_path):
    from dense_armor.roles import Signal
    from dense_armor.utility.stream import iter_csv, merge_by_time

    p = tmp_path / "a.csv"
    p.write_text("a,t\n1,0\n2,1\n")
    with pytest.raises(ValueError):
        list(iter_csv(p, start=0.0))
    bad = [(Signal(values=[1.0], names=["a"], units=[""], t=float("nan")), None)]
    with pytest.raises(ValueError):
        list(merge_by_time(bad))


def test_stats_missing_and_extremes():
    from dense_armor.utility.stats.dependence import Autocorrelation
    from dense_armor.utility.stats.moments import RunningMoments, RunningMomentsVector
    from dense_armor.utility.stats.quantiles import TDigest
    from dense_armor.utility.stats.shrinkage import OnlineShrinkageCovariance
    from dense_armor.utility.stats.sketches import _hash64

    m = RunningMoments(feature="a")
    m.learn_many(np.array([1.0, np.nan, 2.0, np.inf]))
    assert m.n_missing_ == 2 and m.n_ == 2
    m.learn_one({"b": 1.0})
    assert m.n_missing_ == 3
    v = RunningMomentsVector(features=["a", "b"])
    v.learn_many({"a": np.array([1.0, 2.0])})
    assert v.channels_["b"].n_missing_ == 1
    ac = Autocorrelation(feature="a", lag=1)
    ac.learn_one({})
    assert ac.n_missing == 1
    td = TDigest()
    for x in np.random.default_rng(0).pareto(1.5, 2000):
        td.learn_one({"x": float(x)})
    assert td.quantile(1.0) == td.max_ and td.quantile(0.0) == td.min_
    sc = OnlineShrinkageCovariance(n_features=2)
    rng = np.random.default_rng(1)
    for i in range(60):
        sc.learn_one({"v0": float(rng.normal()), "v1": float("nan") if i % 2 else float(rng.normal())})
    assert sc.shrinkage()[1] >= 0.0
    assert _hash64("a", 2**33) == _hash64("a", 0)


@pytest.mark.parametrize("module,name", [
    ("moments", "RunningMoments"),
    ("moments", "EWStats"),
    ("quantiles", "DDSketch"),
    ("quantiles", "TDigest"),
    ("robust", "RollingMedian"),
])
def test_stats_empty_and_non_number_counted(module, name):
    import importlib

    est = getattr(importlib.import_module(f"dense_armor.utility.stats.{module}"), name)()
    est.learn_one({})
    est.learn_one({"a": "text"})
    est.learn_one({"a": None})
    assert est.n_missing == 3


@pytest.mark.parametrize("name", ["TukeyScorer", "ChauvenetScorer", "SigmaClipScorer", "HampelScorer"])
def test_filter_scorers_missing_key(name):
    from dense_armor.utility.anomaly import filters

    s = getattr(filters, name)(radius=5, feature="v")
    for v in range(10):
        s.learn_one({"v": float(v)})
    s.learn_one({})
    s.learn_one({"v": "text"})
    assert s.n_missing == 2
    expected = float("inf") if name == "ChauvenetScorer" else 0.0
    assert s.score_one({}) == expected
    assert len(s._buffer) == 10


def test_hampel_filter_missing_and_outlier():
    from dense_armor.utility.anomaly.filters import HampelFilter

    f = HampelFilter(radius=5, feature="v")
    for v in range(10):
        f.learn_one({"v": float(v)})
    assert f.transform_one({"w": 1.0}) == {"w": 1.0}
    assert f.n_missing == 1
    out = f.transform_one({"v": 1000.0, "z": 2.0})
    assert out["z"] == 2.0 and out["v"] < 20.0
    assert f.transform_one({"v": 4.5}) == {"v": 4.5}


def test_active_strategy_n_missing_property():
    from dense_armor.utility.active.strategies import RandomStrategy

    s = RandomStrategy(budget=0.5)
    assert s.n_missing == 0 and s.n_missing_ == 0


def test_platt_project_shrinks_a_long_vector():
    from dense_armor.utility.learn.calibration import OnlinePlattScaling

    p = OnlinePlattScaling(classifier=None, radius=0.5)
    out = p._project(np.array([5.0, 5.0]))
    assert float(out @ out) <= 0.25 + 1e-6


def test_diag_unsigned_and_sample_rate(tmp_path):
    from scipy.io import wavfile

    sig = (np.sin(np.linspace(0, 20, 400)) * 100 + 128).astype(np.uint8)
    assert diag(sig, sig) is not None
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    wavfile.write(a, 8000, sig)
    wavfile.write(b, 16000, sig)
    with pytest.raises(ValueError):
        diag(str(a), str(b))
    assert diag(str(a), str(a)) is not None


@pytest.mark.parametrize("name", ["CountMinSketch", "HyperLogLog", "BloomFilter", "SpaceSaving"])
def test_sketches_count_missing_key(name):
    from dense_armor.utility.stats import sketches

    s = getattr(sketches, name)(feature="a")
    s.learn_one({"a": "x"})
    s.learn_one({"b": "y"})
    assert s.n_missing == 1 and s.n_ == 1
    if hasattr(s, "merge"):
        assert s.merge(s).n_missing == 2
    if name != "HyperLogLog":
        assert s.transform_one({"b": 1}) == {}
        assert s.n_missing == 2


def test_ffm_orphan_feature_allocates_nothing():
    from dense_armor.utility.facto.ffm import FFMClassifier

    m = FFMClassifier(fields={"a": "f1", "b": "f2"}, k=3, feature_keys=["a", "b", "z"])
    m.learn_one({"a": 1.0, "b": 1.0, "z": 5.0}, 1)
    assert m._w.shape[0] == 2
    assert 0.0 <= m.predict_proba_one({"a": 1.0, "b": 1.0, "z": 5.0}).get(1, 0.5) <= 1.0
