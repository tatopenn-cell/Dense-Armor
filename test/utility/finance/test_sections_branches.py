"""The branches of sections.py that the pages do not reach: a piece that fails, a cache, a missing file."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from dashboard import sections as sec

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _fresh_caches(monkeypatch):
    monkeypatch.setattr(sec, "_RISK_CACHE", {})
    monkeypatch.setattr(sec, "_FEAT_CACHE", {})
    monkeypatch.setattr(sec, "_SIG_CACHE", {})
    monkeypatch.chdir(REPO)


@pytest.fixture
def fake_module(monkeypatch):
    mod = types.ModuleType("fake_pieces")

    class Good:
        mean_ = 1.0
        value = 2.0
        count_ = 3.0

        def learn_one(self, x, y=None):
            pass

        def transform_one(self, x):
            return {"ret": x["ret"] * 2}

        def quantile(self, q):
            return 0.5

        def predict_one(self, x):
            return 1

    class Silent(Good):
        def transform_one(self, x):
            return {}

    class Failing:
        def learn_one(self, *a, **k):
            raise ValueError("learn")

        def transform_one(self, x):
            raise ValueError("transform")

        def predict_one(self, x):
            raise ValueError("predict")

        def quantile(self, q):
            raise ValueError("quantile")

    class NeedsDetector(Good):
        def __init__(self, drift_detector):
            self.drift_detector = drift_detector

    class Broken:
        def __init__(self):
            raise RuntimeError("cannot build")

    for cls in (Good, Silent, Failing, NeedsDetector, Broken):
        setattr(mod, cls.__name__, cls)
    monkeypatch.setitem(sys.modules, "fake_pieces", mod)
    return mod


@pytest.mark.parametrize("extract,expect", [
    ("transform.ret", True), ("running_mean", True), ("running_median", True), ("quantile_0.5", True),
    ("count", True), ("value", True), ("corr", True), ("unknown", True),
])
def test_a_feature_series_reads_each_kind_of_value(fake_module, extract, expect):
    out = sec._feature_series("fake_pieces", "Good", extract)
    assert out["n"] > 0 and out["svg"].startswith("<svg") and out["error"] is None
    assert sec._feature_series("fake_pieces", "Good", extract) is out


def test_a_feature_series_with_no_transform_gives_zeros_and_a_failing_piece_gives_its_error(fake_module):
    silent = sec._feature_series("fake_pieces", "Silent", "transform.ret")
    assert silent["n"] > 0 and silent["error"] is None
    failing = sec._feature_series("fake_pieces", "Failing", "transform.ret")
    assert failing["error"] == "ValueError: transform"
    quantile = sec._feature_series("fake_pieces", "Failing", "quantile_0.5")
    assert quantile["error"] == "ValueError: quantile"


def test_a_feature_series_fills_the_missing_argument_and_reports_a_class_that_cannot_be_built(fake_module):
    assert sec._feature_series("fake_pieces", "NeedsDetector", "value")["n"] > 0
    broken = sec._feature_series("fake_pieces", "Broken", "value")
    assert broken["n"] == 0 and "cannot build" in broken["error"]
    missing = sec._feature_series("no_such_module_anywhere", "X", "value")
    assert missing["error"].startswith("ModuleNotFoundError")


def test_a_signal_preview_trains_reports_and_caches(fake_module):
    out = sec._sig_preview("fake_pieces", "Good")
    assert out["n"] > 0 and out["final_acc"] is not None and out["error"] is None
    assert sec._sig_preview("fake_pieces", "Good") is out
    failing = sec._sig_preview("fake_pieces", "Failing")
    assert failing["n"] > 0 and failing["final_acc"] == 0.0
    assert sec._sig_preview("fake_pieces", "NeedsDetector")["n"] > 0
    assert "cannot build" in sec._sig_preview("fake_pieces", "Broken")["error"]


def test_a_signal_preview_with_no_stream_is_empty(fake_module, monkeypatch):
    monkeypatch.setattr(sec, "_gspc_classification_stream", lambda n=200: ([], []))
    out = sec._sig_preview("fake_pieces", "Good")
    assert out == {"svg": "", "n": 0, "error": None, "final_acc": None}


def test_the_default_arguments_that_a_piece_requires_are_filled_by_name():
    class Needs:
        def __init__(self, drift_detector, warning_detector, model, estimator, classifier, optimizer,
                     loss, link, metric_learner, already=1, none_default=None, plain=0):
            pass

    out = sec._auto_init_kwargs(Needs, {"plain": 5})
    assert set(out) == {"drift_detector", "warning_detector", "model", "estimator", "classifier",
                        "optimizer", "loss", "link", "metric_learner", "plain"}
    assert out["loss"] == "log_loss" and out["link"] == "logistic" and out["plain"] == 5
    assert "already" not in out


def test_a_default_that_cannot_be_imported_is_left_out(monkeypatch):
    class Needs:
        def __init__(self, optimizer, metric_learner):
            pass

    monkeypatch.setitem(sys.modules, "dense_armor.utility.online.optim.sgd", None)
    monkeypatch.setitem(sys.modules, "dense_armor.utility.online.learn.metric_learning", None)
    assert sec._auto_init_kwargs(Needs) == {}


def test_the_role_attributes_of_a_piece_with_good_and_with_bad_values():
    class Health:
        value = "ok"

    class Good:
        budget_s = 0.25
        memory_class = "O(1)"
        score_is_inverted = True
        drift_detected = False
        warning_detected = True
        n_blocked_ = 3
        health = Health()
        coverage = 0.9
        alpha_t_ = 0.1

        class units:
            inputs = {"a": "m"}
            outputs = {}

        def state_dict(self):
            return {"b": 1, "a": 2}

    out = sec._role_attributes(Good())
    assert out["budget_s"] == 0.25 and out["memory_class"] == "O(1)" and out["n_blocked"] == 3
    assert out["score_is_inverted"] is True and out["warning_detected"] is True
    assert out["health"] == "ok" and out["coverage"] == 0.9 and out["alpha_t"] == 0.1
    assert out["state_keys"] == ["a", "b"]
    assert out["units"] == {"inputs": {"a": "m"}, "outputs": {}}

    class Bad:
        budget_s = "x"
        n_blocked_ = "x"
        coverage = "x"
        alpha_t_ = "x"
        health = "plain"

        def state_dict(self):
            raise RuntimeError("no")

    bad = sec._role_attributes(Bad())
    assert "budget_s" not in bad and "n_blocked" not in bad and "coverage" not in bad
    assert "alpha_t" not in bad and "state_keys" not in bad and bad["health"] == "plain"
    assert sec._role_attributes(None) == {}


def test_the_returns_of_a_csv_file_and_of_a_stream(tmp_path):
    assert sec._returns_preview(str(tmp_path / "missing.csv")) == []
    short = tmp_path / "short.csv"
    short.write_text("Date,Close\n2020-01-01,10\n2020-01-02,11\n", encoding="utf-8")
    assert sec._returns_preview(str(short)) == []
    full = tmp_path / "full.csv"
    full.write_text("Date,Close\n2020-01-01,10\nbad\n2020-01-02,x\n2020-01-03,11\n2020-01-04,12\n2020-01-05,13\n",
                    encoding="utf-8")
    rets = sec._returns_preview(str(full), n=2)
    assert len(rets) == 2 and all(isinstance(r, float) for r in rets)
    assert len(sec._driftstream_preview("sudden", n=40)) == 40


def test_a_stream_preview_without_the_datasets_module_is_empty(monkeypatch):
    monkeypatch.setitem(sys.modules, "dense_armor.utility.signals.datasets", None)
    assert sec._driftstream_preview("sudden") == []


def test_a_risk_preview_for_other_kinds_missing_data_and_unknown_pieces(monkeypatch):
    assert sec._risk_preview("other", "x")["svg"] == ""
    assert "KeyError" in sec._risk_preview("anomaly", "no_such_piece")["error"]
    monkeypatch.setattr(sec, "_returns_preview", lambda path, n=100: [])
    assert sec._risk_preview("drift", "adwin") == {"svg": "", "n": 0, "error": None}


def test_a_risk_preview_of_a_detector_that_fails_in_each_step(monkeypatch):
    class Flaky:
        def __init__(self, feature):
            self.feature = feature

        def score_one(self, r):
            raise ValueError("score")

        def learn_one(self, r):
            raise ValueError("learn")

    monkeypatch.setattr("dense_armor.utility.online.anomaly.lof.LocalOutlierFactor", Flaky)
    out = sec._risk_preview("anomaly", "lof")
    assert out["error"] is None and out["n"] == 0

    class Never:
        def __init__(self):
            raise RuntimeError("never")

    monkeypatch.setattr("dense_armor.utility.online.anomaly.hst.HalfSpaceTrees", Never)
    assert "cannot instantiate Never" in sec._risk_preview("anomaly", "hst")["error"]


def test_a_risk_preview_of_a_drift_detector_counts_and_swallows(monkeypatch):
    class Always:
        _drift_detected = True

        def update(self, v):
            pass

    class Raises:
        def update(self, v):
            raise ValueError("update")

    monkeypatch.setattr("dense_armor.utility.online.drift.adwin.ADWIN", Always)
    monkeypatch.setattr("dense_armor.utility.online.drift.kswin.KSWIN", Raises)
    always = sec._risk_preview("drift", "adwin")
    assert always["n"] > 0 and always["svg"].startswith("<svg")
    assert sec._risk_preview("drift", "kswin")["n"] > 0


def test_the_pages_with_an_unknown_category_and_with_sketches_that_have_no_series(monkeypatch):
    def render(template, **kw):
        return kw

    ctx = {}
    for page, expected in ((sec.features_page, 0), (sec.signal_page, 0), (sec.risk_page, 1)):
        out = page("no_such_category", render, ctx)
        assert out["active_cat"] == expected
    monkeypatch.setattr(sec, "_feature_series", lambda *a, **k: {"svg": "", "n": 0, "error": None})
    out = sec.features_page("sketches", render, ctx)
    assert out["pieces"] and all(p["n"] == 0 for p in out["pieces"])


def test_the_meta_of_a_piece_and_its_neighbours():
    assert sec.piece_meta("signal", "trees", "no_such")["ok"] is False
    meta = sec.piece_meta("signal", "trees", "hoeffding")
    assert meta["ok"] is True and meta["name"] == "HoeffdingTreeClassifier"
    assert meta["params"][0]["name"] == "dataset"
    assert sec._piece_prev_next("trees", "no_such") == (None, None)


def test_a_piece_runs_with_typed_parameters_and_with_a_bad_one():
    meta = sec.piece_meta("signal", "trees", "hoeffding")
    typed = {p["name"]: p["default"] for p in meta["params"][1:4] if p["default"] != ""}
    out = sec.piece_detail("signal", "trees", "hoeffding", dict(typed, dataset="finance"))
    assert out["ok"] is True, out
    bad = sec.piece_detail("signal", "trees", "hoeffding", {"grace_period": "not-a-number"})
    assert isinstance(bad, dict) and "ok" in bad


def _csv_tree(tmp_path, lines):
    folder = tmp_path / "test" / "fixtures" / "finance"
    folder.mkdir(parents=True)
    (folder / "GSPC.csv").write_text(chr(10).join(lines) + chr(10), encoding="utf-8")


def test_the_classification_stream_needs_the_file_and_enough_valid_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(sec, "FIXTURES", tmp_path / "test" / "fixtures" / "finance")
    assert sec._gspc_classification_stream(50) == ([], [])
    _csv_tree(tmp_path, ["Date,Close", "short", "2020,notafloat", "2020,10", "2020,11"])
    assert sec._gspc_classification_stream(50) == ([], [])


def test_the_defaults_survive_a_signature_that_cannot_be_read(monkeypatch):
    import inspect

    def refuse(obj):
        raise ValueError("no signature")

    monkeypatch.setattr(inspect, "signature", refuse)
    assert sec._auto_init_kwargs(object, {"a": 1}) == {"a": 1}


def test_a_detector_that_needs_a_default_is_built_from_the_defaults(monkeypatch):
    class Needs:
        def __init__(self, drift_detector):
            self.drift_detector = drift_detector

        def score_one(self, r):
            return 0.5

        def learn_one(self, r):
            pass

    monkeypatch.setattr("dense_armor.utility.online.anomaly.hst.HalfSpaceTrees", Needs)
    out = sec._risk_preview("anomaly", "hst")
    assert out["error"] is None and out["n"] > 0


def test_a_health_that_fails_the_second_time_is_left_out():
    class Twice:
        calls = 0

        @property
        def health(self):
            Twice.calls += 1
            if Twice.calls > 1:
                raise RuntimeError("second read")
            return "fine"

    assert "health" not in sec._role_attributes(Twice())


def test_the_neighbours_of_a_piece_need_the_table_in_the_source(tmp_path, monkeypatch):
    empty = tmp_path / "empty.py"
    empty.write_text("x = 1", encoding="utf-8")
    monkeypatch.setattr(sec, "__file__", str(empty))
    assert sec._piece_prev_next("trees", "hoeffding") == (None, None)


XS = [{"ret": ((i % 7) - 3) / 100.0} for i in range(300)]
YS = [i % 2 for i in range(300)]


@pytest.fixture
def generic(monkeypatch):
    mod = types.ModuleType("fake_generic")

    class Classifier:
        budget_s = "bad"
        memory_class = "O(1)"
        dtype = "float64"
        threshold = "bad"
        n_nodes_ = 0

        def __init__(self, a=1):
            self.a = a

        def predict_one(self, x):
            return 1

        def learn_one(self, x):
            pass

        def explain_one(self, x):
            raise RuntimeError("explain")

        def predict_unaggregated_one(self, x):
            raise RuntimeError("votes")

    class Classifier2(Classifier):
        budget_s = 0.5
        dtype = None
        threshold = 2.5

        @property
        def _dtype(self):
            raise RuntimeError("dtype")

        def explain_one(self, x):
            return [{"feature": "ret", "threshold": 0.5, "side": "left"}]

        def predict_unaggregated_one(self, x):
            return [object()]

    class Scorer:
        inits = 0

        def __init__(self, feature_keys=None):
            Scorer.inits += 1
            if Scorer.inits > 1:
                raise RuntimeError("second")

        def score_one(self, x):
            return 1.0

        def learn_one(self, x, y):
            pass

    class ScorerFails:
        def score_one(self, x):
            raise ValueError("score")

        def learn_one(self, x):
            pass

    class Updater:
        ALARMS = ()

        def __init__(self):
            self.i = 0
            self._drift_detected = False

        def update(self, v):
            self.i += 1
            if self.i == 3:
                raise ValueError("update")
            self._drift_detected = (self.i - 1) in type(self).ALARMS

    class Sudden(Updater):
        ALARMS = tuple(range(10, 300, 10))

    class Gradual(Updater):
        ALARMS = (1, 2, 3, 60, 250)

    class Never:
        def __init__(self, **kw):
            raise RuntimeError("never")

    for cls in (Classifier, Classifier2, Scorer, ScorerFails, Sudden, Gradual, Never):
        setattr(mod, cls.__name__, cls)
        monkeypatch.setitem(sec._SIGNAL_CLASSES_EXT, ("fake", cls.__name__), ("fake_generic", cls.__name__))
    monkeypatch.setitem(sys.modules, "fake_generic", mod)
    monkeypatch.setattr(sec, "_gspc_classification_stream", lambda n=500: (XS, YS))
    return mod


def _run(name, params=None):
    return sec.piece_detail("signal", "fake", name, params or {})


def test_a_generic_classifier_reports_its_declarations_and_what_it_cannot_explain(generic):
    out = _run("Classifier", {"a": "2"})
    m = out["metrics"]
    assert out["ok"] is True and m["kind"] == "classifier" and m["dtype"] == "float64"
    assert "budget_s" not in m and "explain" not in m and "threshold" not in m
    assert m["per_class"] and m["confusion"]


def test_a_generic_classifier_with_a_dtype_that_fails_and_a_path_and_votes(generic):
    m = _run("Classifier2")["metrics"]
    assert m["dtype"] == "float32" and m["budget_s"] == 0.5 and m["threshold"] == 2.5
    assert m["explain"] == [{"feature": "ret", "threshold": 0.5, "side": "left"}]
    assert "tree_votes" not in m


def test_a_generic_anomaly_scorer_builds_the_histogram_of_a_flat_score(generic):
    out = _run("Scorer")
    m = out["metrics"]
    assert out["ok"] is True and m["kind"] == "anomaly"
    assert m["histogram"]["counts"] and m["score_mean"] == 1.0
    fails = _run("ScorerFails")["metrics"]
    assert fails["kind"] == "anomaly" and "histogram" not in fails


def test_a_generic_drift_detector_is_called_sudden_or_gradual_from_its_alarms(generic):
    sudden = _run("Sudden")["metrics"]
    assert sudden["kind"] == "drift" and sudden["drift_type"] == "sudden"
    gradual = _run("Gradual")["metrics"]
    assert gradual["drift_type"] == "gradual" and gradual["alarms"] == 5


def test_a_generic_piece_that_does_not_build_or_has_no_data_or_a_bad_parameter(generic, monkeypatch):
    out = _run("Never")
    assert out["ok"] is False and out["error"].startswith("init: RuntimeError")

    def boom(name, raw, typ):
        raise ValueError("coerce")

    with monkeypatch.context() as m:
        m.setattr(sec, "_coerce", boom)
        assert _run("Classifier", {"a": "2"})["ok"] is True
    monkeypatch.setattr(sec, "_gspc_classification_stream", lambda n=500: ([], []))
    assert _run("Classifier") == {"ok": False, "error": "no GSPC data"}


def test_the_hoeffding_piece_with_an_empty_dataset_or_a_tree_that_fails(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(sec, "_finance_features", lambda n=1500, **k: ([], []))
        empty = sec.piece_detail("signal", "trees", "hoeffding", {"dataset": "finance"})
        assert empty == {"ok": False, "error": "dataset vuoto"}
    from dense_armor.utility.online.tree.hoeffding import HoeffdingTreeClassifier

    def boom(self, x):
        raise RuntimeError("proba")

    monkeypatch.setattr(HoeffdingTreeClassifier, "predict_proba_one", boom)
    out = sec.piece_detail("signal", "trees", "hoeffding", {"dataset": "manual"})
    assert out["ok"] is True


def test_the_finance_features_need_enough_returns(monkeypatch):
    monkeypatch.setattr(sec, "_returns_preview", lambda path, n=100: [0.1] * 10)
    assert sec._finance_features(n=100) == ([], [])


def test_the_signal_page_when_a_default_or_a_preview_fails(monkeypatch):
    def render(template, **kw):
        return kw

    with monkeypatch.context() as m:
        m.setitem(sys.modules, "dense_armor.utility.online.drift.adwin", None)
        m.setattr(sec, "_sig_preview", lambda *a, **k: {"svg": "<svg></svg>", "n": 1, "final_acc": 0.5, "error": None})
        assert sec.signal_page("trees", render, {})["pieces"]

    def boom(*a, **k):
        raise RuntimeError("preview")

    with monkeypatch.context() as m:
        m.setattr(sec, "_sig_preview", boom)
        assert sec.signal_page("trees", render, {})["pieces"]


def test_the_threshold_of_an_anomaly_scorer_without_one_is_learned_from_its_scores(monkeypatch):
    mod = types.ModuleType("fake_spiky")

    class Spiky:
        calls = 0

        def score_one(self, x):
            Spiky.calls += 1
            return 10.0 if Spiky.calls == 100 else 1.0 + (Spiky.calls % 2) * 0.1

        def learn_one(self, x):
            pass

    mod.Spiky = Spiky
    monkeypatch.setitem(sys.modules, "fake_spiky", mod)
    monkeypatch.setitem(sec._SIGNAL_CLASSES_EXT, ("fake", "Spiky"), ("fake_spiky", "Spiky"))
    monkeypatch.setattr(sec, "_gspc_classification_stream", lambda n=500: (XS, YS))
    m = sec.piece_detail("signal", "fake", "Spiky", {})["metrics"]
    assert m["kind"] == "anomaly" and m["threshold_rule"].startswith("mean + 3")
    assert 1.1 < m["threshold"] < 10.0
    assert m["alerts_above_threshold"] == 1


def test_a_piece_that_predicts_nothing_is_a_statistic_with_no_accuracy(monkeypatch):
    mod = types.ModuleType("fake_stat")

    class Counter:
        def learn_one(self, x):
            pass

    mod.Counter = Counter
    monkeypatch.setitem(sys.modules, "fake_stat", mod)
    monkeypatch.setitem(sec._SIGNAL_CLASSES_EXT, ("fake", "Counter"), ("fake_stat", "Counter"))
    monkeypatch.setattr(sec, "_gspc_classification_stream", lambda n=500: (XS, YS))
    out = sec.piece_detail("signal", "fake", "Counter", {})
    assert out["ok"] is True and out["metrics"]["kind"] == "stat" and out["series"] == []
