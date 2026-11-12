"""The composition of the finance chain: the catalogue, the probe of a piece, the data and a run."""

from __future__ import annotations

from pathlib import Path

import pytest
from dashboard import chain

REPO = Path(__file__).resolve().parents[3]


def test_the_catalogue_has_the_six_slots_and_every_entry_has_a_status():
    out = chain.describe_catalog()
    assert set(out) == {"data", "features", "signal", "portfolio", "risk", "execution"}
    for rows in out.values():
        assert rows and all({"id", "status"} <= set(r) for r in rows)
    assert {r["id"]: r["status"] for r in out["data"]}["gspc"] == "stream"
    assert out["features"][0] == {"id": "none", "status": "none"}


def test_the_probe_names_what_it_finds():
    assert chain._probe("features", None) == {"status": "none"}
    assert chain._probe("data", {"source": "A.csv"}) == {"status": "stream", "source": "A.csv"}
    assert chain._probe("signal", "no.such.module.Thing")["status"] == "import_error"
    assert chain._probe("execution", "dense_armor.core.finance.schedule.twap_schedule") == {"status": "func"}
    assert chain._probe("signal", "dashboard.app.STATIC_DIR") == {"status": "not_a_class"}
    assert chain._probe("signal", "dense_armor.core.online.tree.hoeffding.HoeffdingTreeClassifier") == {"status": "ready"}
    adaptive = chain._probe("signal", "dense_armor.core.online.tree.adaptive.HoeffdingAdaptiveTreeClassifier")
    assert adaptive["status"] == "ready" and "drift_detector" in adaptive["default_init"]
    needs = chain._probe("signal", "dense_armor.core.finance.audit_db.AuditDB")
    assert needs["status"] == "needs_args" and needs["error"]


def test_the_defaults_fill_the_arguments_that_a_class_requires():
    class Needs:
        def __init__(self, drift_detector, classifier, optimizer, k, n_features, mode, n_models, members,
                     factory, base, fields, range_ret, range_vol, f, h, q, r, max_size, n, estimator,
                     metric_model, warning_detector, binary_factory):
            pass

    out = chain._auto_defaults(Needs)
    assert set(out) == {
        "drift_detector", "classifier", "optimizer", "k", "n_features", "mode", "n_models", "members",
        "factory", "base", "fields", "range_ret", "range_vol", "f", "h", "q", "r", "max_size", "n",
        "estimator", "metric_model", "warning_detector", "binary_factory",
    }
    assert out["k"] == 3 and out["n_features"] == 4 and out["mode"] == "classification"
    assert out["n_models"] == 10 and out["max_size"] == 1.0 and out["n"] == 4

    class Unknown:
        def __init__(self, strange_name):
            pass

    assert chain._auto_defaults(Unknown) == {}

    class Optional:
        def __init__(self, a=1):
            pass

    assert chain._auto_defaults(Optional) == {}


def test_instantiate_gives_none_a_function_a_class_and_raises_for_the_rest():
    assert chain._instantiate(None) is None
    assert callable(chain._instantiate("dense_armor.core.finance.schedule.twap_schedule"))
    obj = chain._instantiate("dense_armor.core.online.tree.hoeffding.HoeffdingTreeClassifier")
    assert hasattr(obj, "learn_one")
    with pytest.raises(Exception):
        chain._instantiate("dense_armor.core.finance.audit_db.AuditDB")


def test_maybe_call_drops_the_keywords_that_the_class_does_not_take():
    class Takes:
        def __init__(self, a=0):
            self.a = a

    assert chain._maybe_call(Takes, a=5, b=6).a == 5


def test_the_data_of_a_stream_and_of_a_ticker(monkeypatch):
    monkeypatch.chdir(REPO)
    rows, ys, meta = chain._load_data("driftstream_sudden", 50, 1)
    assert len(rows) == len(ys) == 50 and meta["kind"] == "synthetic"
    assert set(rows[0]) == {"x0", "x1", "x2", "x3"}
    rows, ys, meta = chain._load_data("gspc", 60, 0)
    assert rows and len(rows) == len(ys) and meta["kind"] == "finance"
    with pytest.raises(Exception):
        chain._load_data("nosuchticker", 60, 0)


def test_a_chain_with_a_missing_or_unknown_section_is_refused():
    full = {"data": "driftstream_sudden", "features": "none", "signal": "hoeffding",
            "portfolio": "none", "risk": "none", "execution": "none"}
    missing = dict(full)
    del missing["risk"]
    assert chain.run_chain(missing)["error"] == "missing section risk"
    unknown = dict(full, signal="no_such_piece")
    assert "no_such_piece" in chain.run_chain(unknown)["error"]
    assert "nosuchticker" in chain.run_chain(dict(full, data="nosuchticker"), n=60)["error"]


def test_a_data_source_that_fails_is_reported_as_data(monkeypatch):
    def broken(*a, **k):
        raise FileNotFoundError("no file")

    monkeypatch.setattr(chain, "_load_data", broken)
    full = {"data": "gspc", "features": "none", "signal": "hoeffding",
            "portfolio": "none", "risk": "none", "execution": "none"}
    assert chain.run_chain(full, n=60) == {"ok": False, "error": "data: FileNotFoundError: no file"}


def test_a_piece_that_cannot_be_built_is_reported_as_instantiate(monkeypatch):
    def broken(path):
        raise TypeError("needs args")

    monkeypatch.setattr(chain, "_instantiate", broken)
    full = {"data": "driftstream_sudden", "features": "none", "signal": "hoeffding",
            "portfolio": "none", "risk": "none", "execution": "none"}
    assert chain.run_chain(full, n=60) == {"ok": False, "error": "instantiate: needs args"}


PICKS = [
    {"data": "driftstream_sudden", "features": "standard_scaler", "signal": "hoeffding",
     "portfolio": "none", "risk": "adwin", "execution": "none"},
    {"data": "gspc", "features": "running_moments", "signal": "hoeffding",
     "portfolio": "min_variance", "risk": "lof", "execution": "twap"},
    {"data": "driftstream_gradual", "features": "minmax_scaler", "signal": "adaptive",
     "portfolio": "inverse_volatility", "risk": "hst", "execution": "pipeline"},
    {"data": "aapl", "features": "ew_stats", "signal": "gaussian_nb",
     "portfolio": "pipeline", "risk": "page_hinkley", "execution": "fix_gateway"},
    {"data": "driftstream_incremental", "features": "rolling_median", "signal": "logistic",
     "portfolio": "none", "risk": "hampel_scorer", "execution": "vwap"},
]


@pytest.mark.parametrize("picks", PICKS, ids=lambda p: "-".join(p.values())[:60])
def test_a_chain_runs_and_reports_what_each_step_did(picks, monkeypatch):
    monkeypatch.chdir(REPO)
    out = chain.run_chain(picks, n=80, seed=1)
    assert out["ok"] is True, out
    m = out["metrics"]
    assert m["n"] in (79, 80) and 0.0 <= m["accuracy"] <= 1.0
    assert set(m["errors"]) == {"features", "signal", "risk"}
    assert set(m["role_attributes"]) == {"features", "signal", "risk"}
    assert out["series"]["accuracy"]
    if picks["portfolio"] != "none":
        assert m["portfolio"] is not None
    if picks["execution"] != "none":
        assert m["execution"] is not None


def test_a_step_that_raises_is_counted_and_named(monkeypatch):
    monkeypatch.chdir(REPO)

    class Boom:
        def learn_one(self, *a, **k):
            raise ValueError("boom")

        def predict_one(self, *a, **k):
            raise ValueError("boom")

        def transform_one(self, *a, **k):
            raise ValueError("boom")

        def score_one(self, *a, **k):
            raise ValueError("boom")

    monkeypatch.setattr(chain, "_instantiate", lambda path: Boom() if path else None)
    out = chain.run_chain(
        {"data": "driftstream_sudden", "features": "standard_scaler", "signal": "hoeffding",
         "portfolio": "none", "risk": "lof", "execution": "none"}, n=60, seed=0,
    )
    errors = out["metrics"]["errors"]
    assert out["ok"] is True
    assert errors["features"]["n"] == 60 and errors["features"]["first"] == "ValueError: boom"
    assert errors["signal"]["n"] == 60 and errors["risk"]["n"] == 60
    assert out["metrics"]["accuracy"] == 0.0


def test_the_defaults_survive_a_signature_that_cannot_be_read_and_a_missing_default(monkeypatch):
    import inspect
    import sys

    class Needs:
        def __init__(self, drift_detector):
            pass

    def refuse(obj):
        raise ValueError("none")

    with monkeypatch.context() as m:
        m.setattr(inspect, "signature", refuse)
        assert chain._auto_defaults(Needs) == {}
        assert chain._maybe_call(Needs, drift_detector=1) is Needs
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "dense_armor.core.online.drift.adwin", None)
        assert chain._auto_defaults(Needs) == {}
    with monkeypatch.context() as m:
        class Raising:
            def __init__(self):
                raise RuntimeError("no")

        m.setattr("dense_armor.core.online.tree.hoeffding.HoeffdingTreeClassifier", Raising)

        class NeedsClassifier:
            def __init__(self, classifier):
                pass

        assert chain._auto_defaults(NeedsClassifier) == {}
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "dense_armor.core.online.learn.metric_learning", None)

        class NeedsMetric:
            def __init__(self, metric_learner):
                pass

        assert chain._auto_defaults(NeedsMetric) == {}


def test_the_probe_of_a_class_whose_defaults_do_not_build_it(monkeypatch):
    import sys
    import types

    mod = types.ModuleType("fake_probe_pieces")

    class NeedsK:
        def __init__(self, k):
            raise ValueError("bad k")

    mod.NeedsK = NeedsK
    monkeypatch.setitem(sys.modules, "fake_probe_pieces", mod)
    out = chain._probe("signal", "fake_probe_pieces.NeedsK")
    assert out["status"] == "needs_args" and "bad k" in out["error"]

    class BrokenAdwin:
        def __init__(self):
            raise RuntimeError("adwin")

    monkeypatch.setattr("dense_armor.core.online.drift.adwin.ADWIN", BrokenAdwin)
    adaptive = chain._probe("signal", "dense_armor.core.online.tree.adaptive.HoeffdingAdaptiveTreeClassifier")
    assert adaptive["status"] in ("ready", "needs_args")


def test_a_ticker_file_with_bad_lines_and_too_few_rows(tmp_path, monkeypatch):
    folder = tmp_path / "test" / "fixtures" / "finance"
    folder.mkdir(parents=True)
    lines = ["Date,Close", "short", "2020,notafloat", "2020,10", "2020,11"]
    (folder / "ZZZ.csv").write_text(chr(10).join(lines) + chr(10), encoding="utf-8")
    monkeypatch.setattr(chain, "FIXTURES", folder)
    with pytest.raises(RuntimeError, match="too few rows"):
        chain._load_data("zzz", 50, 0)


def _picks(**over):
    base = {"data": "driftstream_sudden", "features": "none", "signal": "hoeffding",
            "portfolio": "none", "risk": "none", "execution": "none"}
    base.update(over)
    return base


def test_a_risk_piece_that_updates_or_only_learns_is_counted_and_run(monkeypatch):
    class Updating:
        _drift_detected = True

        def update(self, v):
            return True

    class OnlyLearns:
        learned = 0

        def learn_one(self, x):
            OnlyLearns.learned += 1

    real = chain._instantiate
    made = {"cls": Updating}

    def fake(path):
        if path and ("anomaly" in path or "drift" in path):
            return made["cls"]()
        return real(path)

    monkeypatch.setattr(chain, "_instantiate", fake)
    out = chain.run_chain(_picks(risk="adwin"), n=60, seed=0)
    assert out["ok"] is True and out["metrics"]["risk_alarms"] == 60
    made["cls"] = OnlyLearns
    out = chain.run_chain(_picks(risk="lof"), n=60, seed=0)
    assert out["ok"] is True and OnlyLearns.learned == 60


def test_a_portfolio_or_an_execution_that_raises_is_reported_in_the_metrics(monkeypatch):
    monkeypatch.chdir(REPO)

    def boom(*a, **k):
        raise RuntimeError("no panel")

    monkeypatch.setattr("pipelines.finance.portfolio.load_panel", boom)

    class ExecutionPipeline:
        def process_order(self, size):
            raise RuntimeError("no order")

    monkeypatch.setattr("pipelines.finance.execution.ExecutionPipeline", ExecutionPipeline)
    out = chain.run_chain(_picks(portfolio="min_variance", execution="pipeline"), n=60, seed=0)
    assert out["ok"] is True
    assert out["metrics"]["portfolio"] == {"error": "RuntimeError: no panel"}
    assert out["metrics"]["execution"] == {"error": "RuntimeError: no order"}


def test_the_score_gate_waits_learns_the_scale_and_fires_on_an_outlier():
    gate = chain._ScoreGate()
    assert not any(gate.alarm(1.0 + (i % 3) * 0.01) for i in range(60))
    assert gate.alarm(5.0) is True
    big = chain._ScoreGate()
    assert not any(big.alarm(100.0 + (i % 3)) for i in range(60))
    assert big.alarm(500.0) is True
    warm = chain._ScoreGate()
    assert [warm.alarm(float(i)) for i in range(30)] == [False] * 30


def test_the_score_gate_of_an_inverted_score_fires_below_the_mean_and_skips_what_is_not_a_number():
    inverted = chain._ScoreGate(inverted=True)
    for i in range(40):
        inverted.alarm(1.0 + (i % 3) * 0.01)
    assert inverted.alarm(5.0) is False
    assert inverted.alarm(-5.0) is True
    ignore = chain._ScoreGate()
    assert ignore.alarm(float("nan")) is False and ignore.alarm(float("inf")) is False and ignore.n == 0


def test_a_risk_detector_with_a_big_scale_alarms_on_its_outlier_only(monkeypatch):
    class Score100:
        calls = 0
        score_is_inverted = False

        def score_one(self, x):
            Score100.calls += 1
            return 1000.0 if Score100.calls == 51 else 100.0 + (Score100.calls % 3) * 0.1

        def learn_one(self, x):
            pass

    real = chain._instantiate
    monkeypatch.setattr(chain, "_instantiate", lambda path: Score100() if path and "anomaly" in path else real(path))
    out = chain.run_chain(_picks(risk="lof"), n=80, seed=0)
    assert out["ok"] is True
    assert out["metrics"]["risk_alarms"] == 1 and out["alarms"] == [50]
