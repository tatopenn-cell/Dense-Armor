"""Tests for dense_armor.utility.finance.sections."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from dense_armor.utility.finance import sections as sec


class _FakeTemplate:
    def __init__(self):
        self.calls = []

    def __call__(self, template_name, **ctx):
        self.calls.append((template_name, ctx))
        return f"<html>{template_name}</html>"


def _ctx():
    return {"config": {"paths": {"log": Path("/tmp/log.jsonl")}},
            "pyver": "3.13", "machine": "test", "config_path": "/tmp/demo.toml"}


def test_svg_preview_empty_returns_empty_svg():
    svg = sec._svg_preview([])
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert "polyline" not in svg


def test_svg_preview_flat_series():
    svg = sec._svg_preview([1.0, 1.0, 1.0])
    assert "<polyline" in svg


def test_svg_preview_normal_series():
    svg = sec._svg_preview([0.0, 1.0, 2.0, 3.0], color="#ff0000")
    assert "<polyline" in svg
    assert "#ff0000" in svg


def test_svg_preview_drops_nan_and_inf():
    svg = sec._svg_preview([0.0, float("nan"), float("inf"), 1.0])
    assert "<polyline" in svg


def test_svg_bars_empty():
    svg = sec._svg_bars([])
    assert svg.startswith("<svg") and "<rect" not in svg


def test_svg_bars_normal():
    svg = sec._svg_bars([0.1, 0.5, -0.3], color="#00ff00")
    assert svg.count("<rect") == 3
    assert "#00ff00" in svg


def test_svg_bars_flat():
    svg = sec._svg_bars([0.0, 0.0])
    assert svg.count("<rect") == 2


def test_dashboard_cards_shape():
    cards = sec._dashboard_cards()
    assert len(cards) == 4
    for c in cards:
        assert "name" in c and "value" in c and "svg" in c


def test_feature_bar_placeholder():
    svg = sec._feature_bar_placeholder()
    assert svg.startswith("<svg") and "<rect" not in svg


def test_svg_line_placeholder():
    svg = sec._svg_line_placeholder()
    assert svg.startswith("<svg") and "<line" in svg


def test_wilson_band_basic():
    lo, hi = sec._wilson_band(50, 100)
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0
    assert lo < hi


def test_wilson_band_zero():
    lo, hi = sec._wilson_band(0, 0)
    assert lo == 0.0 and hi == 1.0


def test_wilson_band_perfect():
    lo, hi = sec._wilson_band(100, 100)
    assert lo > 0.9 and hi >= 0.99


def test_role_attributes_none():
    assert sec._role_attributes(None) == {}


def test_role_attributes_empty_instance():
    class Bare:
        pass
    assert sec._role_attributes(Bare()) == {}


def test_role_attributes_full():
    from dense_armor.utility.online.anomaly.lof import LocalOutlierFactor
    inst = LocalOutlierFactor(n_neighbors=5, window=20)
    attrs = sec._role_attributes(inst)
    assert attrs["budget_s"] > 0
    assert attrs["memory_class"] in ("O(1)", "O(window)")
    assert "score_is_inverted" in attrs


def test_role_attributes_with_state():
    from dense_armor.utility.online.drift.adwin import ADWIN
    det = ADWIN()
    for v in [0.1, 0.2, 0.3]:
        det.update(v)
    attrs = sec._role_attributes(det)
    assert "drift_detected" in attrs
    assert isinstance(attrs["drift_detected"], bool)


def test_piece_prev_next_middle():
    prev, nxt = sec._piece_prev_next("trees", "anytime")
    assert prev == "hoeffding"
    assert nxt == "adaptive"


def test_piece_prev_next_first():
    prev, nxt = sec._piece_prev_next("trees", "hoeffding")
    assert prev is None
    assert nxt == "anytime"


def test_piece_prev_next_last():
    prev, nxt = sec._piece_prev_next("trees", "subspace_ht")
    assert prev == "oxt"
    assert nxt is None


def test_piece_prev_next_unknown():
    prev, nxt = sec._piece_prev_next("trees", "nonexistent")
    assert prev is None and nxt is None


def test_auto_init_drift_detector():
    from dense_armor.utility.online.tree.adaptive import HoeffdingAdaptiveTreeClassifier
    kwargs = sec._auto_init_kwargs(HoeffdingAdaptiveTreeClassifier)
    assert "drift_detector" in kwargs
    assert kwargs["drift_detector"] is not None


def test_auto_init_optimizer():
    from dense_armor.utility.online.linear.glm import LogisticRegression
    kwargs = sec._auto_init_kwargs(LogisticRegression)
    assert "optimizer" in kwargs


def test_auto_init_nothing_needed():
    from dense_armor.utility.online.tree.hoeffding import HoeffdingTreeClassifier
    kwargs = sec._auto_init_kwargs(HoeffdingTreeClassifier)
    assert kwargs == {}


def test_auto_init_base_kwargs_kept():
    from dense_armor.utility.online.tree.hoeffding import HoeffdingTreeClassifier
    kwargs = sec._auto_init_kwargs(HoeffdingTreeClassifier, {"grace_period": 50})
    assert kwargs["grace_period"] == 50


def test_coerce_int():
    assert sec._coerce("x", "42", "int") == 42
    assert sec._coerce("x", "notint", "int") is None


def test_coerce_float():
    assert sec._coerce("x", "3.14", "float") == 3.14
    assert sec._coerce("x", "bad", "float") is None


def test_coerce_str():
    assert sec._coerce("x", "hello", "str") == "hello"
    assert sec._coerce("x", "'quoted'", "str") == "quoted"
    assert sec._coerce("x", '"quoted"', "str") == "quoted"


def test_coerce_bool():
    assert sec._coerce("x", "true", "bool") is True
    assert sec._coerce("x", "no", "bool") is False


def test_coerce_empty():
    assert sec._coerce("x", "", "int") is None


def test_coerce_fallback():
    assert sec._coerce("x", "42", "") == 42
    assert sec._coerce("x", "3.14", "") == 3.14
    assert sec._coerce("x", "abc", "") == "abc"


def test_returns_preview_real_file():
    rets = sec._returns_preview("test/fixtures/finance/GSPC.csv", n=50)
    assert len(rets) > 0
    assert all(isinstance(r, float) for r in rets)


def test_returns_preview_missing_file():
    assert sec._returns_preview("/nonexistent/file.csv") == []


def test_gspc_returns_200():
    rets = sec._gspc_returns_200()
    assert len(rets) == 200


def test_driftstream_preview():
    vals = sec._driftstream_preview("sudden", n=20)
    assert len(vals) == 20


def test_gspc_classification_stream():
    xs, ys = sec._gspc_classification_stream(n=50)
    assert len(xs) == len(ys)
    assert all("ret" in x for x in xs)
    assert all(y in (0, 1) for y in ys)


def test_feature_series_scaler():
    prev = sec._feature_series(
        "dense_armor.utility.signals.preprocessing.scale",
        "StandardScaler",
        "transform.ret",
    )
    assert prev["n"] > 0
    assert prev["svg"]


def test_sig_preview_hoeffding():
    prev = sec._sig_preview(
        "dense_armor.utility.online.tree.hoeffding",
        "HoeffdingTreeClassifier",
        {},
    )
    assert prev["n"] > 0
    assert prev["svg"]


def test_risk_preview_lof():
    prev = sec._risk_preview("anomaly", "lof")
    assert prev["n"] > 0
    assert prev["svg"]


def test_risk_preview_adwin():
    prev = sec._risk_preview("drift", "adwin")
    assert prev["n"] > 0
    assert prev["svg"]


def test_finance_features():
    X, Y = sec._finance_features(n=200)
    assert len(X) == len(Y) > 0
    assert "ret_1" in X[0]
    assert all(y in (0, 1) for y in Y)


def test_synthetic_features():
    X, Y = sec._synthetic_features(n=100)
    assert len(X) == len(Y) == 100
    assert "a" in X[0] and "b" in X[0]


def test_piece_hoeffding_manual():
    out = sec._piece_hoeffding({"dataset": "manual"})
    assert out["ok"] is True
    assert out["metrics"]["kind"] == "classifier"
    assert "accuracy" in out["metrics"]


def test_piece_hoeffding_finance():
    out = sec._piece_hoeffding({"dataset": "finance"})
    assert out["ok"] is True
    assert len(out["series"]) > 0


def test_piece_meta_existing():
    meta = sec.piece_meta("signal", "trees", "hoeffding")
    assert meta["ok"] is True
    assert meta["name"] == "HoeffdingTreeClassifier"
    assert "params" in meta
    assert "bases" in meta
    assert meta["prev_id"] is None
    assert meta["next_id"] == "anytime"


def test_piece_meta_unknown():
    meta = sec.piece_meta("signal", "trees", "nonexistent")
    assert meta["ok"] is False
    assert "unknown piece" in meta["error"]


def test_piece_detail_hoeffding():
    out = sec.piece_detail("signal", "trees", "hoeffding", {"dataset": "manual"})
    assert out["ok"] is True
    assert out["metrics"]["kind"] == "classifier"


def test_piece_detail_lof():
    out = sec.piece_detail("risk", "anomaly", "lof", {})
    assert out["ok"] is True
    assert out["metrics"]["kind"] == "anomaly"


def test_piece_detail_adwin():
    out = sec.piece_detail("risk", "drift", "adwin", {})
    assert out["ok"] is True
    assert out["metrics"]["kind"] == "drift"


def test_piece_detail_unknown():
    out = sec.piece_detail("signal", "trees", "nonexistent", {})
    assert out["ok"] is False


def test_data_page():
    fake = _FakeTemplate()
    out = sec.data_page(fake, _ctx())
    assert "section.html" in out
    assert fake.calls[0][1]["page"] == "data"
    assert len(fake.calls[0][1]["pieces"]) >= 4


def test_stub_page():
    fake = _FakeTemplate()
    sec.stub_page("More", 0, ["Reserved"], fake, _ctx())
    assert fake.calls[0][1]["total"] == 0


def test_execution_page():
    fake = _FakeTemplate()
    sec.execution_page(fake, _ctx())
    assert fake.calls[0][1]["page"] == "execution"
    assert len(fake.calls[0][1]["pieces"]) == 5


def test_portfolio_page():
    fake = _FakeTemplate()
    sec.portfolio_page(fake, _ctx())
    assert fake.calls[0][1]["page"] == "portfolio"
    assert len(fake.calls[0][1]["pieces"]) == 4


def test_risk_page_anomaly():
    fake = _FakeTemplate()
    sec.risk_page("anomaly", fake, _ctx())
    assert fake.calls[0][1]["active_cat"] == 0


def test_risk_page_drift():
    fake = _FakeTemplate()
    sec.risk_page("drift", fake, _ctx())
    assert fake.calls[0][1]["active_cat"] == 1


def test_features_page_all_categories():
    for cat in ("scaling", "moments", "sketches", "dependence",
                "selection", "distribution", ""):
        fake = _FakeTemplate()
        sec.features_page(cat, fake, _ctx())
        assert fake.calls[0][1]["page"] == "features"


def test_signal_page_all_categories():
    for cat in ("trees", "linear", "ensembles", "facto", "multi", ""):
        fake = _FakeTemplate()
        sec.signal_page(cat, fake, _ctx())
        assert fake.calls[0][1]["page"] == "signal"


def test_ema_in_features():
    # _finance_features carica GSPC (1500 campioni) e tronca a len(arr)-1;
    # 'n' è solo un bound inferiore, non un limite superiore
    X, Y = sec._finance_features(n=10)
    assert len(X) > 0 and len(X) == len(Y)
