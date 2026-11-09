from __future__ import annotations

import logging

import inspect
import importlib
"""Section pages: static skeleton, one per icon of the rail."""


from typing import Any

_log = logging.getLogger(__name__)


def _swallow(exc: BaseException, where: str) -> None:
    """Log a swallowed exception at DEBUG. Visible with --log-level DEBUG."""
    _log.debug("swallowed %s in %s: %s", type(exc).__name__, where, exc)



def _svg_preview(values: list[float], w: int = 220, h: int = 90,
                 color: str = "#4a9eff") -> str:
    """A tiny SVG line of the given values, sized for a card."""
    import math
    values = [float(v) for v in values
              if v is not None and math.isfinite(float(v))]
    if len(values) < 2:
        return f'<svg viewBox="0 0 {w} {h}"></svg>'
    lo, hi = min(values), max(values)
    if hi <= lo:
        # flat series: put it in the middle, and add padding around
        mid = (lo + hi) / 2
        pad_val = max(abs(mid) * 0.5, 1.0)
        lo, hi = mid - pad_val, mid + pad_val
    pad = 4
    pts = []
    n = len(values)
    for i, v in enumerate(values):
        x = pad + (w - 2 * pad) * i / (n - 1)
        y = h - pad - (v - lo) / (hi - lo) * (h - 2 * pad)
        pts.append(f"{x:.1f},{y:.1f}")
    return (f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none">'
            f'<polyline points="{" ".join(pts)}" fill="none" '
            f'stroke="{color}" stroke-width="1.2" '
            f'vector-effect="non-scaling-stroke"/></svg>')




def _svg_bars(values: list[float], w: int = 220, h: int = 90,
               color: str = "#4a9eff") -> str:
    """A tiny SVG bar chart, one bar per value."""
    if not values:
        return f'<svg viewBox="0 0 {w} {h}"></svg>'
    lo, hi = min(values + [0.0]), max(values + [0.0])
    if hi <= lo:
        hi = lo + 1
    n = len(values)
    bar_w = max(2, (w - 4) / max(n * 1.6, 1))
    gap = (w - n * bar_w) / max(n + 1, 1)
    zero_y = h - 4 - (0 - lo) / (hi - lo) * (h - 8)
    parts = []
    for i, v in enumerate(values):
        x = gap + i * (bar_w + gap)
        y0 = h - 4 - (0 - lo) / (hi - lo) * (h - 8)
        y1 = h - 4 - (v - lo) / (hi - lo) * (h - 8)
        top = min(y0, y1)
        bh = abs(y1 - y0)
        parts.append(f'<rect x="{x:.1f}" y="{top:.1f}" width="{bar_w:.1f}" '
                     f'height="{bh:.1f}" fill="{color}"/>')
    return (f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none">'
            + "".join(parts) + "</svg>")



_RISK_CACHE: dict[str, dict] = {}


def _risk_preview(kind: str, piece_id: str) -> dict:
    """Compute (and cache) a 200-sample preview for one risk piece."""
    key = f"{kind}:{piece_id}"
    if key in _RISK_CACHE:
        return _RISK_CACHE[key]
    import numpy as _np
    from pathlib import Path as _P
    out = {"svg": "", "n": 0, "error": None}
    try:
        vals = _returns_preview("test/fixtures/finance/GSPC.csv", n=200)
        if not vals:
            _RISK_CACHE[key] = out
            return out
        rows = [{"ret": float(v)} for v in vals]
        if kind == "anomaly":
            from dense_armor.utility.online.anomaly.lof import LocalOutlierFactor
            from dense_armor.utility.online.anomaly.hst import HalfSpaceTrees
            from dense_armor.utility.online.anomaly.loda import LODA
            from dense_armor.utility.online.anomaly.oiforest import OnlineIsolationForest
            from dense_armor.utility.online.anomaly.mahalanobis import OnlineRobustMahalanobis
            from dense_armor.utility.online.anomaly.ocsvm import OneClassSGD
            from dense_armor.utility.online.anomaly.filters import (
                HampelScorer, ChauvenetScorer, SigmaClipScorer, TukeyScorer,
            )
            classes = {
                "lof": LocalOutlierFactor, "hst": HalfSpaceTrees,
                "loda": LODA, "iforest": OnlineIsolationForest,
                "mahalanobis": OnlineRobustMahalanobis, "ocsvm": OneClassSGD,
                "hampel": HampelScorer, "chauvenet": ChauvenetScorer,
                "sigma_clip": SigmaClipScorer, "tukey": TukeyScorer,
            }
            cls = classes[piece_id]
            det = None
            for init in ({}, {"feature": "ret"}, {"feature_keys": ["ret"]}):
                try:
                    det = cls(**init)
                    break
                except Exception:
                    try:
                        det = cls(**_auto_init_kwargs(cls, init))
                        break
                    except Exception:
                        continue
            if det is None:
                raise RuntimeError(f"cannot instantiate {cls.__name__}")
            scores = []
            first_exc = None
            for r in rows:
                try:
                    s_val = det.score_one(r)
                    if s_val is not None and hasattr(s_val, "__float__"):
                        scores.append(float(s_val))
                except Exception as e:
                    if first_exc is None:
                        first_exc = e
                try:
                    det.learn_one(r)
                except Exception as _e:
                    _swallow(_e, "line 136")
            out = {"svg": _svg_preview(scores, color="#e67e22"),
                   "n": len(scores), "error": None}
        elif kind == "drift":
            from dense_armor.utility.online.drift.adwin import ADWIN
            from dense_armor.utility.online.drift.page_hinkley import PageHinkley
            from dense_armor.utility.online.drift.kswin import KSWIN
            from dense_armor.utility.online.drift.detector import CUSUMDriftDetector
            from dense_armor.utility.online.drift.bocpd import BayesianOnlineChangepoint
            classes = {
                "adwin": ADWIN, "page_hinkley": PageHinkley, "kswin": KSWIN,
                "cusum": CUSUMDriftDetector, "bocpd": BayesianOnlineChangepoint,
            }
            det = classes[piece_id]()
            cum = []
            cnt = 0
            for r in rows:
                try:
                    det.update(r["ret"])
                    if bool(getattr(det, "_drift_detected", False)):
                        cnt += 1
                except Exception as _e:
                    _swallow(_e, "line 158")
                cum.append(cnt)
            out = {"svg": _svg_preview(cum, color="#f85149"),  # type: ignore
                   "n": len(cum), "error": None}
        else:
            out = {"svg": "", "n": 0, "error": None}
    except Exception as e:
        out = {"svg": "", "n": 0, "error": f"{type(e).__name__}: {e}"}
    _RISK_CACHE[key] = out
    return out

def _returns_preview(path: str, n: int = 100) -> list[float]:
    from pathlib import Path as _P
    import numpy as _np
    p = _P(path)
    if not p.exists():
        return []
    rows = p.read_text(encoding="utf-8", errors="replace").splitlines()
    start = next((i + 1 for i, ln in enumerate(rows)
                  if ln.split(",")[0].strip().lower() == "date"), 0)
    closes = []
    for ln in rows[start:]:
        parts = ln.split(",")
        if len(parts) < 2:
            continue
        try:
            closes.append(float(parts[1]))
        except ValueError:
            continue
    arr = _np.asarray(closes, dtype=float)  # type: ignore
    if len(arr) < 3:
        return []
    rets = _np.diff(_np.log(arr))
    rets = rets[-n:]
    return [float(x) for x in rets]


def _driftstream_preview(kind: str, n: int = 100) -> list[float]:
    try:
        from dense_armor.utility.signals.datasets import DriftStream
    except Exception:
        return []
    ds = DriftStream(n_samples=n, n_features=1, n_drifts=1, kind=kind, seed=0)
    X, y = ds.data()
    return [float(v) for v in X[:n, 0]]


def _dashboard_cards() -> list[dict]:
    """The four cards on the right, all 'waiting' for now."""
    return [
        {"name": "accuracy",   "value": "—", "svg": ""},
        {"name": "features",   "value": "—", "svg": ""},
        {"name": "risk",       "value": "—", "svg": ""},
        {"name": "portfolio",  "value": "—", "svg": ""},
    ]




# ---------- features previews ----------

_FEAT_CACHE: dict[str, dict] = {}


def _gspc_returns_200() -> list[float]:
    return _returns_preview("test/fixtures/finance/GSPC.csv", n=200)


def _feature_series(module_path: str, class_name: str,
                    extract: str, init_kwargs: dict | None = None) -> dict:
    """Run a transformer/stat on 200 GSPC returns and extract a scalar series."""
    key = f"{module_path}.{class_name}:{extract}:{init_kwargs}"
    if key in _FEAT_CACHE:
        return _FEAT_CACHE[key]
    out = {"svg": "", "n": 0, "error": None}
    try:
        import importlib as _im
        mod = _im.import_module(module_path)
        cls = getattr(mod, class_name)
        try:
            instance = cls(**(init_kwargs or {}))
        except Exception:
            instance = cls(**_auto_init_kwargs(cls, init_kwargs))
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        _FEAT_CACHE[key] = out
        return out
    series = []
    first_exc = None
    for v in _gspc_returns_200():
        x = {"ret": float(v)}
        try:
            instance.learn_one(x)
        except Exception as _e:
            _swallow(_e, "line 252")
        try:
            if extract == "transform.ret":
                r = instance.transform_one(x)
                if r and "ret" in r:
                    series.append(float(r["ret"]))
                else:
                    series.append(0.0)
            elif extract == "running_mean":
                series.append(float(getattr(instance, "mean_", 0.0)))
            elif extract == "running_median":
                series.append(float(getattr(instance, "value", 0.0)))
            elif extract == "quantile_0.5":
                series.append(float(instance.quantile(0.5)))
            elif extract == "count":
                series.append(float(getattr(instance, "count_", 0.0)))
            elif extract == "value":
                series.append(float(getattr(instance, "value", 0.0)))
            elif extract == "corr":
                series.append(float(getattr(instance, "value", 0.0)))
            else:
                series.append(0.0)
        except Exception as e:
            if first_exc is None:
                first_exc = e
            series.append(0.0)
    out["n"] = len(series)
    if series:
        out["svg"] = _svg_preview(series)
    if first_exc and not any(v != 0 for v in series):
        out["error"] = f"{type(first_exc).__name__}: {first_exc}"
    _FEAT_CACHE[key] = out
    return out


def _feature_bar_placeholder(color: str = "#607d8b") -> str:
    return _svg_bars([0.5] * 3, color=color)



_SIG_CACHE: dict[str, dict] = {}




def _wilson_band(k: int, n: int, z: float = 1.96):
    """Wilson 95% confidence interval for a binomial proportion."""
    import math
    if n == 0:
        return (0.0, 1.0)
    p_hat = k / n
    denom = 1 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    half = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))

def _gspc_classification_stream(n: int = 200):
    """(x, y) pairs from GSPC returns, y = sign of next return."""
    import numpy as _np
    from pathlib import Path as _P
    p = _P("test/fixtures/finance/GSPC.csv")
    if not p.exists():
        return [], []
    rows = p.read_text(encoding="utf-8", errors="replace").splitlines()
    start = next((i + 1 for i, ln in enumerate(rows)
                  if ln.split(",")[0].strip().lower() == "date"), 0)
    closes = []
    for ln in rows[start:]:
        parts = ln.split(",")
        if len(parts) < 2:
            continue
        try:
            closes.append(float(parts[1]))
        except ValueError:
            continue
    arr = _np.asarray(closes, dtype=float)  # type: ignore
    if len(arr) < 3:
        return [], []
    rets = _np.diff(_np.log(arr))[-n - 1:]
    xs = [{"ret": float(rets[i])} for i in range(len(rets) - 1)]
    ys = [int(rets[i + 1] > 0) for i in range(len(rets) - 1)]
    return xs, ys




def _auto_init_kwargs(cls, base_kwargs: dict | None = None) -> dict:
    """Aggiunge default per kwargs obbligatori noti (drift_detector, model, ecc.).
    Usato da _sig_preview, _feature_series, _risk_preview e piece_detail.
    Non tocca la primitiva: solo l'istanziazione.
    """
    import inspect as _insp
    kwargs = dict(base_kwargs or {})
    try:
        sig = _insp.signature(cls.__init__)
    except (ValueError, TypeError):
        return kwargs
    for name, prm in list(sig.parameters.items())[1:]:
        if name == "self" or name in kwargs:
            continue
        if prm.default is not _insp.Parameter.empty and prm.default is not None:
            continue
        low = name.lower()
        try:
            if "drift" in low and "detector" in low:
                from dense_armor.utility.online.drift.adwin import ADWIN
                kwargs[name] = ADWIN()
            elif "warning" in low and "detector" in low:
                from dense_armor.utility.online.drift.adwin import ADWIN
                kwargs[name] = ADWIN()
            elif low in ("model", "estimator", "base_model", "base_estimator"):
                from dense_armor.utility.online.tree.hoeffding import HoeffdingTreeClassifier
                kwargs[name] = HoeffdingTreeClassifier()
            elif low == "classifier":
                from dense_armor.utility.online.tree.hoeffding import HoeffdingTreeClassifier
                kwargs[name] = HoeffdingTreeClassifier()
            elif low == "optimizer":
                try:
                    from dense_armor.utility.online.optim.sgd import SGD
                    kwargs[name] = SGD()
                except Exception as _e:
                    _swallow(_e, "line 373")
            elif low in ("loss", "loss_fn", "loss_function"):
                kwargs[name] = "log_loss"
            elif low in ("link", "inverse_link", "link_fn"):
                kwargs[name] = "logistic"
            elif "metric" in low and "learner" in low:
                from dense_armor.utility.online.learn.metric_learning import LEGO
                kwargs[name] = LEGO()
        except Exception as _e:
            _swallow(_e, "line 382")
    return kwargs

def _sig_preview(module_path: str, class_name: str,
                 init_kwargs: dict | None = None) -> dict:
    """Train a classifier on 200 GSPC samples, return accuracy curve."""
    key = f"{module_path}.{class_name}:{init_kwargs}"
    if key in _SIG_CACHE:
        return _SIG_CACHE[key]
    out = {"svg": "", "n": 0, "error": None, "final_acc": None}
    try:
        import importlib as _im
        mod = _im.import_module(module_path)
        cls = getattr(mod, class_name)
        try:
            instance = cls(**(init_kwargs or {}))
        except Exception:
            instance = cls(**_auto_init_kwargs(cls, init_kwargs))
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:80]}"
        _SIG_CACHE[key] = out
        return out
    xs, ys = _gspc_classification_stream(200)
    if not xs:
        _SIG_CACHE[key] = out
        return out
    correct = 0
    acc = []
    first_exc = None
    for i, (x, y) in enumerate(zip(xs, ys)):
        try:
            pred = instance.predict_one(x)
            if pred is not None and int(pred) == int(y):
                correct += 1
        except Exception as e:
            if first_exc is None:
                first_exc = e
        try:
            instance.learn_one(x, y)
        except Exception as _e:
            _swallow(_e, "line 422")
        acc.append(correct / (i + 1))
    out["n"] = len(acc)
    if acc:
        out["svg"] = _svg_preview(acc, color="#4a9eff")
        out["final_acc"] = round(acc[-1], 3)
    _SIG_CACHE[key] = out
    return out

def data_page(render_template, ctx: dict) -> Any:
    pieces = []
    for ticker in ("GSPC", "AAPL", "MSFT", "TSLA"):
        vals = _returns_preview(f"test/fixtures/finance/{ticker}.csv")
        pieces.append({
            "id": ticker.lower(),
            "name": ticker,
            "kind": "finance stream",
            "svg": _svg_preview(vals, color="#4a9eff"),
            "n": len(vals),
            "size": "M",
        })
    for kind in ("sudden", "gradual", "incremental"):
        vals = _driftstream_preview(kind)
        pieces.append({
            "id": f"driftstream_{kind}",
            "name": f"DriftStream {kind}",
            "kind": "synthetic stream",
            "svg": _svg_preview(vals, color="#e67e22"),
            "n": len(vals),
            "size": "M",
        })
    return render_template(
        "section.html",
        page="data",
        title="Data",
        section="Data",
        total=len(pieces),
        categories=["Streams"],
        active_cat=0,
        pieces=pieces,
        dashboard=_dashboard_cards(),
        **ctx,
    )


def stub_page(title: str, total: int, categories: list[str],
              render_template, ctx: dict) -> Any:
    return render_template(
        "section.html",
        page=title.lower(),
        title=title,
        section=title,
        total=total,
        categories=categories,
        active_cat=0,
        pieces=[],
        dashboard=_dashboard_cards(),
        **ctx,
    )


def execution_page(render_template, ctx: dict) -> Any:
    import numpy as _np
    pieces = [
        {"id": "pipeline", "name": "ExecutionPipeline",
         "kind": "order damping + pre-trade",
         "svg": _svg_line_placeholder("#4a9eff"),
         "n": 0, "size": "M"},
        {"id": "fix_gateway", "name": "FIX gateway",
         "kind": "FIX 4.4 send_orders",
         "svg": _svg_line_placeholder("#2c6fa6"),
         "n": 0, "size": "M"},
        {"id": "twap", "name": "TWAP",
         "kind": "equal slices",
         "svg": _svg_bars([1.0] * 10, color="#4a9eff"),
         "n": 10, "size": "M"},
        {"id": "vwap", "name": "VWAP",
         "kind": "volume weighted",
         "svg": _svg_bars([0.07, 0.31, 0.03, 0.06, 0.07, 0.02, 0.13, 0.05, 0.12, 0.14],
                          color="#3fb950"),
         "n": 10, "size": "M"},
        {"id": "none", "name": "none",
         "kind": "no execution",
         "svg": _svg_bars([], color="#607d8b"), "n": 0, "size": "M"},
    ]
    return render_template(
        "section.html", page="execution", title="Execution",
        section="Execution", total=len(pieces),
        categories=["Orders"], active_cat=0,
        pieces=pieces, dashboard=_dashboard_cards(), **ctx,
    )


def _svg_line_placeholder(color: str = "#4a9eff") -> str:
    return (f'<svg viewBox="0 0 220 90"><line x1="10" y1="45" x2="210" y2="45" '
            f'stroke="{color}" stroke-width="1" stroke-dasharray="2 2" '
            f'vector-effect="non-scaling-stroke"/></svg>')


def portfolio_page(render_template, ctx: dict) -> Any:
    import numpy as _np
    weights_mv = [-0.0865, 1.287, -0.0896, -0.1109]
    weights_iv = [0.2432, 0.4009, 0.2423, 0.1136]
    pieces = [
        {"id": "min_variance", "name": "min_variance",
         "kind": "min-variance weights",
         "svg": _svg_bars(weights_mv, color="#4a9eff"), "n": 4, "size": "M"},
        {"id": "inverse_volatility", "name": "inverse_volatility",
         "kind": "1 / sqrt(S_ii)",
         "svg": _svg_bars(weights_iv, color="#3fb950"), "n": 4, "size": "M"},
        {"id": "pipeline", "name": "PortfolioPipeline",
         "kind": "rolling refit + turnover",
         "svg": _svg_bars(weights_mv, color="#a371f7"), "n": 1253, "size": "M"},
        {"id": "none", "name": "none",
         "kind": "no portfolio",
         "svg": _svg_bars([], color="#607d8b"), "n": 0, "size": "M"},
    ]
    return render_template(
        "section.html", page="portfolio", title="Portfolio",
        section="Portfolio", total=len(pieces),
        categories=["Weights"], active_cat=0,
        pieces=pieces, dashboard=_dashboard_cards(), **ctx,
    )


def risk_page(category: str, render_template, ctx: dict) -> Any:
    if category == "anomaly":
        ids = [
            ("lof", "LocalOutlierFactor", "LOF"),
            ("hst", "HalfSpaceTrees", "HST"),
            ("loda", "LODA", "LODA"),
            ("iforest", "OnlineIsolationForest", "IForest"),
            ("mahalanobis", "OnlineRobustMahalanobis", "Mahalanobis"),
            ("ocsvm", "OneClassSGD", "OCSVM"),
            ("hampel", "HampelScorer", "Hampel"),
            ("chauvenet", "ChauvenetScorer", "Chauvenet"),
            ("sigma_clip", "SigmaClipScorer", "SigmaClip"),
            ("tukey", "TukeyScorer", "Tukey"),
        ]
        pieces = []
        for pid, name, label in ids:
            prev = _risk_preview("anomaly", pid)
            pieces.append({"id": pid, "name": name, "kind": label,
                           "svg": prev["svg"], "n": prev["n"], "size": "S"})
        return render_template(
            "section.html", page="risk", title="Risk",
            section="Risk", total=21,
            categories=["Anomaly", "Drift & Pipelines"], active_cat=0,
            pieces=pieces, dashboard=_dashboard_cards(), **ctx,
        )
    # drift
    ids = [
        ("adwin", "ADWIN", "ADWIN"),
        ("page_hinkley", "PageHinkley", "PH"),
        ("kswin", "KSWIN", "KSWIN"),
        ("cusum", "CUSUMDriftDetector", "CUSUM"),
        ("bocpd", "BayesianOnlineChangepoint", "BOCPD"),
    ]
    pieces = []
    for pid, name, label in ids:
        prev = _risk_preview("drift", pid)
        pieces.append({"id": pid, "name": name, "kind": label,
                       "svg": prev["svg"], "n": prev["n"], "size": "S"})
    # pipelines e pretrade come barre placeholder
    extra = [
        ("finance_pipeline", "FinancePipeline", "surveillance", "#a371f7"),
        ("risk_pipeline", "RiskPipeline", "VaR + CUSUM", "#a371f7"),
        ("regime_pipeline", "RegimePipeline", "BOCPD on returns", "#a371f7"),
        ("drift_vs_noise", "DriftVsNoisePipeline", "raw vs robust", "#a371f7"),
        ("pretrade", "PreTradeChecker", "five limits", "#e67e22"),
    ]
    for pid, name, label, color in extra:
        pieces.append({"id": pid, "name": name, "kind": label,
                       "svg": _svg_bars([0.5] * 5, color=color), "n": 5, "size": "S"})
    pieces.append({"id": "none", "name": "none", "kind": "no risk",
                   "svg": _svg_bars([], color="#607d8b"), "n": 0, "size": "S"})
    return render_template(
        "section.html", page="risk", title="Risk",
        section="Risk", total=21,
        categories=["Anomaly", "Drift & Pipelines"], active_cat=1,
        pieces=pieces, dashboard=_dashboard_cards(), **ctx,
    )


def features_page(category: str, render_template, ctx: dict) -> Any:
    categories = ["Scaling", "Moments", "Sketches", "Dependence",
                  "Selection", "Distribution & Time"]
    cat_map = {"scaling": 0, "moments": 1, "sketches": 2, "dependence": 3,
               "selection": 4, "distribution": 5}
    active = cat_map.get(category, 0)
    pieces: list[dict] = []

    if category == "scaling" or not category:
        scaling = [
            ("standard_scaler", "StandardScaler", "dense_armor.utility.signals.preprocessing.scale", "StandardScaler"),
            ("minmax_scaler", "MinMaxScaler", "dense_armor.utility.signals.preprocessing.scale", "MinMaxScaler"),
            ("robust_scaler", "RobustScaler", "dense_armor.utility.signals.preprocessing.scale", "RobustScaler"),
            ("ew_scaler", "EWScaler", "dense_armor.utility.signals.preprocessing.scale", "EWScaler"),
            ("metro", "Metro", "dense_armor.utility.signals.preprocessing.metro", "Metro"),
        ]
        for pid, name, mod, cls in scaling:
            prev = _feature_series(mod, cls, "transform.ret")
            pieces.append({"id": pid, "name": name, "kind": "scaler",
                           "svg": prev["svg"], "n": prev["n"], "size": "M"})
    elif category == "moments":
        moments = [
            ("running_moments", "RunningMoments", "dense_armor.utility.signals.stats.moments", "RunningMoments", "running_mean"),
            ("ew_stats", "EWStats", "dense_armor.utility.signals.stats.moments", "EWStats", "running_mean"),
            ("rolling_median", "RollingMedian", "dense_armor.utility.signals.stats.robust", "RollingMedian", "running_median"),
            ("rolling_mad", "RollingMAD", "dense_armor.utility.signals.stats.robust", "RollingMAD", "running_median"),
            ("rolling_iqr", "RollingIQR", "dense_armor.utility.signals.stats.robust", "RollingIQR", "running_median"),
            ("rolling_quantile", "RollingQuantile", "dense_armor.utility.signals.stats.robust", "RollingQuantile", "running_median"),
            ("running_moments_vec", "RunningMomentsVector", "dense_armor.utility.signals.stats.moments", "RunningMomentsVector", None),
            ("rolling_median_vec", "RollingMedianVector", "dense_armor.utility.signals.stats.robust", "RollingMedianVector", None),
        ]
        for pid, name, mod, cls, extract in moments:
            if extract is None:
                svg = _feature_bar_placeholder("#a371f7"); n = 0
            else:
                prev = _feature_series(mod, cls, extract)
                svg = prev["svg"]; n = prev["n"]
            pieces.append({"id": pid, "name": name, "kind": "moments",
                           "svg": svg, "n": n, "size": "M"})
    elif category == "sketches":
        sketches = [
            ("ddsketch", "DDSketch", "dense_armor.utility.signals.stats.quantiles", "DDSketch"),
            ("tdigest", "TDigest", "dense_armor.utility.signals.stats.quantiles", "TDigest"),
            ("bloom", "BloomFilter", "dense_armor.utility.signals.stats.sketches", "BloomFilter"),
            ("cms", "CountMinSketch", "dense_armor.utility.signals.stats.sketches", "CountMinSketch"),
            ("hll", "HyperLogLog", "dense_armor.utility.signals.stats.sketches", "HyperLogLog"),
            ("space_saving", "SpaceSaving", "dense_armor.utility.signals.stats.sketches", "SpaceSaving"),
        ]
        for pid, name, mod, cls in sketches:
            prev = _feature_series(mod, cls, "quantile_0.5")
            if not prev["svg"]:
                svg = _feature_bar_placeholder("#3fb950"); n = 0
            else:
                svg = prev["svg"]; n = prev["n"]
            pieces.append({"id": pid, "name": name, "kind": "sketch",
                           "svg": svg, "n": n, "size": "S"})
    elif category == "dependence":
        dep = [
            ("autocorr", "Autocorrelation", "dense_armor.utility.signals.stats.dependence", "Autocorrelation"),
            ("running_corr", "RunningCorrelation", "dense_armor.utility.signals.stats.dependence", "RunningCorrelation"),
            ("running_cov", "RunningCovariance", "dense_armor.utility.signals.stats.dependence", "RunningCovariance"),
            ("rolling_corr", "RollingCorrelation", "dense_armor.utility.signals.stats.dependence", "RollingCorrelation"),
            ("rolling_cov", "RollingCovariance", "dense_armor.utility.signals.stats.dependence", "RollingCovariance"),
        ]
        for pid, name, mod, cls in dep:
            pieces.append({"id": pid, "name": name, "kind": "dependence",
                           "svg": _feature_bar_placeholder("#e67e22"), "n": 0, "size": "S"})
    elif category == "selection":
        sel = [
            ("select_kbest", "SelectKBest", "dense_armor.utility.signals.preprocessing.select"),
            ("var_threshold", "VarianceThreshold", "dense_armor.utility.signals.preprocessing.select"),
            ("select", "Select", "dense_armor.utility.infrastructure.compose.select"),
            ("discard", "Discard", "dense_armor.utility.infrastructure.compose.select"),
            ("func_transformer", "FuncTransformer", "dense_armor.utility.infrastructure.compose.func"),
        ]
        for pid, name, mod in sel:
            pieces.append({"id": pid, "name": name, "kind": "compose",
                           "svg": _feature_bar_placeholder("#607d8b"), "n": 0, "size": "S"})
    else:  # distribution
        dist = [
            ("shrinkage_cov", "OnlineShrinkageCovariance", "dense_armor.utility.signals.stats.shrinkage"),
            ("gmm", "OnlineGaussianMixture", "dense_armor.utility.signals.stats.mixture"),
            ("student_t", "StudentT", "dense_armor.utility.signals.stats.student_t"),
            ("sdft", "SDFT", "dense_armor.utility.signals.stats.sdft"),
            ("skyline", "Skyline", "dense_armor.utility.signals.stats.skyline"),
            ("queue_resampler", "QueueResampler", "dense_armor.utility.signals.preprocessing.imbalance"),
            ("locate", "Locate", "dense_armor.utility.signals.stream.iodat"),
        ]
        for pid, name, mod in dist:
            pieces.append({"id": pid, "name": name, "kind": "distribution",
                           "svg": _feature_bar_placeholder("#4a9eff"), "n": 0, "size": "S"})

    return render_template(
        "section.html", page="features", title="Features",
        section="Features", total=36,
        categories=categories, active_cat=active,
        pieces=pieces, dashboard=_dashboard_cards(), **ctx,
        cat_slugs=["scaling", "moments", "sketches", "dependence",
                   "selection", "distribution"],
    )


def signal_page(category: str, render_template, ctx: dict) -> Any:
    categories = ["Trees & Forests", "Linear & Bayes", "Ensembles & Rules",
                  "Factorization & Recommenders", "Multi / KNN / Cluster"]
    slugs = ["trees", "linear", "ensembles", "facto", "multi"]
    try:
        active = slugs.index(category) if category else 0
    except ValueError:
        active = 0

    if category == "trees" or not category:
        pieces = [
            ("hoeffding", "HoeffdingTreeClassifier", "dense_armor.utility.online.tree.hoeffding", "HoeffdingTreeClassifier", None),
            ("anytime", "HoeffdingAnytimeTreeClassifier", "dense_armor.utility.online.tree.efdt", "HoeffdingAnytimeTreeClassifier", None),
            ("adaptive", "HoeffdingAdaptiveTreeClassifier", "dense_armor.utility.online.tree.adaptive", "HoeffdingAdaptiveTreeClassifier", {"drift_detector": "ADWIN"}),
            ("mondrian", "MondrianForestClassifier", "dense_armor.utility.online.tree.mondrian", "MondrianForestClassifier", None),
            ("mondrian_reg", "MondrianForestRegressor", "dense_armor.utility.online.tree.mondrian", "MondrianForestRegressor", None),
            ("sgt_clf", "SGTClassifier", "dense_armor.utility.online.tree.sgt", "SGTClassifier", None),
            ("sgt_reg", "SGTRegressor", "dense_armor.utility.online.tree.sgt", "SGTRegressor", None),
            ("batch_tree", "BatchDecisionTreeClassifier", "dense_armor.utility.online.tree.batch", "BatchDecisionTreeClassifier", None),
            ("fimtdd", "FIMTDDRegressor", "dense_armor.utility.online.tree.fimt", "FIMTDDRegressor", None),
            ("amf", "AggregatedMondrianForestClassifier", "dense_armor.utility.online.forest.amf", "AggregatedMondrianForestClassifier", None),
            ("amf_reg", "AggregatedMondrianForestRegressor", "dense_armor.utility.online.forest.amf", "AggregatedMondrianForestRegressor", None),
            ("arf", "AdaptiveRandomForestClassifier", "dense_armor.utility.online.forest.arf", "AdaptiveRandomForestClassifier", None),
            ("oxt", "OnlineExtraTreesRegressor", "dense_armor.utility.online.forest.oxt", "OnlineExtraTreesRegressor", None),
            ("subspace_ht", "SubspaceHoeffdingTree", "dense_armor.utility.online.forest.arf", "SubspaceHoeffdingTree", None),
        ]
    elif category == "linear":
        pieces = [
            ("logistic", "LogisticRegression", "dense_armor.utility.online.linear.glm", "LogisticRegression", "optimizer"),  # type: ignore
            ("bayes_gauss", "BayesianLinearRegression", "dense_armor.utility.online.linear.bayes", "BayesianLinearRegression", None),
            ("multinomial_nb", "MultinomialNB", "dense_armor.utility.online.bayes.multinomial", "MultinomialNB", None),
            ("bernoulli_nb", "BernoulliNB", "dense_armor.utility.online.bayes.bernoulli", "BernoulliNB", None),
            ("complement_nb", "ComplementNB", "dense_armor.utility.online.bayes.complement", "ComplementNB", None),
            ("categorical_nb", "CategoricalNB", "dense_armor.utility.online.bayes.categorical", "CategoricalNB", None),
            ("gaussian_nb", "OnlineGaussianNB", "dense_armor.utility.online.learn.online_classifiers", "OnlineGaussianNB", None),
            ("softmax", "OnlineSoftmaxRegression", "dense_armor.utility.online.learn.online_classifiers", "OnlineSoftmaxRegression", None),
            ("platt", "OnlinePlattScaling", "dense_armor.utility.online.learn.calibration", "OnlinePlattScaling", "classifier"),  # type: ignore
            ("rls", "RecursiveLeastSquares", "dense_armor.utility.online.learn.online_dynamics", "RecursiveLeastSquares", None),
            ("metric_knn", "MetricKNNClassifier", "dense_armor.utility.online.learn.metric_learning", "MetricKNNClassifier", "metric_learner"),  # type: ignore
        ]
    elif category == "ensembles":
        pieces = [
            ("bagging", "OnlineBaggingClassifier", "dense_armor.utility.online.ensemble.bagging", "OnlineBaggingClassifier", "mode"),  # type: ignore
            ("boosting", "OnlineBoostingClassifier", "dense_armor.utility.online.ensemble.boosting", "OnlineBoostingClassifier", "mode"),  # type: ignore
            ("stacking", "StackingClassifier", "dense_armor.utility.online.ensemble.stacking", "StackingClassifier", "factory"),  # type: ignore
            ("voting", "VotingClassifier", "dense_armor.utility.online.ensemble.voting", "VotingClassifier", "factory"),  # type: ignore
            ("ewa_reg", "ExponentiallyWeightedAverageRegressor", "dense_armor.utility.online.ensemble.ewa", "ExponentiallyWeightedAverageRegressor", "members"),  # type: ignore
            ("amrules", "AMRulesRegressor", "dense_armor.utility.online.rules.amrules", "AMRulesRegressor", None),
        ]
    elif category == "facto":
        pieces = [
            ("fm", "FMClassifier", "dense_armor.utility.online.facto.fm", "FMClassifier", "optimizer"),  # type: ignore
            ("ffm", "FFMClassifier", "dense_armor.utility.online.facto.ffm", "FFMClassifier", "fields"),  # type: ignore
            ("fwfm", "FwFMClassifier", "dense_armor.utility.online.facto.fwfm", "FwFMClassifier", "optimizer"),  # type: ignore
            ("hofm", "HOFMClassifier", "dense_armor.utility.online.facto.hofm", "HOFMClassifier", "optimizer"),  # type: ignore
            ("baseline", "Baseline", "dense_armor.utility.online.reco.reco", "Baseline", None),
            ("biased_mf", "BiasedMF", "dense_armor.utility.online.reco.reco", "BiasedMF", None),
            ("funk_mf", "FunkMF", "dense_armor.utility.online.reco.reco", "FunkMF", None),
        ]
    else:  # multi
        pieces = [
            ("ovr", "OneVsRestClassifier", "dense_armor.utility.online.multiclass.multiclass", "OneVsRestClassifier", "factory"),  # type: ignore
            ("ovo", "OneVsOneClassifier", "dense_armor.utility.online.multiclass.multiclass", "OneVsOneClassifier", "factory"),  # type: ignore
            ("output_code", "OutputCodeClassifier", "dense_armor.utility.online.multiclass.multiclass", "OutputCodeClassifier", "factory"),  # type: ignore
            ("chain_clf", "ClassifierChain", "dense_armor.utility.online.multioutput.multioutput", "ClassifierChain", "factory"),  # type: ignore
            ("chain_reg", "RegressorChain", "dense_armor.utility.online.multioutput.multioutput", "RegressorChain", "factory"),  # type: ignore
            ("knn", "KNNClassifier", "dense_armor.utility.online.neighbors.knn", "KNNClassifier", None),
            ("knn_reg", "KNNRegressor", "dense_armor.utility.online.neighbors.knn", "KNNRegressor", None),
            ("denstream", "DenStream", "dense_armor.utility.online.cluster.denstream", "DenStream", None),
            ("online_kmeans", "OnlineKMeans", "dense_armor.utility.online.cluster.kmeans", "OnlineKMeans", "k"),  # type: ignore
        ]

    # preview: has_real_preview se l'init è None o dict semplice, placeholder altrimenti
    pieces_out = []
    for entry in pieces:
        pid, name, mod, cls, special = entry
        init_kwargs = None
        has_real_preview = False
        if special is None:
            init_kwargs = {}
            has_real_preview = True
        elif isinstance(special, dict):
            if special.get("drift_detector") == "ADWIN":
                try:
                    from dense_armor.utility.online.drift.adwin import ADWIN
                    init_kwargs = {"drift_detector": ADWIN()}
                    has_real_preview = True
                except Exception as _e:
                    _swallow(_e, "line 796")
        if has_real_preview and init_kwargs is not None:
            try:
                prev = _sig_preview(mod, cls, init_kwargs)
                if prev["svg"]:
                    pieces_out.append({
                        "id": pid, "name": name, "kind": "classifier",
                        "svg": prev["svg"], "n": prev["n"],
                        "final_acc": prev["final_acc"],
                        "size": "M",
                    })
                    continue
            except Exception as _e:
                _swallow(_e, "line 809")
        pieces_out.append({
            "id": pid, "name": name, "kind": "classifier",
            "svg": _feature_bar_placeholder("#607d8b"), "n": 0,
            "size": "S",
        })

    return render_template(
        "section.html", page="signal", title="Signal",
        section="Signal", total=47,
        categories=categories, active_cat=active,
        pieces=pieces_out, dashboard=_dashboard_cards(), **ctx,
        cat_slugs=slugs,
    )


_SIGNAL_CLASSES_EXT = {
    # trees
    ("trees", "hoeffding"): ("dense_armor.utility.online.tree.hoeffding", "HoeffdingTreeClassifier"),
    ("trees", "anytime"): ("dense_armor.utility.online.tree.efdt", "HoeffdingAnytimeTreeClassifier"),
    ("trees", "adaptive"): ("dense_armor.utility.online.tree.adaptive", "HoeffdingAdaptiveTreeClassifier"),
    ("trees", "mondrian"): ("dense_armor.utility.online.tree.mondrian", "MondrianForestClassifier"),
    ("trees", "mondrian_reg"): ("dense_armor.utility.online.tree.mondrian", "MondrianForestRegressor"),
    ("trees", "sgt_clf"): ("dense_armor.utility.online.tree.sgt", "SGTClassifier"),
    ("trees", "sgt_reg"): ("dense_armor.utility.online.tree.sgt", "SGTRegressor"),
    ("trees", "batch_tree"): ("dense_armor.utility.online.tree.batch", "BatchDecisionTreeClassifier"),
    ("trees", "fimtdd"): ("dense_armor.utility.online.tree.fimt", "FIMTDDRegressor"),
    ("trees", "amf"): ("dense_armor.utility.online.forest.amf", "AggregatedMondrianForestClassifier"),
    ("trees", "amf_reg"): ("dense_armor.utility.online.forest.amf", "AggregatedMondrianForestRegressor"),
    ("trees", "arf"): ("dense_armor.utility.online.forest.arf", "AdaptiveRandomForestClassifier"),
    ("trees", "oxt"): ("dense_armor.utility.online.forest.oxt", "OnlineExtraTreesRegressor"),
    ("trees", "subspace_ht"): ("dense_armor.utility.online.forest.arf", "SubspaceHoeffdingTree"),
    # risk anomaly
    ("anomaly", "lof"): ("dense_armor.utility.online.anomaly.lof", "LocalOutlierFactor"),
    ("anomaly", "hst"): ("dense_armor.utility.online.anomaly.hst", "HalfSpaceTrees"),
    ("anomaly", "loda"): ("dense_armor.utility.online.anomaly.loda", "LODA"),
    ("anomaly", "iforest"): ("dense_armor.utility.online.anomaly.oiforest", "OnlineIsolationForest"),
    ("anomaly", "mahalanobis"): ("dense_armor.utility.online.anomaly.mahalanobis", "OnlineRobustMahalanobis"),
    ("anomaly", "ocsvm"): ("dense_armor.utility.online.anomaly.ocsvm", "OneClassSGD"),
    ("anomaly", "hampel"): ("dense_armor.utility.online.anomaly.filters", "HampelScorer"),
    ("anomaly", "chauvenet"): ("dense_armor.utility.online.anomaly.filters", "ChauvenetScorer"),
    ("anomaly", "sigma_clip"): ("dense_armor.utility.online.anomaly.filters", "SigmaClipScorer"),
    ("anomaly", "tukey"): ("dense_armor.utility.online.anomaly.filters", "TukeyScorer"),
    # risk drift
    ("drift", "adwin"): ("dense_armor.utility.online.drift.adwin", "ADWIN"),
    ("drift", "page_hinkley"): ("dense_armor.utility.online.drift.page_hinkley", "PageHinkley"),
    ("drift", "kswin"): ("dense_armor.utility.online.drift.kswin", "KSWIN"),
    ("drift", "cusum"): ("dense_armor.utility.online.drift.detector", "CUSUMDriftDetector"),
    ("drift", "bocpd"): ("dense_armor.utility.online.drift.bocpd", "BayesianOnlineChangepoint"),
    # features scaling
    ("scaling", "standard_scaler"): ("dense_armor.utility.signals.preprocessing.scale", "StandardScaler"),
    ("scaling", "minmax_scaler"): ("dense_armor.utility.signals.preprocessing.scale", "MinMaxScaler"),
    ("scaling", "robust_scaler"): ("dense_armor.utility.signals.preprocessing.scale", "RobustScaler"),
    ("scaling", "ew_scaler"): ("dense_armor.utility.signals.preprocessing.scale", "EWScaler"),
    # features moments
    ("moments", "running_moments"): ("dense_armor.utility.signals.stats.moments", "RunningMoments"),
    ("moments", "ew_stats"): ("dense_armor.utility.signals.stats.moments", "EWStats"),
    ("moments", "rolling_median"): ("dense_armor.utility.signals.stats.robust", "RollingMedian"),

    # signal / linear
    ("linear", "logistic"): ("dense_armor.utility.online.linear.glm", "LogisticRegression"),
    ("linear", "bayes_gauss"): ("dense_armor.utility.online.linear.bayes", "BayesianLinearRegression"),
    ("linear", "multinomial_nb"): ("dense_armor.utility.online.bayes.multinomial", "MultinomialNB"),
    ("linear", "bernoulli_nb"): ("dense_armor.utility.online.bayes.bernoulli", "BernoulliNB"),
    ("linear", "complement_nb"): ("dense_armor.utility.online.bayes.complement", "ComplementNB"),
    ("linear", "categorical_nb"): ("dense_armor.utility.online.bayes.categorical", "CategoricalNB"),
    ("linear", "gaussian_nb"): ("dense_armor.utility.online.learn.online_classifiers", "OnlineGaussianNB"),
    ("linear", "softmax"): ("dense_armor.utility.online.learn.online_classifiers", "OnlineSoftmaxRegression"),
    ("linear", "rls"): ("dense_armor.utility.online.learn.online_dynamics", "RecursiveLeastSquares"),
    # signal / ensembles
    ("ensembles", "bagging"): ("dense_armor.utility.online.ensemble.bagging", "OnlineBaggingClassifier"),
    ("ensembles", "boosting"): ("dense_armor.utility.online.ensemble.boosting", "OnlineBoostingClassifier"),
    ("ensembles", "amrules"): ("dense_armor.utility.online.rules.amrules", "AMRulesRegressor"),
    # signal / facto
    ("facto", "baseline"): ("dense_armor.utility.online.reco.reco", "Baseline"),
    ("facto", "biased_mf"): ("dense_armor.utility.online.reco.reco", "BiasedMF"),
    ("facto", "funk_mf"): ("dense_armor.utility.online.reco.reco", "FunkMF"),
    # signal / multi
    ("multi", "knn"): ("dense_armor.utility.online.neighbors.knn", "KNNClassifier"),
    ("multi", "knn_reg"): ("dense_armor.utility.online.neighbors.knn", "KNNRegressor"),
    ("multi", "denstream"): ("dense_armor.utility.online.cluster.denstream", "DenStream"),
    # features / sketches
    ("sketches", "ddsketch"): ("dense_armor.utility.signals.stats.quantiles", "DDSketch"),
    ("sketches", "tdigest"): ("dense_armor.utility.signals.stats.quantiles", "TDigest"),
    ("sketches", "bloom"): ("dense_armor.utility.signals.stats.sketches", "BloomFilter"),
    ("sketches", "cms"): ("dense_armor.utility.signals.stats.sketches", "CountMinSketch"),
    ("sketches", "hll"): ("dense_armor.utility.signals.stats.sketches", "HyperLogLog"),
    ("sketches", "space_saving"): ("dense_armor.utility.signals.stats.sketches", "SpaceSaving"),
    # features / dependence
    ("dependence", "autocorr"): ("dense_armor.utility.signals.stats.dependence", "Autocorrelation"),
    ("dependence", "running_corr"): ("dense_armor.utility.signals.stats.dependence", "RunningCorrelation"),
    ("dependence", "running_cov"): ("dense_armor.utility.signals.stats.dependence", "RunningCovariance"),
    # features / moments extra
    ("moments", "rolling_mad"): ("dense_armor.utility.signals.stats.robust", "RollingMAD"),
    ("moments", "rolling_iqr"): ("dense_armor.utility.signals.stats.robust", "RollingIQR"),
    ("moments", "rolling_quantile"): ("dense_armor.utility.signals.stats.robust", "RollingQuantile"),
    # features / distribution
    ("distribution", "gmm"): ("dense_armor.utility.signals.stats.mixture", "OnlineGaussianMixture"),
    ("distribution", "student_t"): ("dense_armor.utility.signals.stats.student_t", "StudentT"),
    # risk anomaly extra
    ("anomaly", "streaming_dev"): ("dense_armor.utility.online.anomaly.deviation", "StreamingDeviationScorer"),
}




def _role_attributes(instance) -> dict:
    """Legge gli attributi dei ruoli senza modificare la primitiva."""
    out = {}  # type: ignore
    if instance is None:
        return out
    b = getattr(instance, "budget_s", None)
    if b is not None:
        try: out["budget_s"] = float(b)
        except (TypeError, ValueError): pass
    mc = getattr(instance, "memory_class", None)
    if mc is not None:
        out["memory_class"] = str(mc)  # type: ignore
    if hasattr(instance, "score_is_inverted"):
        out["score_is_inverted"] = bool(instance.score_is_inverted)
    if hasattr(instance, "drift_detected"):
        out["drift_detected"] = bool(instance.drift_detected)
    if hasattr(instance, "warning_detected"):
        out["warning_detected"] = bool(instance.warning_detected)
    nb = getattr(instance, "n_blocked_", None)
    if nb is not None:
        try: out["n_blocked"] = int(nb)
        except (TypeError, ValueError): pass
    if hasattr(instance, "health"):
        try:
            h = instance.health
            out["health"] = getattr(h, "value", str(h))  # type: ignore
        except Exception: pass
    if hasattr(instance, "coverage"):
        try: out["coverage"] = float(instance.coverage)
        except (TypeError, ValueError): pass
    at = getattr(instance, "alpha_t_", None)
    if at is not None:
        try: out["alpha_t"] = float(at)
        except (TypeError, ValueError): pass
    if hasattr(instance, "state_dict"):
        try:
            sd = instance.state_dict()
            if isinstance(sd, dict) and sd:
                out["state_keys"] = sorted(str(k) for k in sd.keys())[:20]  # type: ignore
        except Exception: pass
    u = getattr(instance, "units", None)
    if u is not None and hasattr(u, "inputs") and hasattr(u, "outputs"):
        out["units"] = {"inputs": dict(u.inputs) if u.inputs else {},  # type: ignore
                        "outputs": dict(u.outputs) if u.outputs else {}}
    return out



def _piece_prev_next(category: str, piece_id: str):
    """Ritorna (prev_id, next_id) nella stessa categoria, o (None, None)."""
    import re as _re
    m = _re.search(r'_SIGNAL_CLASSES_EXT\s*=\s*\{([\s\S]*?)\n\}', open(__file__).read())
    if not m:
        return (None, None)
    entries = _re.findall(r'\("(\w+)",\s*"(\w+)"\)', m.group(1))
    same = [pid for cat, pid in entries if cat == category]
    if piece_id not in same:
        return (None, None)
    i = same.index(piece_id)
    prev_id = same[i-1] if i > 0 else None
    next_id = same[i+1] if i < len(same)-1 else None
    return (prev_id, next_id)

def piece_meta(section: str, category: str, piece_id: str) -> dict:
    """Read the class signature and docstring for a piece."""
    key = (category, piece_id)
    if key not in _SIGNAL_CLASSES_EXT:
        return {"ok": False, "error": f"unknown piece {category}/{piece_id}"}
    mod_name, cls_name = _SIGNAL_CLASSES_EXT[key]
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    bases = [b.__name__ for b in cls.__mro__[1:] if b is not object][:5]

    sig = inspect.signature(cls.__init__)
    params = []
    for name, prm in list(sig.parameters.items())[1:]:
        if name == "self":
            continue
        default = prm.default if prm.default is not inspect._empty else None
        typ = str(prm.annotation) if prm.annotation is not inspect._empty else ""
        if isinstance(default, str):
            default_str = default
        elif default is None:
            default_str = ""
        else:
            default_str = repr(default)
        params.append({
            "name": name,
            "default": default_str,
            "type": typ,
            "required": prm.default is inspect._empty,
        })
    doc = inspect.getdoc(cls) or ""
    extra = [
        {"name": "dataset", "default": "finance", "type": "finance|manual",
         "required": False, "choices": ["finance", "manual"]},
    ]
    prev_id, next_id = _piece_prev_next(category, piece_id)
    all_params = extra + params

    return {"ok": True, "name": cls_name, "module": mod_name,
            "docstring": doc, "params": all_params,
            "bases": bases,
            "prev_id": prev_id,
            "next_id": next_id}


def _coerce(name: str, raw: str, annotation: str):
    raw = raw.strip()
    if raw == "":
        return None
    if annotation in ("int", "int | None"):
        try: return int(raw)
        except ValueError: return None
    if annotation in ("float", "float | None"):
        try: return float(raw)
        except ValueError: return None
    if annotation == "bool":
        return raw.lower() in ("true", "1", "yes")
    if annotation == "str":
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
            return raw[1:-1]
        return raw
    try: return int(raw)
    except ValueError: pass
    try: return float(raw)
    except ValueError: pass
    return raw


def piece_detail(section: str, category: str, piece_id: str, params: dict) -> dict:
    """Run one piece with the given params. Handler-specific when available."""
    if (category, piece_id) in _PIECE_FNS:
        return _PIECE_FNS[(category, piece_id)](params)
    key = (category, piece_id)
    if key not in _SIGNAL_CLASSES_EXT:
        return {"ok": False, "error": f"unknown piece {category}/{piece_id}"}
    mod_name, cls_name = _SIGNAL_CLASSES_EXT[key]
    cls = getattr(importlib.import_module(mod_name), cls_name)

    sig = inspect.signature(cls.__init__)
    kwargs = {}
    for name, prm in list(sig.parameters.items())[1:]:
        if name == "self":
            continue
        if name in params:
            typ = str(prm.annotation) if prm.annotation is not inspect._empty else ""
            try:
                v = _coerce(name, str(params[name]), typ)
                if v is not None:
                    kwargs[name] = v
            except Exception as _e:
                _swallow(_e, "line 1067")

    # Fornisci default per kwargs obbligatori noti (helper condiviso con i preview)
    kwargs = _auto_init_kwargs(cls, kwargs)

    try:
        instance = cls(**kwargs)
    except Exception as e:
        return {"ok": False, "error": f"init: {type(e).__name__}: {e}"}

    # se il detector ha una threshold, la rendo disponibile come metrica
    probe_thr = getattr(instance, "threshold", None)

    xs, ys = _gspc_classification_stream(500)
    if not xs:
        return {"ok": False, "error": "no GSPC data"}


    has_predict = hasattr(instance, "predict_one")
    has_score = hasattr(instance, "score_one")
    has_update = hasattr(instance, "update") and not hasattr(instance, "learn_one")
    has_proba = hasattr(instance, "predict_proba_one")

    # anomaly: due feature per LOF/IForest/GMM/HST (una feature è troppo piatta)
    if has_score and not has_predict:
        xs = [{"ret": x["ret"], "abs_ret": abs(x["ret"])} for x in xs]
        # riprova a istanziare con feature_keys esplicite, se accettate
        if "feature_keys" in str(sig.parameters):
            try:
                instance = cls(feature_keys=["ret", "abs_ret"], **kwargs)
            except Exception as _e:
                _swallow(_e, "line 1098")

    correct = 0
    n = 0
    curve = []
    step = max(1, len(xs) // 250)
    scores = []
    scores_curve = []
    alarms = []
    n_nodes_curve = []
    n_nodes = 0
    confusion = {f"{a}|{b}": 0 for a in (0, 1) for b in (0, 1)}

    for i, (x, y) in enumerate(zip(xs, ys)):
        try:
            if has_update:
                v = float(next(iter(x.values())))
                try:
                    instance.update(v)
                except Exception as _e:
                    _swallow(_e, "line 1118")
                if getattr(instance, "_drift_detected", False):
                    alarms.append(i)
                if i % step == 0:
                    curve.append({"i": i, "v": round(v, 6)})
            elif has_score and not has_predict:
                # anomaly: impara prima, poi score (la finestra è già popolata)
                if hasattr(instance, "learn_one"):
                    try: instance.learn_one(x)
                    except TypeError: instance.learn_one(x, y)
                try:
                    s_val = instance.score_one(x)
                    if s_val is not None and hasattr(s_val, "__float__"):
                        sv = float(s_val)
                        if sv != float("inf") and sv == sv:
                            scores.append(sv)
                            if i % step == 0:
                                scores_curve.append({"i": i, "v": round(sv, 4)})
                except Exception as _e:
                    _swallow(_e, "line 1137")
                n += 1
            else:
                pred = instance.predict_one(x) if has_predict else None
                if hasattr(instance, "learn_one"):
                    try: instance.learn_one(x, y)
                    except TypeError: instance.learn_one(x)
                if pred is not None:
                    if int(pred) == int(y):
                        correct += 1
                    if int(y) in (0, 1) and int(pred) in (0, 1):
                        confusion[f"{int(y)}|{int(pred)}"] += 1
                n += 1
                if i % step == 0:
                    lo, hi = _wilson_band(correct, n)
                    nn = int(getattr(instance, "n_nodes_", 0))
                    if nn > n_nodes:
                        n_nodes = nn
                        n_nodes_curve.append({"i": i, "v": nn})
                    curve.append({"i": i, "v": round(correct / max(n, 1), 4),
                                  "lo": round(lo, 4), "hi": round(hi, 4)})
        except Exception as _e:
            _swallow(_e, "line 1159")

    kind = ("drift" if has_update else
            "anomaly" if has_score and not has_predict else "classifier")

    metrics = {
        "accuracy": round(correct / max(n, 1), 4),
        "n": n,
        "kind": kind,
        "n_nodes": int(getattr(instance, "n_nodes_", 0)),
    
    }

    # dichiarazioni della classe
    b = getattr(instance, "budget_s", None)
    if b is not None:
        try: metrics["budget_s"] = float(b)
        except (TypeError, ValueError): pass
    mc = getattr(instance, "memory_class", None)
    if mc is not None:
        metrics["memory_class"] = str(mc)
    dtype_attr = getattr(instance, "dtype", None)
    if dtype_attr is not None:
        metrics["dtype"] = str(dtype_attr)
    else:
        try:
            metrics["dtype"] = str(instance._dtype) if hasattr(instance, "_dtype") else "float32"
        except Exception:
            metrics["dtype"] = "float32"

    # explain_one: solo per chi ce l'ha
    if hasattr(instance, "explain_one"):
        try:
            sample = xs[-1] if xs else {}
            path = instance.explain_one(sample)
            if path:
                metrics["explain"] = [
                    {"feature": str(p.get("feature")),
                     "threshold": round(float(p.get("threshold", 0)), 4),
                     "side": str(p.get("side", ""))}
                    for p in path[:10]
                ]
        except Exception as _e:
            _swallow(_e, "line 1202")

    # predict_unaggregated_one per Mondrian/AMF
    if hasattr(instance, "predict_unaggregated_one"):
        try:
            sample = xs[-1] if xs else {}
            votes = instance.predict_unaggregated_one(sample)
            if votes is not None:
                try:
                    metrics["tree_votes"] = [float(v) if not hasattr(v, "tolist")
                                             else [float(x) for x in v.tolist()[:5]]
                                             for v in list(votes)[:10]]
                except Exception as _e:
                    _swallow(_e, "line 1215")
        except Exception as _e:
            _swallow(_e, "line 1217")
    if kind == "classifier":
        metrics["confusion"] = {k: v for k, v in confusion.items() if v > 0}
        # precision / recall / F1 per classe, dalla confusion
        per_class = {}
        for c in (0, 1):
            tp = confusion.get(f"{c}|{c}", 0)
            fp = confusion.get(f"{1-c}|{c}", 0)
            fn = confusion.get(f"{c}|{1-c}", 0)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            support = tp + fn
            per_class[c] = {"precision": round(prec, 3),
                             "recall": round(rec, 3),
                             "f1": round(f1, 3),
                             "support": support}
        metrics["per_class"] = per_class
        # class representation
        metrics["class_count"] = {
            "true_0": confusion.get("0|0", 0) + confusion.get("0|1", 0),
            "true_1": confusion.get("1|0", 0) + confusion.get("1|1", 0),
        }
        # accuracy rolling su finestra 100
        correct_window = []  # type: ignore
        for i, (x, y) in enumerate(zip(xs[:n], ys[:n])):
            pass  # già calcolato, il rolling lo ricostruiamo dal cumulativo
        # approssimazione rolling: differenza del cumulativo
        if len(curve) > 10:
            rolling = []
            for j in range(1, len(curve)):
                di = curve[j]["i"] - curve[j-1]["i"]
                dv = curve[j]["v"] * (curve[j]["i"] + 1) - curve[j-1]["v"] * (curve[j-1]["i"] + 1)
                if di > 0:
                    rolling.append({"i": curve[j]["i"], "v": round(dv / di, 4)})
            metrics["rolling_accuracy"] = rolling
    if kind == "anomaly":
        # alert count sopra threshold
        thr_val = metrics.get("threshold", 1.5)
        n_above = sum(1 for sv in scores if sv > thr_val)  # type: ignore
        metrics["alerts_above_threshold"] = n_above
        # istogramma degli score (10 bin tra min e max)
        if scores:
            import numpy as _np
            arr = _np.asarray(scores)
            lo = float(arr.min())
            hi = float(arr.max())
            if hi <= lo:
                hi = lo + 1e-6
            edges = _np.linspace(lo, hi, 11)
            hist, _ = _np.histogram(arr, bins=edges)
            metrics["histogram"] = {
                "edges": [round(float(e), 4) for e in edges],
                "counts": [int(c) for c in hist],
            }
    if kind == "drift":
        import numpy as _np
        if alarms:
            metrics["first_alarm"] = int(alarms[0])
            metrics["last_alarm"] = int(alarms[-1])
        # drift score: frazione di alarm sul totale
        metrics["drift_score"] = round(len(alarms) / max(n, 1), 4)
        # tipo di drift: se gli alarm sono concentrati -> sudden, sparsi -> gradual
        if len(alarms) > 2:
            gaps = _np.diff(alarms)
            if float(_np.std(gaps)) < float(_np.mean(gaps)) * 0.3:
                metrics["drift_type"] = "sudden"
            else:
                metrics["drift_type"] = "gradual"
        else:
            metrics["drift_type"] = "none"
    if kind == "drift":
        metrics["alarms"] = len(alarms)
    if scores:
        import numpy as _np
        metrics["score_mean"] = round(float(_np.mean(scores)), 4)
        metrics["score_p95"] = round(float(_np.percentile(scores, 95)), 4)
    if probe_thr is not None:
        try:
            tv = float(probe_thr)
            import math as _m
            if _m.isfinite(tv) and 0 < tv < 100:
                metrics["threshold"] = tv
        except (TypeError, ValueError):
            pass

    metrics.update(_role_attributes(instance))

    return {
        "ok": True,
        "series": curve if kind != "anomaly" else scores_curve,
        "drift_markers": n_nodes_curve[:30],
        "alarms": alarms[:200],
        "metrics": metrics,
    }


# ============ pezzo-specifico, come da docs/tree/index.md §1 ============
def _finance_features(n: int = 1500, change_at_frac: float = 0.5):
    """Finance dataset with declared volatility regime shift.
    First half: normal vol. Second half: vol x 2 (dichiarato, come manuale §6).
    Target: high-volatility regime next day (learnable by construction)."""
    import numpy as _np
    rets = _returns_preview("test/fixtures/finance/GSPC.csv", n=1500)
    if len(rets) < 60:
        return [], []
    arr = _np.asarray(rets, dtype=float)  # type: ignore
    n_total = len(arr)
    change_at = int(n_total * change_at_frac)
    # inietta regime shift dichiarato: dopo change_at, amplifica la volatilità x2
    rng = _np.random.default_rng(42)
    amplified = arr.copy()
    amplified[change_at:] = amplified[change_at:] * 2.0 + rng.normal(0, 0.005, n_total - change_at)
    arr = amplified
    abs_rets = _np.abs(arr)
    med_thr = float(_np.median(abs_rets[-60:]))
    X, Y = [], []
    for k in range(30, len(arr) - 1):
        window = arr[max(0, k - 20):k + 1]
        def ema(data, span):
            alpha = 2.0 / (span + 1)
            e = data[0]
            for v in data[1:]:
                e = alpha * v + (1 - alpha) * e
            return e
        gains = sum(max(0, arr[i]) for i in range(max(0, k - 13), k + 1))  # type: ignore
        losses = sum(max(0, -arr[i]) for i in range(max(0, k - 13), k + 1))  # type: ignore
        rsi = 100.0 * gains / (gains + losses + 1e-9)
        ema12 = ema(window, 12)
        ema26 = ema(window, 26)
        x = {
            "ret_1": float(arr[k]),
            "ret_5": float(arr[k] - arr[k - 5]),
            "ret_10": float(arr[k] - arr[k - 10]),
            "mom_5": float(_np.sum(arr[k - 4:k + 1])),
            "mom_10": float(_np.sum(arr[k - 9:k + 1])),
            "vol_20": float(_np.std(window)),
            "rsi_14": float(rsi),
            "macd": float(ema12 - ema26),
        }
        y = int(abs(arr[k + 1]) > med_thr)
        X.append(x)
        Y.append(y)
    return X, Y



def _synthetic_features(n: int = 2000, change_at: int = 1000):
    """Manuale §6: a ~ N(3y, 0.5) poi cambia, b ~ N(0, 0.5) poi cambia."""
    import numpy as _np
    rng = _np.random.default_rng(2)
    X, Y = [], []
    for i in range(n):
        y = int(rng.random() < 0.5)
        a = float(rng.normal(3.0 * y if i < change_at else 0.0, 0.5))
        b = float(rng.normal(3.0 * y if i >= change_at else 0.0, 0.5))
        X.append({"a": a, "b": b})
        Y.append(y)
    return X, Y


def _piece_hoeffding(params: dict) -> dict:
    import numpy as _np
    from dense_armor.utility.online.tree.hoeffding import HoeffdingTreeClassifier

    dataset = str(params.pop("dataset", "manual")).lower()

    sig = inspect.signature(HoeffdingTreeClassifier.__init__)
    kwargs = {}
    for name, prm in list(sig.parameters.items())[1:]:
        if name in params:
            typ = str(prm.annotation) if prm.annotation is not inspect._empty else ""
            v = _coerce(name, str(params[name]), typ)
            if v is not None:
                kwargs[name] = v
    if "grace_period" not in kwargs:
        kwargs["grace_period"] = 20

    try:
        ht = HoeffdingTreeClassifier(**kwargs)
    except Exception as e:
        return {"ok": False, "error": f"init: {type(e).__name__}: {e}"}

    if dataset == "manual":
        X, Y = _synthetic_features(n=2000, change_at=1000)
    else:
        X, Y = _finance_features(n=1500)

    if not X:
        return {"ok": False, "error": "dataset vuoto"}

    correct = 0
    acc_curve = []
    y_true, y_pred = [], []
    proba_list = []
    step = max(1, len(X) // 250)
    for i, (x, y) in enumerate(zip(X, Y)):
        pred = ht.predict_one(x)
        if pred is not None:
            y_true.append(y)
            y_pred.append(int(pred))
            if int(pred) == int(y):
                correct += 1
        try:
            proba = ht.predict_proba_one(x) if hasattr(ht, "predict_proba_one") else None
            proba_list.append(float(proba.get(1, 0.5)) if proba else 0.5)
        except Exception:
            proba_list.append(0.5)
        ht.learn_one(x, y)
        if i % step == 0:
            acc_curve.append({"i": i, "v": round(correct / (i + 1), 4)})

    cm = {"00": 0, "01": 0, "10": 0, "11": 0}
    for t, pv in zip(y_true, y_pred):
        cm[f"{t}{pv}"] += 1

    per_class = {}
    for c in (0, 1):
        tp = cm[f"{c}{c}"]
        fp = cm[f"{1-c}{c}"]
        fn = cm[f"{c}{1-c}"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[c] = {"precision": round(prec, 3), "recall": round(rec, 3),
                        "f1": round(f1, 3), "support": tp + fn}

    # ROC + AUC
    roc = []
    auc = None
    try:
        scores = list(zip(proba_list, y_true))
        scores.sort(reverse=True)
        P = sum(1 for _, t in scores if t == 1)
        N = sum(1 for _, t in scores if t == 0)
        tps, fps = 0, 0
        auc = 0.0
        prev_fpr = 0.0
        roc.append({"fpr": 0.0, "tpr": 0.0})
        for sc, t in scores:
            if t == 1: tps += 1
            else: fps += 1
            tpr = tps / P if P else 0
            fpr = fps / N if N else 0
            auc += (fpr - prev_fpr) * tpr
            prev_fpr = fpr
            roc.append({"fpr": round(fpr, 4), "tpr": round(tpr, 4)})
        auc = round(float(auc), 4)
    except Exception:
        auc = None

    # log loss
    import math
    ll = 0.0
    for pv, t in zip(proba_list, y_true):  # type: ignore
        eps = 1e-12
        pv = min(max(pv, eps), 1 - eps)  # type: ignore
        ll -= t * math.log(pv) + (1 - t) * math.log(1 - pv)
    logloss = round(ll / max(len(y_true), 1), 4)

    # calibration (reliability diagram): 10 bins
    calibration = []
    try:
        arr_p = _np.asarray(proba_list)
        arr_t = _np.asarray(y_true)
        edges = _np.linspace(0, 1, 11)
        for b in range(10):
            lo, hi = edges[b], edges[b+1]
            mask = (arr_p >= lo) & (arr_p < hi) if b < 9 else (arr_p >= lo) & (arr_p <= hi)
            if mask.sum() > 0:
                calibration.append({
                    "bin": round((lo + hi) / 2, 2),
                    "count": int(mask.sum()),
                    "freq": round(float(arr_t[mask].mean()), 4),
                })
    except Exception as _e:
        _swallow(_e, "line 1493")

    metrics = {
        "kind": "classifier", "dataset": dataset,
        "accuracy": round(correct / max(len(y_true), 1), 4),
        "precision": per_class[1]["precision"],
        "recall": per_class[1]["recall"],
        "f1": per_class[1]["f1"],
        "roc_auc": auc,
        "logloss": logloss,
        "n": len(y_true),
        "n_nodes": int(getattr(ht, "n_nodes_", 0)),
        "confusion": {"true=0,pred=0": cm["00"], "true=0,pred=1": cm["01"],
                      "true=1,pred=0": cm["10"], "true=1,pred=1": cm["11"]},
        "per_class": per_class,
        "features": list(X[0].keys()),
    }
    b = getattr(ht, "budget_s", None)  # type: ignore
    if b is not None: metrics["budget_s"] = float(b)
    mc = getattr(ht, "memory_class", None)
    if mc is not None: metrics["memory_class"] = str(mc)
    d = getattr(ht, "dtype", None)
    if d is not None: metrics["dtype"] = str(d)

    return {
        "ok": True, "series": acc_curve, "roc": roc,
        "calibration": calibration, "metrics": metrics,
    }



_PIECE_FNS = {
    ("trees", "hoeffding"): _piece_hoeffding,
}
