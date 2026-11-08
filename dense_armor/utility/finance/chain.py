"""The six sections of the chain.

Data -> Features -> Signal -> Portfolio -> Risk -> Execution.
Each section has its own dropdown in /chain. The catalog is static: a
class enters only after it is listed here and passes the run on GSPC.
River is a subset of this library, not the other way around.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as _np


CATALOG: dict[str, dict[str, Any]] = {
    "data": {
        "gspc":  {"kind": "finance", "source": "GSPC.csv",  "target": "sign next log return"},
        "aapl":  {"kind": "finance", "source": "AAPL.csv",  "target": "sign next log return"},
        "msft":  {"kind": "finance", "source": "MSFT.csv",  "target": "sign next log return"},
        "tsla":  {"kind": "finance", "source": "TSLA.csv",  "target": "sign next log return"},
        "driftstream_sudden":      {"kind": "synthetic", "source": "DriftStream sudden",      "target": "y"},
        "driftstream_gradual":     {"kind": "synthetic", "source": "DriftStream gradual",     "target": "y"},
        "driftstream_incremental": {"kind": "synthetic", "source": "DriftStream incremental", "target": "y"},
    },
    "features": {
        "none": None,
        "standard_scaler": "dense_armor.utility.signals.preprocessing.scale.StandardScaler",
        "minmax_scaler":   "dense_armor.utility.signals.preprocessing.scale.MinMaxScaler",
        "robust_scaler":   "dense_armor.utility.signals.preprocessing.scale.RobustScaler",
        "ew_scaler":       "dense_armor.utility.signals.preprocessing.scale.EWScaler",
        "select_kbest":    "dense_armor.utility.signals.preprocessing.select.SelectKBest",
        "var_threshold":   "dense_armor.utility.signals.preprocessing.select.VarianceThreshold",
        "running_moments":     "dense_armor.utility.signals.stats.moments.RunningMoments",
        "ew_stats":            "dense_armor.utility.signals.stats.moments.EWStats",
        "rolling_median":      "dense_armor.utility.signals.stats.robust.RollingMedian",
        "rolling_mad":         "dense_armor.utility.signals.stats.robust.RollingMAD",
        "rolling_iqr":         "dense_armor.utility.signals.stats.robust.RollingIQR",
        "rolling_quantile":    "dense_armor.utility.signals.stats.robust.RollingQuantile",
        "ddsketch":            "dense_armor.utility.signals.stats.quantiles.DDSketch",
        "tdigest":             "dense_armor.utility.signals.stats.quantiles.TDigest",
        "running_correlation": "dense_armor.utility.signals.stats.dependence.RunningCorrelation",
        "running_covariance":  "dense_armor.utility.signals.stats.dependence.RunningCovariance",
        "online_shrinkage_cov":"dense_armor.utility.signals.stats.shrinkage.OnlineShrinkageCovariance",
        "autocorrelation":     "dense_armor.utility.signals.stats.dependence.Autocorrelation",
        "bloom":               "dense_armor.utility.signals.stats.sketches.BloomFilter",
        "cms":                 "dense_armor.utility.signals.stats.sketches.CountMinSketch",
        "hll":                 "dense_armor.utility.signals.stats.sketches.HyperLogLog",
        "space_saving":        "dense_armor.utility.signals.stats.sketches.SpaceSaving",
        "sdft":                "dense_armor.utility.signals.stats.sdft.SDFT",
        "skyline":             "dense_armor.utility.signals.stats.skyline.Skyline",
        "garch":               "dense_armor.utility.online.timeseries.garch.GARCH11",
        "online_ar":           "dense_armor.utility.online.timeseries.ar.OnlineAR",
        "online_var":          "dense_armor.utility.online.timeseries.ar.OnlineVAR",
        "kalman":              "dense_armor.utility.online.timeseries.kalman.KalmanFilter",
        "innovation_kalman":   "dense_armor.utility.online.timeseries.kalman.InnovationAdaptiveKalman",
        "vb_kalman":           "dense_armor.utility.online.timeseries.kalman.VBAdaptiveKalman",
        "cycle_residual":      "dense_armor.utility.online.timeseries.cycle.CycleResidual",
        "gmm":                 "dense_armor.utility.signals.stats.mixture.OnlineGaussianMixture",
        "beta":                "dense_armor.utility.online.proba.beta.Beta",
        "gaussian":            "dense_armor.utility.online.proba.gaussian.Gaussian",
        "multinomial_dist":    "dense_armor.utility.online.proba.multinomial.Multinomial",
    },
    "signal": {
        "hoeffding":      "dense_armor.utility.online.tree.hoeffding.HoeffdingTreeClassifier",
        "anytime":        "dense_armor.utility.online.tree.efdt.HoeffdingAnytimeTreeClassifier",
        "adaptive":       "dense_armor.utility.online.tree.adaptive.HoeffdingAdaptiveTreeClassifier",
        "mondrian":       "dense_armor.utility.online.tree.mondrian.MondrianForestClassifier",
        "mondrian_reg":   "dense_armor.utility.online.tree.mondrian.MondrianForestRegressor",
        "sgt_clf":        "dense_armor.utility.online.tree.sgt.SGTClassifier",
        "sgt_reg":        "dense_armor.utility.online.tree.sgt.SGTRegressor",
        "batch_tree":     "dense_armor.utility.online.tree.batch.BatchDecisionTreeClassifier",
        "fimtdd":         "dense_armor.utility.online.tree.fimt.FIMTDDRegressor",
        "amf":            "dense_armor.utility.online.forest.amf.AggregatedMondrianForestClassifier",
        "amf_reg":        "dense_armor.utility.online.forest.amf.AggregatedMondrianForestRegressor",
        "arf":            "dense_armor.utility.online.forest.arf.AdaptiveRandomForestClassifier",
        "oxt":            "dense_armor.utility.online.forest.oxt.OnlineExtraTreesRegressor",
        "subspace_ht":    "dense_armor.utility.online.forest.arf.SubspaceHoeffdingTree",
        "gaussian_nb":    "dense_armor.utility.online.learn.online_classifiers.OnlineGaussianNB",
        "softmax":        "dense_armor.utility.online.learn.online_classifiers.OnlineSoftmaxRegression",
        "platt":          "dense_armor.utility.online.learn.calibration.OnlinePlattScaling",
        "rls":            "dense_armor.utility.online.learn.online_dynamics.RecursiveLeastSquares",
        "metric_knn":     "dense_armor.utility.online.learn.metric_learning.MetricKNNClassifier",
        "logistic":       "dense_armor.utility.online.linear.glm.LogisticRegression",
        "bayes_gauss":    "dense_armor.utility.online.linear.bayes.BayesianLinearRegression",
        "multinomial_nb": "dense_armor.utility.online.bayes.multinomial.MultinomialNB",
        "bernoulli_nb":   "dense_armor.utility.online.bayes.bernoulli.BernoulliNB",
        "complement_nb":  "dense_armor.utility.online.bayes.complement.ComplementNB",
        "categorical_nb": "dense_armor.utility.online.bayes.categorical.CategoricalNB",
        "knn":            "dense_armor.utility.online.neighbors.knn.KNNClassifier",
        "knn_reg":        "dense_armor.utility.online.neighbors.knn.KNNRegressor",
        "bagging":        "dense_armor.utility.online.ensemble.bagging.OnlineBaggingClassifier",
        "boosting":       "dense_armor.utility.online.ensemble.boosting.OnlineBoostingClassifier",
        "stacking":       "dense_armor.utility.online.ensemble.stacking.StackingClassifier",
        "voting":         "dense_armor.utility.online.ensemble.voting.VotingClassifier",
        "ewa_reg":        "dense_armor.utility.online.ensemble.ewa.ExponentiallyWeightedAverageRegressor",
        "amrules":        "dense_armor.utility.online.rules.amrules.AMRulesRegressor",
        "ovr":            "dense_armor.utility.online.multiclass.multiclass.OneVsRestClassifier",
        "ovo":            "dense_armor.utility.online.multiclass.multiclass.OneVsOneClassifier",
        "output_code":    "dense_armor.utility.online.multiclass.multiclass.OutputCodeClassifier",
        "chain_clf":      "dense_armor.utility.online.multioutput.multioutput.ClassifierChain",
        "chain_reg":      "dense_armor.utility.online.multioutput.multioutput.RegressorChain",
        "fm":             "dense_armor.utility.online.facto.fm.FMClassifier",
        "ffm":            "dense_armor.utility.online.facto.ffm.FFMClassifier",
        "fwfm":           "dense_armor.utility.online.facto.fwfm.FwFMClassifier",
        "hofm":           "dense_armor.utility.online.facto.hofm.HOFMClassifier",
        "baseline":       "dense_armor.utility.online.reco.reco.Baseline",
        "biased_mf":      "dense_armor.utility.online.reco.reco.BiasedMF",
        "funk_mf":        "dense_armor.utility.online.reco.reco.FunkMF",
        "denstream":      "dense_armor.utility.online.cluster.denstream.DenStream",
        "online_kmeans":  "dense_armor.utility.online.cluster.kmeans.OnlineKMeans",
    },
    "portfolio": {
        "none": None,
        "min_variance":       "pipelines.finance.portfolio.min_variance_weights",
        "inverse_volatility": "pipelines.finance.portfolio.inverse_volatility_weights",
        "pipeline":           "pipelines.finance.portfolio.PortfolioPipeline",
    },
    "risk": {
        "none": None,
        "lof":            "dense_armor.utility.online.anomaly.lof.LocalOutlierFactor",
        "hst":            "dense_armor.utility.online.anomaly.hst.HalfSpaceTrees",
        "loda":           "dense_armor.utility.online.anomaly.loda.LODA",
        "iforest":        "dense_armor.utility.online.anomaly.oiforest.OnlineIsolationForest",
        "mahalanobis":    "dense_armor.utility.online.anomaly.mahalanobis.OnlineRobustMahalanobis",
        "ocsvm":          "dense_armor.utility.online.anomaly.ocsvm.OneClassSGD",
        "hampel_scorer":  "dense_armor.utility.online.anomaly.filters.HampelScorer",
        "chauvenet":      "dense_armor.utility.online.anomaly.filters.ChauvenetScorer",
        "sigma_clip":     "dense_armor.utility.online.anomaly.filters.SigmaClipScorer",
        "tukey":          "dense_armor.utility.online.anomaly.filters.TukeyScorer",
        "adwin":          "dense_armor.utility.online.drift.adwin.ADWIN",
        "page_hinkley":   "dense_armor.utility.online.drift.page_hinkley.PageHinkley",
        "kswin":          "dense_armor.utility.online.drift.kswin.KSWIN",
        "cusum":          "dense_armor.utility.online.drift.detector.CUSUMDriftDetector",
        "bocpd":          "dense_armor.utility.online.drift.bocpd.BayesianOnlineChangepoint",
        "surveillance":   "pipelines.finance.surveillance.FinancePipeline",
        "risk_pipeline":  "pipelines.finance.risk.RiskPipeline",
        "regime_pipeline":"pipelines.finance.regime.RegimePipeline",
        "drift_vs_noise": "pipelines.finance.drift_vs_noise.DriftVsNoisePipeline",
        "pretrade":       "dense_armor.utility.finance.pretrade.PreTradeChecker",
    },
    "execution": {
        "none": None,
        "pipeline":    "pipelines.finance.execution.ExecutionPipeline",
        "fix_gateway": "dense_armor.utility.finance.fix_gateway.send_orders",
        "twap":        "dense_armor.utility.finance.schedule.twap_schedule",
        "vwap":        "dense_armor.utility.finance.schedule.vwap_schedule",
    },
}


def _auto_defaults(obj) -> dict:
    """Best-effort defaults for classes that require positional arguments."""
    import inspect
    try:
        sig = inspect.signature(obj.__init__)
    except (ValueError, TypeError):
        return {}
    try:
        from dense_armor.utility.online.drift.adwin import ADWIN
        from dense_armor.utility.online.tree.hoeffding import HoeffdingTreeClassifier
        from dense_armor.utility.online.optim.sgd import SGD
    except Exception:
        return {}
    import numpy as _np
    out = {}
    for name, prm in list(sig.parameters.items())[1:]:
        if prm.default is not prm.empty:
            continue
        low = name.lower()
        try:
            if low in ("drift_detector", "warning_detector"):
                out[name] = ADWIN()
            elif low == "classifier":
                out[name] = HoeffdingTreeClassifier()
            elif low == "metric_learner":
                try:
                    from dense_armor.utility.online.learn.metric_learning import LEGO
                    out[name] = LEGO()
                except Exception:
                    return {}
            elif low in ("model", "estimator", "base_model", "base_estimator",
                         "meta_model"):
                out[name] = HoeffdingTreeClassifier()
            elif low == "optimizer":
                out[name] = SGD()
            elif low in ("k", "n_clusters"):
                out[name] = 3
            elif low in ("n_features", "n_inputs", "dim"):
                out[name] = 4
            elif low == "mode":
                out[name] = "classification"
            elif low in ("n_models", "n_learners", "n_members"):
                out[name] = 10
            elif low in ("members", "base_estimators", "estimators",
                         "learners", "models"):
                out[name] = [lambda: HoeffdingTreeClassifier()]
            elif low in ("binary_factory", "factory", "model_factory"):
                out[name] = lambda: HoeffdingTreeClassifier()
            elif low == "metric_model":
                out[name] = HoeffdingTreeClassifier()
            elif low == "base":
                out[name] = HoeffdingTreeClassifier()
            elif low == "fields":
                out[name] = {}
            elif low in ("range_ret",):
                out[name] = (-0.01, 0.01)
            elif low in ("range_vol",):
                out[name] = (0.5, 1.5)
            elif low == "f":
                out[name] = _np.eye(1)
            elif low == "h":
                out[name] = _np.eye(1)
            elif low == "q":
                out[name] = _np.eye(1) * 0.01
            elif low == "r":
                out[name] = _np.eye(1) * 0.1
            elif low.startswith("max_") or low in ("price_band", "window_s",
                                                    "duplicate_window_s"):
                out[name] = 1.0
            elif low == "n":
                out[name] = 4
            else:
                return {}
        except Exception:
            return {}
    return out


def _probe(kind: str, path) -> dict:
    """Try to instantiate. Return ready / needs_args / func / stream."""
    if path is None:
        return {"status": "none"}
    if kind == "data":
        return {"status": "stream", "source": path.get("source", "")}
    mod_name, name = path.rsplit(".", 1)
    try:
        mod = importlib.import_module(mod_name)
        obj = getattr(mod, name)
    except Exception as e:
        return {"status": "import_error", "error": f"{type(e).__name__}: {e}"}
    import inspect
    if inspect.isfunction(obj):
        return {"status": "func"}
    if not inspect.isclass(obj):
        return {"status": "not_a_class"}
    # Special case: some classes need a keyword even if None is the default.
    try:
        if name == "HoeffdingAdaptiveTreeClassifier":
            from dense_armor.utility.online.drift.adwin import ADWIN
            obj(drift_detector=ADWIN())
            return {"status": "ready", "default_init": {"drift_detector": "ADWIN()"}}
    except Exception:
        pass
    try:
        obj()
        return {"status": "ready"}
    except Exception as e0:
        kwargs = _auto_defaults(obj)
        if not kwargs:
            return {"status": "needs_args",
                    "error": f"{type(e0).__name__}: {str(e0)[:100]}"}
        try:
            obj(**kwargs)
            return {"status": "ready", "default_init": kwargs}
        except Exception as e1:
            return {"status": "needs_args",
                    "error": f"{type(e1).__name__}: {str(e1)[:100]}"}


def describe_catalog() -> dict:
    """Scan and report the status of every entry, section by section."""
    out = {}
    for section, entries in CATALOG.items():
        rows = []
        for k, v in entries.items():
            p = _probe(section, v)
            rows.append({"id": k, **p})
        out[section] = rows
    return out




def _instantiate(path: str | None) -> Any:
    """Import the class from path and instantiate it, applying defaults."""
    if path is None:
        return None
    mod_name, name = path.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, name)
    import inspect
    if inspect.isfunction(cls):
        return cls
    try:
        return cls()
    except Exception:
        kwargs = _auto_defaults(cls)
        if kwargs:
            return cls(**kwargs)
        raise

def _load_data(data_id: str, n: int, seed: int):
    """Return (rows, ys, meta) from the data section."""
    if data_id.startswith("driftstream"):
        from dense_armor.utility.signals.datasets import DriftStream
        kind = data_id.replace("driftstream_", "")
        ds = DriftStream(n_samples=n, n_features=4, n_drifts=1, kind=kind, seed=seed)
        X, y = ds.data()
        names = [f"x{i}" for i in range(X.shape[1])]
        rows = [{k: float(X[i, j]) for j, k in enumerate(names)} for i in range(n)]
        return rows, [int(v) for v in y], {"kind": "synthetic", "source": data_id}
    ticker = data_id.upper()
    path = f"test/fixtures/finance/{ticker}.csv"
    if not Path(path).exists():
        raise RuntimeError(f"{path} missing")
    rows_raw = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    start = next((i + 1 for i, ln in enumerate(rows_raw)
                  if ln.split(",")[0].strip().lower() == "date"), 0)
    closes = []
    for ln in rows_raw[start:]:
        parts = ln.split(",")
        if len(parts) < 2:
            continue
        try:
            closes.append(float(parts[1]))
        except ValueError:
            continue
    import numpy as _np
    arr = _np.asarray(closes, dtype=float)
    if len(arr) < 3:
        raise RuntimeError(f"{path}: too few rows")
    if n < len(arr) - 1:
        arr = arr[-n - 1:]
    rets = _np.diff(_np.log(arr))
    rows = [{"ret": float(rets[i])} for i in range(len(rets) - 1)]
    ys = [int(rets[i + 1] > 0) for i in range(len(rets) - 1)]
    return rows, ys, {"kind": "finance", "source": ticker + ".csv", "n": len(rows)}


def _maybe_call(obj, **kwargs):
    """Call obj(**kwargs) ignoring params it does not accept."""
    import inspect
    try:
        sig = inspect.signature(obj.__init__)
        accepted = set(sig.parameters)
    except (ValueError, TypeError):
        return obj
    return obj(**{k: v for k, v in kwargs.items() if k in accepted})


def run_chain(section_picks: dict, n: int = 500, seed: int = 0) -> dict:
    """Run the six-section chain. section_picks has keys:
    data, features, signal, portfolio, risk, execution.
    """
    for s in ("data", "features", "signal", "portfolio", "risk", "execution"):
        if s not in section_picks:
            return {"ok": False, "error": f"missing section {s}"}
        if section_picks[s] not in CATALOG[s]:
            return {"ok": False, "error": f"unknown {s}: {section_picks[s]!r}"}

    data_id = section_picks["data"]
    feat_id = section_picks["features"]
    sig_id  = section_picks["signal"]
    port_id = section_picks["portfolio"]
    risk_id = section_picks["risk"]
    exec_id = section_picks["execution"]

    try:
        rows, ys, meta = _load_data(data_id, n, seed)
    except Exception as e:
        return {"ok": False, "error": f"data: {type(e).__name__}: {e}"}

    try:
        feat = _instantiate(CATALOG["features"][feat_id])
        sig_obj = _instantiate(CATALOG["signal"][sig_id])
        risk_obj = _instantiate(CATALOG["risk"][risk_id])
    except Exception as e:
        return {"ok": False, "error": f"instantiate: {e}"}

    # portfolio ed execution: classi o funzioni
    port_callable = None
    if port_id != "none":
        mod_name, name = CATALOG["portfolio"][port_id].rsplit(".", 1)
        import importlib
        m = importlib.import_module(mod_name)
        port_callable = getattr(m, name)

    exec_callable = None
    if exec_id != "none":
        mod_name, name = CATALOG["execution"][exec_id].rsplit(".", 1)
        import importlib
        m = importlib.import_module(mod_name)
        exec_callable = getattr(m, name)

    total = len(rows)
    step = max(1, total // 140)
    acc_series, feat_series, risk_series = [], [], []
    alarms = []
    correct = 0
    risk_alarms = 0

    for i, (x, y) in enumerate(zip(rows, ys)):
        x_used = x
        if feat is not None:
            try:
                feat.learn_one(x)
                tr = feat.transform_one(x)
                if tr:
                    x_used = tr
                    if i % step == 0:
                        first_val = next(iter(tr.values()), None)
                        if first_val is not None and hasattr(first_val, "__float__"):
                            feat_series.append({"i": i, "v": round(float(first_val), 6)})
            except Exception:
                pass
        try:
            pred = sig_obj.predict_one(x_used)
            sig_obj.learn_one(x_used, y)
            if pred is not None and int(pred) == int(y):
                correct += 1
            if i % step == 0:
                acc_series.append({"i": i, "v": round(correct / (i + 1), 4)})
        except Exception:
            pass
        if risk_obj is not None:
            try:
                if hasattr(risk_obj, "update") and not hasattr(risk_obj, "learn_one"):
                    flag = risk_obj.update(float(next(iter(x.values()))))
                    detected = bool(getattr(risk_obj, "_drift_detected", flag))
                    if detected:
                        risk_alarms += 1
                        alarms.append(i)
                elif hasattr(risk_obj, "score_one"):
                    s = risk_obj.score_one(x_used)
                    risk_obj.learn_one(x_used)
                    if s is not None and float(s) > 1.5:
                        risk_alarms += 1
                        alarms.append(i)
                elif hasattr(risk_obj, "learn_one"):
                    risk_obj.learn_one(x_used)
                if i % step == 0:
                    risk_series.append({"i": i, "v": int(risk_alarms)})
            except Exception:
                pass

    # portfolio: usa load_panel (4 asset GSPC, AAPL, MSFT, TSLA) e cov 4x4
    port_out = None
    if port_callable is not None:
        try:
            import numpy as _np
            from pipelines.finance.portfolio import load_panel
            panel = load_panel("test/fixtures/finance")
            cov = _np.cov(panel, rowvar=False)
            if port_id == "pipeline":
                pipe = port_callable()
                result = pipe.run(panel)
                port_out = {
                    "weights_min_var": [round(float(v), 4)
                                        for v in result.weights_min_var[:4]],
                    "weights_inv_vol": [round(float(v), 4)
                                        for v in result.weights_inv_vol[:4]],
                    "realised_vol_min_var": round(result.realised_vol_min_var, 5),
                    "realised_vol_inv_vol": round(result.realised_vol_inv_vol, 5),
                    "n_samples": int(result.n_samples),
                }
            else:
                w = port_callable(cov)
                port_out = {"weights": [round(float(v), 4) for v in w]}
        except Exception as e:
            port_out = {"error": f"{type(e).__name__}: {e}"}

    # execution: prova su un ordine sintetico (dimensione = media del segnale)
    exec_out = None
    if exec_callable is not None:
        try:
            if "ExecutionPipeline" in str(exec_callable):
                pipe = exec_callable()
                r = pipe.process_order(size=100.0)
                exec_out = {"sent_size": float(r.sent_size),
                            "approved": bool(r.approved),
                            "reason": r.reason}
            else:
                exec_out = {"note": "callable, no live acceptor in chain"}
        except Exception as e:
            exec_out = {"error": f"{type(e).__name__}: {e}"}

    # legge gli attributi dei ruoli delle primitive effettivamente usate
    from dense_armor.utility.finance.sections import _role_attributes
    role_attrs = {}
    for _name, _obj in (("features", feat), ("signal", sig_obj), ("risk", risk_obj)):
        role_attrs[_name] = _role_attributes(_obj)
    total_budget = 0.0
    budget_missing = []
    for _name, _attrs in role_attrs.items():
        if "budget_s" in _attrs:
            total_budget += _attrs["budget_s"]
        else:
            budget_missing.append(_name)

    return {
        "ok": True,
        "chain": dict(section_picks, n=n, seed=seed),
        "data_meta": meta,
        "series": {"accuracy": acc_series, "features": feat_series,
                   "risk": risk_series},
        "alarms": alarms[:200],
        "metrics": {
            "accuracy": round(correct / max(total, 1), 4),
            "n": total,
            "risk_alarms": risk_alarms,
            "portfolio": port_out,
            "execution": exec_out,
            "role_attributes": role_attrs,
            "total_budget_s": round(total_budget, 6) if not budget_missing else None,
            "budget_missing": budget_missing,
        },
    }
