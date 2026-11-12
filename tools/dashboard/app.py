"""Dashboard: one page per finance primitive, plus the emergency stop."""

from __future__ import annotations

import argparse
import html
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dense_armor.core.finance.config import load_config
from dense_armor.core.finance.decision_log import DecisionLog
from dense_armor.core.finance.feed import frame, iter_itch
from dense_armor.core.finance.itch import decode_message
from dense_armor.core.finance.pretrade import Decision, PreTradeChecker
from dense_armor.core.finance.ring import RingBuffer
from dense_armor.core.finance.schedule import twap_schedule, vwap_schedule
from dense_armor.core.finance.status import collect_status
from dense_armor.core.finance.synthetic import synthetic_itch

STATIC_DIR = Path(__file__).resolve().parent / "static"
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")

# The playground runs the code that a visitor posts. Its subprocess gets only these variables of the
# environment of the server (and PYTHONPATH), never the rest: the database URL and any token stay out.
_PLAYGROUND_ENV_KEYS = (
    "PATH", "SYSTEMROOT", "SYSTEMDRIVE", "TEMP", "TMP", "TMPDIR", "HOME", "USERPROFILE", "LANG", "LC_ALL",
)


def _playground_env(pythonpath: str) -> dict[str, str]:
    """The environment of the playground subprocess: the allowed keys of ``os.environ`` and PYTHONPATH."""
    env = {k: os.environ[k] for k in _PLAYGROUND_ENV_KEYS if k in os.environ}
    env["PYTHONPATH"] = pythonpath
    return env


def _iter_decisions(log_path: Path) -> list[dict]:
    """Read the log file and return the list of decision dicts."""
    out: list[dict] = []
    if not log_path.exists():
        return out
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = obj.get("payload") if isinstance(obj, dict) else None
        if not isinstance(payload, dict):
            continue
        out.append({
            "seq": obj.get("seq"),
            "t": obj.get("t"),
            "reason": payload.get("reason"),
            "stock": payload.get("stock"),
            "approved": payload.get("approved"),
            "order_ref": payload.get("order_ref"),
            "hash": obj.get("hash"),
        })
    return out


def _count_reasons(log_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for d in _iter_decisions(log_path):
        r = d["reason"]
        if isinstance(r, str):
            counts[r] = counts.get(r, 0) + 1
    return counts




    mod_name, cls_name = class_path.rsplit(".", 1)
    module = _im.import_module(mod_name)
    cls = getattr(module, cls_name)
    try:
        instance = cls(**params)
    except TypeError as e:
        return {"ok": False, "error": f"instantiation failed: {e}"}

    from dense_armor.core.signals.datasets import DriftStream
    ds = DriftStream(n_samples=n, n_features=4, n_drifts=1, kind="sudden", seed=seed)
    X, y = ds.data()
    names = [f"x{i}" for i in range(X.shape[1])]
    rows = [{k: float(X[i, j]) for j, k in enumerate(names)} for i in range(n)]

    has_predict = hasattr(instance, "predict_one")
    has_proba = hasattr(instance, "predict_proba_one")
    has_score = hasattr(instance, "score_one")
    has_transform = hasattr(instance, "transform_one")
    has_update = hasattr(instance, "update") and not hasattr(instance, "learn_one")
    has_bandit = hasattr(instance, "pull")

    t0 = _t.perf_counter()
    try:
        if has_bandit:
            rewards = 0.0
            for i in range(n):
                try:
                    a = instance.pull()
                    r = 1.0 if (i % 2) == 0 else 0.0
                    instance.update(a, r)
                    rewards += r
                except Exception:
                    pass
            elapsed = _t.perf_counter() - t0
            return {"ok": True, "kind": "bandit", "n": n,
                    "total_reward": round(rewards, 3),
                    "elapsed_s": round(elapsed, 3)}

        if has_update:
            hits = 0
            for i in range(n):
                try:
                    if instance.update(float(X[i, 0])):
                        hits += 1
                except Exception:
                    pass
            elapsed = _t.perf_counter() - t0
            return {"ok": True, "kind": "drift", "n": n, "drifts": hits,
                    "elapsed_s": round(elapsed, 3)}

        if has_proba:
            correct = 0
            for i in range(n):
                try:
                    p = instance.predict_one(rows[i])
                    instance.learn_one(rows[i], int(y[i]))
                    if p is not None and int(p) == int(y[i]):
                        correct += 1
                except Exception:
                    pass
            elapsed = _t.perf_counter() - t0
            return {"ok": True, "kind": "classifier", "n": n,
                    "accuracy": round(correct / max(n, 1), 4),
                    "elapsed_s": round(elapsed, 3)}

        if has_predict and not has_proba:
            errs, c = 0.0, 0
            for i in range(n):
                try:
                    p = instance.predict_one(rows[i])
                    if p is not None and hasattr(p, "__float__"):
                        errs += (float(p) - float(y[i])) ** 2
                        c += 1
                    instance.learn_one(rows[i], float(y[i]))
                except Exception:
                    pass
            elapsed = _t.perf_counter() - t0
            return {"ok": True, "kind": "regressor", "n": n,
                    "mse": round(errs / max(c, 1), 4), "count": c,
                    "elapsed_s": round(elapsed, 3)}

        if has_score and not has_transform:
            scores = []
            for i in range(n):
                try:
                    s = instance.score_one(rows[i])
                    if s is not None:
                        scores.append(float(s))
                    instance.learn_one(rows[i])
                except Exception:
                    pass
            elapsed = _t.perf_counter() - t0
            arr = _np.asarray(scores) if scores else _np.array([0.0])
            return {"ok": True, "kind": "anomaly", "n": n,
                    "mean_score": round(float(arr.mean()), 4),
                    "p95_score": round(float(_np.percentile(arr, 95)), 4),
                    "elapsed_s": round(elapsed, 3)}

        if has_transform:
            for i in range(n):
                try:
                    instance.learn_one(rows[i])
                except Exception:
                    pass
            sample = None
            try:
                s = instance.transform_one(rows[0])
                if s is not None:
                    sample = {k: (float(v) if hasattr(v, "__float__") else str(v))
                              for k, v in list(s.items())[:6]}
            except Exception:
                pass
            elapsed = _t.perf_counter() - t0
            return {"ok": True, "kind": "transformer", "n": n,
                    "sample": sample, "elapsed_s": round(elapsed, 3)}

        return {"ok": False, "error": "no known interface"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

def create_app(config_path: str, enable_playground: bool = False) -> Any:
    """Return the Flask application for one configuration.

    The configuration is loaded once, here. ``GET /status``, ``POST /emergency-stop`` and
    ``GET /decisions.svg`` read the file again at each request, so that the three agree.
    ``POST /emergency-stop`` answers ``403`` when the request carries an ``Origin`` of another
    host. ``POST /api/playground/run`` executes the code of the request, so it answers ``403``
    unless ``enable_playground`` is true (``--enable-playground`` on the command line).
    """
    from flask import Flask, Response, jsonify, render_template

    config = load_config(config_path)
    app = Flask(
        __name__,
        template_folder=str(TEMPLATE_DIR),
        static_folder=str(STATIC_DIR),
    )

    from dashboard.sections import _SIGNAL_CLASSES_EXT
    app.jinja_env.globals["piece_has_page"] = lambda cat, pid: (cat, pid) in _SIGNAL_CLASSES_EXT

    from dashboard import own_data as od
    from dashboard.paths import FIXTURES

    data_dir = config.paths.log.parent

    def base_ctx(**kw: Any) -> dict[str, Any]:
        ctx = {
            "config": config,
            "config_path": config_path,
            "pyver": platform.python_version(),
            "machine": platform.platform(),
        }
        ctx.update(kw)
        return ctx

    # ------------------------------------------------------------ dashboard

    @app.get("/")
    def dashboard() -> Any:
        return render_template(
            "dashboard.html", page="dashboard", title="Dashboard",
            **base_ctx(),
        )

    @app.get("/api/counts")
    def api_counts() -> Any:
        return jsonify(_count_reasons(config.paths.log))

    @app.get("/api/decisions")
    def api_decisions() -> Any:
        from flask import request

        try:
            limit = int(request.args.get("limit", "50"))
        except ValueError:
            limit = 50
        limit = max(1, min(limit, 1000))
        rows = _iter_decisions(config.paths.log)
        return jsonify(rows[-limit:])

    # ------------------------------------------------------------ decisions

    @app.get("/decisions")
    def decisions() -> Any:
        rows = _iter_decisions(config.paths.log)
        return render_template(
            "decisions.html", page="decisions", title="Decisions",
            rows=rows[::-1],
            **base_ctx(),
        )

    # ------------------------------------------------------------ audit table

    @app.get("/audit")
    def audit() -> Any:
        info: dict[str, Any] = {"available": False}
        try:
            from dense_armor.core.finance.audit_db import AuditDB

            url = (
                os.environ.get("DENSE_ARMOR_PG_URL")
                or "postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/dense_armor_test"
            )
            db = AuditDB(url)
            info = {
                "available": True,
                "url": url.split("@")[-1],
                "records": len(db),
                "last_hash": db.last_hash,
                "verified": db.verify(),
            }
        except Exception as e:
            info["error"] = f"{type(e).__name__}: {e}"
        return render_template(
            "audit.html", page="audit", title="Audit table",
            info=info, **base_ctx(),
        )

    # ------------------------------------------------------------ ITCH feed

    @app.get("/feed")
    def feed() -> Any:
        info: dict[str, Any] = {"available": config.paths.feed.exists()}
        if info["available"]:
            types: dict[str, int] = {}
            total = 0
            try:
                for m in iter_itch(config.paths.feed):
                    total += 1
                    types[m["type"]] = types.get(m["type"], 0) + 1
            except Exception as e:
                info["error"] = f"{type(e).__name__}: {e}"
            info["total"] = total
            info["types"] = types
        return render_template(
            "feed.html", page="feed", title="ITCH feed",
            info=info, **base_ctx(),
        )

    @app.get("/synthetic")
    def synthetic() -> Any:
        n = 20
        from flask import request

        try:
            n = max(1, min(int(request.args.get("n", "20")), 1000))
        except ValueError:
            pass
        data = synthetic_itch(n, 0)
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(data)
            tmp = Path(f.name)
        try:
            types: dict[str, int] = {}
            for m in iter_itch(tmp):
                types[m["type"]] = types.get(m["type"], 0) + 1
        finally:
            tmp.unlink()
        return render_template(
            "synthetic.html", page="synthetic", title="Synthetic feed",
            n=n, size=len(data), types=types, **base_ctx(),
        )

    # ------------------------------------------------------------ schedule

    @app.get("/schedule")
    def schedule() -> Any:
        twap = twap_schedule(config.schedule.total, config.schedule.n)
        vwap = vwap_schedule(
            config.schedule.total,
            __import__("numpy").ones((1, config.schedule.n)),
        )
        return render_template(
            "schedule.html", page="schedule", title="Schedule",
            twap=[float(x) for x in twap],
            vwap=[float(x) for x in vwap],
            total=config.schedule.total,
            n=config.schedule.n,
            **base_ctx(),
        )

    # ------------------------------------------------------------ pretrade

    @app.get("/pretrade")
    def pretrade() -> Any:
        checker = config.make_checker()
        decision = checker.check(
            account="AAPL", side="buy", size=5.0,
            price=100.0, ref_price=100.0, t=0.0,
        )
        return render_template(
            "pretrade.html", page="pretrade", title="Pre-trade",
            decision=decision, checker=checker, **base_ctx(),
        )

    # ------------------------------------------------------------ ring

    @app.get("/ring")
    def ring() -> Any:
        import numpy as np

        rb = config.make_ring()
        pushed = 0
        popped = 0
        out = np.zeros(config.ring.width)
        for i in range(config.ring.capacity + 2):
            if rb.push(np.arange(config.ring.width) + i):
                pushed += 1
        while rb.pop_into(out):
            popped += 1
        return render_template(
            "ring.html", page="ring", title="Ring buffer",
            capacity=config.ring.capacity, width=config.ring.width,
            pushed=pushed, popped=popped, **base_ctx(),
        )

    # ------------------------------------------------------------ decode

    @app.get("/decode")
    def decode() -> Any:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(synthetic_itch(3, 0))
            tmp = Path(f.name)
        try:
            messages = list(iter_itch(tmp))
        finally:
            tmp.unlink()
        return render_template(
            "decode.html", page="decode", title="Decode",
            messages=messages, **base_ctx(),
        )

    # ------------------------------------------------------------ run demo

    @app.get("/run")
    def run() -> Any:
        result = collect_status(config_path)
        return render_template(
            "run.html", page="run", title="Run demo",
            result=result, **base_ctx(),
        )

    # ------------------------------------------------------------ FIX

    @app.get("/fix")
    def fix() -> Any:
        info: dict[str, Any] = {"available": False}
        try:
            import quickfix  # noqa: F401

            info["available"] = True
            info["version"] = __import__("importlib.metadata", fromlist=["version"]).version("quickfix")
        except Exception as e:
            info["error"] = f"{type(e).__name__}: {e}"
        return render_template(
            "fix.html", page="fix", title="FIX gateway",
            info=info, **base_ctx(),
        )

    # ------------------------------------------------------------ emergency

    @app.get("/ping")
    def ping() -> Any:
        import time as _time

        app.config["LAST_PING"] = _time.time()
        return jsonify({"ok": True})

    @app.get("/status")
    def status_json() -> Any:
        return jsonify(collect_status(config_path))

    # ------------------------------------------------------ learners
    def _run_demo() -> dict[str, Any]:
        import tempfile
        import time

        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            feed_path = tmp / "feed.bin"
            log_path = tmp / "log.jsonl"
            feed_path.write_bytes(synthetic_itch(1000, 0))
            checker = config.make_checker()
            dlog = DecisionLog(log_path)
            records: list[tuple[float, dict]] = []
            ref_price: dict[str, float] = {}
            counts: dict[str, int] = {}
            n_messages = 0
            n_orders = 0
            for m in iter_itch(feed_path):
                n_messages += 1
                if m["type"] not in ("A", "F"):
                    continue
                n_orders += 1
                stock = m["stock"]
                if stock not in ref_price:
                    ref_price[stock] = m["price"]
                t_order = m["timestamp"] / 1e9
                dec = checker.check(
                    account=stock,
                    side="buy" if m["side"] == "B" else "sell",
                    size=m["shares"], price=m["price"],
                    ref_price=ref_price[stock], t=t_order,
                )
                counts[dec.reason] = counts.get(dec.reason, 0) + 1
                records.append((t_order, {
                    "approved": bool(dec.approved),
                    "order_ref": m["order_ref"],
                    "reason": dec.reason,
                    "stock": stock,
                }))
            dlog.append_many(records)
            # the log is checked here, while its file still exists: a missing file counts as a valid chain
            log_info = {"records": len(dlog), "verified": dlog.verify(), "last_hash": dlog.last_hash}
        elapsed = time.perf_counter() - t0
        twap = twap_schedule(config.schedule.total, config.schedule.n)
        return {
            "ok": True,
            "n_messages": n_messages,
            "n_orders": n_orders,
            "counts": counts,
            "log": log_info,
            "schedule": {"total": config.schedule.total,
                         "n": config.schedule.n,
                         "sum": round(float(twap.sum()), 4)},
            "elapsed_s": round(elapsed, 3),
            "synthetic": True,
        }

    @app.post("/api/demo/run")
    def api_demo_run() -> Any:
        """Run the demo on a synthetic feed (about 7 seconds) and keep the answer."""
        app.config["LAST_DEMO"] = _run_demo()
        return jsonify(app.config["LAST_DEMO"])

    @app.get("/api/demo/summary")
    def api_demo_summary() -> Any:
        """The answer of the last ``POST /api/demo/run``; a GET never runs the demo."""
        last = app.config.get("LAST_DEMO")
        if last is None:
            return jsonify({"ok": False, "error": "the demo has not been run: POST /api/demo/run"}), 404
        return jsonify(last)

    # ------------------------------------------------------ the visitor's own files

    def _json_error(message: str, code: int = 400) -> Any:
        return jsonify({"ok": False, "error": message}), code

    def _csv_response(text: str, filename: str) -> Any:
        return Response(
            text, mimetype="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    def _sent_csv() -> tuple[str, str]:
        """The text and the name of the CSV of the request: the form field ``file``, or JSON ``{"csv": ...}``."""
        from flask import request

        f = request.files.get("file")
        if f is not None:
            return od.decode(f.read(od.MAX_BYTES + 1)), f.filename or "file.csv"
        body = request.get_json(silent=True) or {}
        if isinstance(body.get("csv"), str):
            return od.decode(body["csv"].encode("utf-8")), str(body.get("name") or "pasted.csv")
        raise od.DataError("send a CSV file in the form field 'file', or JSON {\"csv\": \"...\"}")

    @app.post("/api/data/upload")
    def api_data_upload() -> Any:
        """Check and keep a CSV of prices; the answer says what was found in it."""
        from flask import request

        f = request.files.get("file")
        if f is None:
            return _json_error("send the CSV of prices in the form field 'file'")
        try:
            meta = od.store_prices(data_dir, f.filename or "prices.csv", f.read(od.MAX_BYTES + 1))
        except od.DataError as e:
            return _json_error(str(e))
        return jsonify({"ok": True, **meta})

    @app.get("/api/data/uploads")
    def api_data_uploads() -> Any:
        return jsonify(od.list_uploads(data_dir))

    @app.get("/api/report/<name>")
    def api_report(name: str) -> Any:
        """The result of a chain run, as ``<run id>.csv`` or ``<run id>.json``."""
        rid, _, ext = name.rpartition(".")
        try:
            result = od.load_report(data_dir, rid)
        except od.DataError as e:
            return _json_error(str(e), 404)
        if ext == "csv":
            return _csv_response(od.report_csv(result), f"dense-armor-run-{rid}.csv")
        if ext == "json":
            return Response(json.dumps(result), mimetype="application/json")
        return _json_error("the report is a .csv or a .json", 404)

    @app.get("/sample/prices.csv")
    def sample_prices() -> Any:
        return _csv_response(od.sample_prices_csv(FIXTURES), "sample-prices.csv")

    @app.get("/sample/orders.csv")
    def sample_orders() -> Any:
        return _csv_response(od.sample_orders_csv(), "sample-orders.csv")

    @app.get("/sample/volumes.csv")
    def sample_volumes() -> Any:
        return _csv_response(od.sample_volumes_csv(), "sample-volumes.csv")

    @app.post("/api/pretrade/check")
    def api_pretrade_check() -> Any:
        """Send every order of a CSV to the pre-trade checker, with the limits of the form or of the config."""
        from dataclasses import asdict

        from flask import request

        try:
            text, name = _sent_csv()
            given = {k: request.form.get(k) for k in od.LIMIT_KEYS}
            body = request.get_json(silent=True) or {}
            if isinstance(body.get("limits"), dict):
                given.update(body["limits"])
            limits = od.read_limits(asdict(config.pretrade), given)
            orders = od.parse_orders(text)
            rid = od.new_run_id()
            reports = data_dir / "reports"
            reports.mkdir(parents=True, exist_ok=True)
            out = od.check_orders(orders, limits, reports / f"{rid}.log.jsonl")
        except od.DataError as e:
            return _json_error(str(e))
        (reports / f"{rid}.orders.csv").write_text(od.orders_csv(out["results"]), encoding="utf-8")
        shown = out["results"][:500]
        return jsonify({
            "ok": True, "run_id": rid, "name": name, "limits": limits,
            "n": out["n"], "approved": out["approved"], "counts": out["counts"], "log": out["log"],
            "results": shown, "truncated": len(out["results"]) > len(shown),
        })

    @app.get("/api/pretrade/<name>")
    def api_pretrade_download(name: str) -> Any:
        """The checked orders (``<run id>.csv``) or their hash-chained log (``<run id>.jsonl``)."""
        rid, _, ext = name.rpartition(".")
        suffix = {"csv": "orders.csv", "jsonl": "log.jsonl"}.get(ext)
        if suffix is None:
            return _json_error("the download is a .csv or a .jsonl", 404)
        try:
            path = od.report_path(data_dir, rid, suffix)
        except od.DataError as e:
            return _json_error(str(e), 404)
        text = path.read_text(encoding="utf-8")
        if ext == "csv":
            return _csv_response(text, f"dense-armor-orders-{rid}.csv")
        return Response(text, mimetype="application/x-ndjson", headers={
            "Content-Disposition": f'attachment; filename="dense-armor-log-{rid}.jsonl"'})

    @app.post("/api/schedule")
    def api_schedule() -> Any:
        """The TWAP and VWAP slices of an order; ``history`` is the volume of past sessions, as text or lists."""
        from flask import request

        body = request.get_json(silent=True) or {}
        try:
            history = body.get("history")
            if isinstance(history, str) and history.strip():
                history = od.parse_history(history)
            elif not history:
                history = None
            plan = od.schedule_plan(body.get("total", config.schedule.total),
                                    body.get("n", config.schedule.n), history)
        except od.DataError as e:
            return _json_error(str(e))
        return jsonify({"ok": True, **plan})

    @app.post("/api/chain/run")
    def api_chain_run() -> Any:
        from flask import request

        from dashboard.chain import run_chain

        body = request.get_json(silent=True) or {}
        picks = {s: body.get(s, "none") for s in
                 ("data", "features", "signal", "portfolio", "risk", "execution")}
        if not picks["data"] or picks["data"] == "none":
            picks["data"] = "gspc"
        if not picks["signal"] or picks["signal"] == "none":
            picks["signal"] = "hoeffding"
        n = max(50, min(int(body.get("n", 1253)), 5000))
        seed = int(body.get("seed", 0))
        try:
            out = run_chain(picks, n=n, seed=seed, folder=data_dir,
                            column=body.get("column") or None)
        except Exception as e:
            return jsonify({"ok": False,
                            "error": f"{type(e).__name__}: {e}"}), 500
        if out.get("ok"):
            od.save_report(data_dir, out)
        return jsonify(out)

        body = request.get_json(silent=True) or {}
        learner_id = body.get("learner_id")
        anomaly_id = body.get("anomaly_id")
        mixture_id = body.get("mixture_id")
        ts_id = body.get("ts_id")
        drift_id = body.get("drift_id")
        preprocess_id = body.get("preprocess_id")
        metric_id = body.get("metric_id")
        stream_id = body.get("stream_id")

        if learner_id is None:
            legacy = str(body.get("learner", "HoeffdingTreeClassifier"))
            learner_id = {
                "HoeffdingTreeClassifier": "hoeffding",
                "HoeffdingAdaptiveTreeClassifier": "adaptive",
                "HoeffdingAnytimeTreeClassifier": "anytime",
                "MondrianForestClassifier": "mondrian",
                "AggregatedMondrianForestClassifier": "amf",
                "SGTClassifier": "sgt",
            }.get(legacy, "hoeffding")
        if anomaly_id is None:
            anomaly_id = "lof" if body.get("use_anomaly") else "none"
        if mixture_id is None:
            mixture_id = "gmm" if body.get("use_mixture") else "none"
        if ts_id is None:
            ts_id = "garch" if body.get("use_garch") else "none"
        if drift_id is None:
            drift_id = "none"
        if preprocess_id is None:
            preprocess_id = "none"
        if metric_id is None:
            metric_id = "accuracy"
        if stream_id is None:
            legacy_stream = str(body.get("stream", "finance_gspc"))
            stream_id = {
                "sudden": "driftstream_sudden",
                "gradual": "driftstream_gradual",
                "incremental": "driftstream_incremental",
            }.get(legacy_stream, legacy_stream)
            if stream_id not in CATALOG["stream"]:
                stream_id = "finance_gspc"

        n = int(body.get("n", 500))
        seed = int(body.get("seed", 0))
        n = max(50, min(n, 5000))

        try:
            out = run_chain(learner_id, anomaly_id, mixture_id, ts_id,
                            drift_id, stream_id, preprocess_id, metric_id,
                            n, seed)
        except Exception as e:
            return jsonify({"ok": False,
                            "error": f"{type(e).__name__}: {e}"}), 500
        return jsonify(out)

        try:
            body = request.get_json(silent=True) or {}
            learner_name = str(body.get("learner", "HoeffdingTreeClassifier"))
            stream_kind = str(body.get("stream", "sudden"))
            n = int(body.get("n", 1000))
            seed = int(body.get("seed", 0))
            use_anomaly = bool(body.get("use_anomaly", False))
            use_mixture = bool(body.get("use_mixture", False))
            use_garch = bool(body.get("use_garch", False))
        except (TypeError, ValueError) as e:
            return jsonify({"ok": False, "error": f"bad input: {e}"}), 400
        if stream_kind not in ("sudden", "gradual", "incremental"):
            return jsonify({"ok": False, "error": "bad stream"}), 400
        n = max(100, min(n, 5000))

        from dense_armor.core.online.forest.amf import (
            AggregatedMondrianForestClassifier,
        )
        from dense_armor.core.online.tree.adaptive import (
            HoeffdingAdaptiveTreeClassifier,
        )
        from dense_armor.core.online.tree.hoeffding import HoeffdingTreeClassifier
        from dense_armor.core.online.tree.mondrian import MondrianForestClassifier
        from dense_armor.core.signals.datasets import DriftStream

        catalog = {
            "HoeffdingTreeClassifier": lambda: HoeffdingTreeClassifier(),
            "HoeffdingAdaptiveTreeClassifier": lambda: HoeffdingAdaptiveTreeClassifier(),
            "MondrianForestClassifier": lambda: MondrianForestClassifier(seed=seed),
            "AggregatedMondrianForestClassifier": lambda: AggregatedMondrianForestClassifier(seed=seed),
        }
        if learner_name not in catalog:
            return jsonify({"ok": False, "error": f"unknown learner {learner_name}"}), 400

        anomaly = None
        mixture = None
        garch = None
        try:
            if use_anomaly:
                from dense_armor.core.online.anomaly.lof import LocalOutlierFactor
                anomaly = LocalOutlierFactor()
            if use_mixture:
                from dense_armor.core.signals.stats import OnlineGaussianMixture
                mixture = OnlineGaussianMixture(n_components=2)
            if use_garch:
                from dense_armor.core.online.timeseries.garch import GARCH11
                garch = GARCH11()
        except Exception as e:
            return jsonify({"ok": False, "error": f"chain build: {e}"}), 500

        ds = DriftStream(n_samples=n, n_features=4, n_drifts=1,
                         kind=stream_kind, seed=seed)
        X, y = ds.data()
        names = [f"x{i}" for i in range(X.shape[1])]
        est = catalog[learner_name]()

        correct = np.zeros(n, dtype=np.int8)
        anomaly_scores = np.zeros(n, dtype=np.float64)
        correct_series = []
        anomaly_marks = []
        mixture_scores = []
        garch_vars = []
        anomalies = 0
        mixture_score = None
        garch_var = None
        t0 = time.perf_counter()
        for i in range(n):
            x = {k: float(X[i, j]) for j, k in enumerate(names)}
            p = est.predict_one(x)
            est.learn_one(x, int(y[i]))
            if p is not None and int(p) == int(y[i]):
                correct[i] = 1
            if anomaly is not None:
                s = anomaly.score_one(x)
                anomaly.learn_one(x)
                anomaly_scores[i] = float(s)
                if s > 1.5:
                    anomalies += 1
                    anomaly_marks.append(int(i))
            if mixture is not None:
                mixture.learn_one(x)
                try:
                    mixture_score = float(mixture.score_one(x))
                    if i % max(1, n // 120) == 0:
                        mixture_scores.append({"i": int(i), "v": round(mixture_score, 4)})
                except Exception:
                    pass
            if garch is not None and i > 0:
                dx = float(X[i, 0]) - float(X[i - 1, 0])
                garch.learn_one({"x": dx})
                try:
                    # GARCH11 espone la varianza condizionale in un attributo
                    v = None
                    for attr in ("variance_", "variance", "sigma2_", "current_variance"):
                        if hasattr(garch, attr):
                            val = getattr(garch, attr)
                            v = float(val() if callable(val) else val)
                            break
                    if v is not None:
                        garch_var = v
                        if i % max(1, n // 120) == 0:
                            garch_vars.append({"i": int(i), "v": round(v, 6)})
                except Exception:
                    pass
            if i % max(1, n // 120) == 0:
                correct_series.append({"i": int(i), "cum": round(float(correct[:i+1].mean()), 4)})
        elapsed = time.perf_counter() - t0

        return jsonify({
            "ok": True,
            "learner": learner_name,
            "stream": stream_kind,
            "n": n,
            "seed": seed,
            "accuracy": round(float(correct.mean()), 4),
            "anomalies": anomalies,
            "anomaly_marks": anomaly_marks[:200],
            "mixture_score": round(mixture_score, 4) if mixture_score is not None else None,
            "garch_var": round(garch_var, 8) if garch_var is not None else None,
            "series": {
                "accuracy": correct_series,
                "anomaly_scores": [{"i": int(i), "v": round(float(anomaly_scores[i]), 3)}
                                   for i in range(0, n, max(1, n // 200))]
                                if use_anomaly else [],
                "mixture": mixture_scores,
                "garch": garch_vars,
            },
            "chain": {"stream": stream_kind, "learner": learner_name,
                      "anomaly": use_anomaly, "mixture": use_mixture, "garch": use_garch},
            "elapsed_s": round(elapsed, 3),
        })


    @app.get("/api/chain/catalog")
    def api_chain_catalog() -> Any:
        from dashboard.chain import CATALOG
        out = {}
        for slot, pieces in CATALOG.items():
            out[slot] = [{"id": k, **(v if isinstance(v, dict) else {"class": v})}
                          for k, v in pieces.items()]
        return jsonify(out)

        try:
            body = request.get_json(silent=True) or {}
            entry_id = str(body.get("id", ""))
            overrides = dict(body.get("params", {}) or {})
            n = int(body.get("n", 500))
            seed = int(body.get("seed", 0))
        except (TypeError, ValueError) as e:
            return jsonify({"ok": False, "error": f"bad input: {e}"}), 400
        entry = BY_ID.get(entry_id)
        if not entry:
            return jsonify({"ok": False, "error": f"unknown id {entry_id!r}"}), 404
        n = max(50, min(n, 5000))
        params = dict(entry["default_params"])
        params.update(overrides)
        result = _run_universal(entry["class_path"], params, n, seed)
        result["id"] = entry_id
        result["subsystem"] = entry["subsystem"]
        return jsonify(result), (200 if result.get("ok") else 500)




    @app.get("/s/data")
    def s_data() -> Any:
        from dashboard.sections import data_page
        return data_page(render_template, base_ctx())

    @app.get("/s/features")
    def s_features() -> Any:
        from flask import request

        from dashboard.sections import features_page
        cat = request.args.get("cat", "scaling")
        return features_page(cat, render_template, base_ctx())

    @app.get("/s/signal")
    def s_signal() -> Any:
        from flask import request

        from dashboard.sections import signal_page
        cat = request.args.get("cat", "trees")
        return signal_page(cat, render_template, base_ctx())

    @app.get("/s/portfolio")
    def s_portfolio() -> Any:
        from dashboard.sections import portfolio_page
        return portfolio_page(render_template, base_ctx())

    @app.get("/s/risk")
    def s_risk() -> Any:
        from flask import request

        from dashboard.sections import risk_page
        cat = request.args.get("cat", "anomaly")
        return risk_page(cat, render_template, base_ctx())

    @app.get("/s/execution")
    def s_execution() -> Any:
        from dashboard.sections import execution_page
        return execution_page(render_template, base_ctx())

    @app.get("/s/more")
    def s_more() -> Any:
        from dashboard.sections import stub_page
        return stub_page("More", 0, ["Reserved"],
                          render_template, base_ctx())

    @app.get("/piece/<section>/<category>/<piece_id>")
    def piece_page(section: str, category: str, piece_id: str) -> Any:
        from dashboard.sections import piece_meta
        meta = piece_meta(section, category, piece_id)
        if not meta.get("ok"):
            return Response(f"unknown piece: {meta.get('error')}", status=404)
        return render_template(
            "piece.html", page="piece", title=meta["name"],
            section=section, category=category, piece_id=piece_id,
            meta=meta, **base_ctx(),
        )

    @app.post("/api/piece/<section>/<category>/<piece_id>/run")
    def api_piece_run(section: str, category: str, piece_id: str) -> Any:
        from flask import request

        from dashboard.sections import piece_detail
        try:
            body = request.get_json(silent=True) or {}
            params = dict(body.get("params", {}) or {})
        except (TypeError, ValueError) as e:
            return jsonify({"ok": False, "error": f"bad input: {e}"}), 400
        out = piece_detail(section, category, piece_id, params)
        return jsonify(out), (200 if out.get("ok") else 400)


    @app.get("/api/pieces/search")
    def api_pieces_search() -> Any:
        from flask import request

        from dashboard.sections import _SIGNAL_CLASSES_EXT
        q = (request.args.get("q") or "").lower().strip()
        out = []
        for (cat, pid), (mod_name, cls_name) in _SIGNAL_CLASSES_EXT.items():
            hay = f"{cat} {pid} {cls_name}".lower()
            if q and q not in hay:
                continue
            out.append({
                "section": "signal",  # verra' sovrascritto sotto
                "category": cat,
                "id": pid,
                "name": cls_name,
            })
        # mappa categoria -> sezione
        SECTION_MAP = {
            "trees": "signal", "linear": "signal", "ensembles": "signal",
            "facto": "signal", "multi": "signal",
            "anomaly": "risk", "drift": "risk",
            "scaling": "features", "moments": "features",
            "sketches": "features", "dependence": "features",
            "selection": "features", "distribution": "features",
        }
        for entry in out:
            entry["section"] = SECTION_MAP.get(entry["category"], "signal")
        out.sort(key=lambda e: (e["section"], e["category"], e["id"]))
        return jsonify(out[:200])

    @app.get("/chain")
    def chain_page() -> Any:
        return render_template(
            "chain.html", page="chain", title="Chain", **base_ctx(),
        )

    @app.get("/playground")
    def playground() -> Any:
        return render_template(
            "playground.html", page="playground", title="Playground", **base_ctx(),
        )

    @app.post("/api/playground/run")
    def api_playground_run() -> Any:
        import subprocess as _sp
        import time

        from flask import request

        if not enable_playground:
            return jsonify({
                "ok": False,
                "error": "playground disabled: it executes the code that is posted; "
                         "start the dashboard with --enable-playground on the loopback",
            }), 403

        try:
            body = request.get_json(silent=True) or {}
            code = str(body.get("code", ""))
            params = body.get("params", {}) or {}
        except Exception as e:
            return jsonify({"ok": False, "error": f"bad request: {e}"}), 400
        if not code.strip():
            return jsonify({"ok": False, "error": "empty code"}), 400
        if len(code) > 20000:
            return jsonify({"ok": False, "error": "code too long"}), 400

        try:
            n_val = int(params.get("n", 1000))
            seed_val = int(params.get("seed", 0))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "bad n or seed"}), 400
        n_val = max(100, min(n_val, 5000))
        seed_val = max(0, seed_val)
        stream_kind = str(params.get("stream", "sudden"))
        if stream_kind not in ("sudden", "gradual", "incremental"):
            return jsonify({"ok": False, "error": "bad stream"}), 400
        learner_name = str(params.get("learner", "HoeffdingTreeClassifier"))
        use_anomaly = bool(params.get("use_anomaly", False))
        use_mixture = bool(params.get("use_mixture", False))
        use_garch = bool(params.get("use_garch", False))

        preamble = (
            "import sys\n"
            "STREAM_KIND = " + repr(stream_kind) + "\n"
            "LEARNER_NAME = " + repr(learner_name) + "\n"
            "N = " + str(n_val) + "\n"
            "SEED = " + str(seed_val) + "\n"
            "USE_ANOMALY = " + repr(use_anomaly) + "\n"
            "USE_MIXTURE = " + repr(use_mixture) + "\n"
            "USE_GARCH = " + repr(use_garch) + "\n"
            "from dense_armor.core.online.drift.adwin import ADWIN\n"
            "from dense_armor.core.online.tree.hoeffding import HoeffdingTreeClassifier\n"
            "from dense_armor.core.online.tree.adaptive import HoeffdingAdaptiveTreeClassifier\n"
            "from dense_armor.core.online.tree.mondrian import MondrianForestClassifier\n"
            "from dense_armor.core.online.forest.amf import AggregatedMondrianForestClassifier\n"
            "LEARNERS = {\n"
            "    'HoeffdingTreeClassifier': lambda: HoeffdingTreeClassifier(),\n"
            "    'HoeffdingAdaptiveTreeClassifier': lambda: HoeffdingAdaptiveTreeClassifier(drift_detector=ADWIN()),\n"
            "    'MondrianForestClassifier': lambda: MondrianForestClassifier(seed=SEED),\n"
            "    'AggregatedMondrianForestClassifier': lambda: AggregatedMondrianForestClassifier(seed=SEED),\n"
            "}\n"
            "if LEARNER_NAME not in LEARNERS:\n"
            "    raise SystemExit('unknown learner: ' + LEARNER_NAME)\n"
            "ANOMALY = None\n"
            "MIXTURE = None\n"
            "GARCH = None\n"
            "if USE_ANOMALY:\n"
            "    from dense_armor.core.online.anomaly.lof import LocalOutlierFactor\n"
            "    ANOMALY = LocalOutlierFactor()\n"
            "if USE_MIXTURE:\n"
            "    from dense_armor.core.signals.stats import OnlineGaussianMixture\n"
            "    MIXTURE = OnlineGaussianMixture(n_components=2)\n"
            "if USE_GARCH:\n"
            "    from dense_armor.core.online.timeseries.garch import GARCH11\n"
            "    GARCH = GARCH11()\n"
            "\n"
        )
        full_code = preamble + code

        env = _playground_env(str(Path(__file__).resolve().parents[3]))
        t0 = time.perf_counter()
        try:
            r = _sp.run(
                [sys.executable, "-c", full_code],
                capture_output=True, text=True,
                timeout=20, env=env,
                cwd=str(Path(__file__).resolve().parents[3]),
            )
            elapsed = time.perf_counter() - t0
            out = r.stdout or ""
            err = r.stderr or ""
            chart = None
            text = out
            # se l'ultima riga di stdout e' {"__chart__": {...}}, estrai
            lines = out.rstrip().splitlines()
            if lines:
                import json as _json
                try:
                    last = _json.loads(lines[-1])
                    if isinstance(last, dict) and "__chart__" in last:
                        chart = last["__chart__"]
                        text = "\n".join(lines[:-1])
                except Exception:
                    pass
            return jsonify({
                "ok": r.returncode == 0,
                "returncode": r.returncode,
                "elapsed_s": round(elapsed, 3),
                "stdout": text,
                "stderr": err,
                "chart": chart,
            })
        except _sp.TimeoutExpired:
            return jsonify({
                "ok": False, "returncode": -1,
                "elapsed_s": round(time.perf_counter() - t0, 3),
                "stdout": "", "stderr": "timeout after 20 s", "chart": None,
            }), 200

    @app.get("/learners")
    def learners() -> Any:
        return render_template(
            "learners.html", page="learners", title="Learners", **base_ctx(),
        )

    @app.post("/api/learners/run")
    def api_learners_run() -> Any:
        import time

        import numpy as np
        from flask import request

        try:
            body = request.get_json(silent=True) or {}
            learner_name = str(body.get("learner", "HoeffdingTreeClassifier"))
            stream_kind = str(body.get("stream", "sudden"))
            n = int(body.get("n", 1000))
            seed = int(body.get("seed", 0))
            window = int(body.get("window", 100))
        except (TypeError, ValueError) as e:
            return jsonify({"ok": False, "error": f"bad input: {e}"}), 400

        if stream_kind not in ("sudden", "gradual", "incremental"):
            return jsonify({"ok": False, "error": "bad stream"}), 400
        n = max(100, min(n, 5000))
        window = max(10, min(window, 500))

        try:
            from dense_armor.core.online.drift.adwin import ADWIN
            from dense_armor.core.online.forest.amf import (
                AggregatedMondrianForestClassifier,
            )
            from dense_armor.core.online.tree.adaptive import (
                HoeffdingAdaptiveTreeClassifier,
            )
            from dense_armor.core.online.tree.hoeffding import (
                HoeffdingTreeClassifier,
            )
            from dense_armor.core.online.tree.mondrian import (
                MondrianForestClassifier,
            )
            from dense_armor.core.signals.datasets import DriftStream

            catalog = {
                "HoeffdingTreeClassifier": lambda: HoeffdingTreeClassifier(),
                "HoeffdingAdaptiveTreeClassifier": lambda: HoeffdingAdaptiveTreeClassifier(
                    drift_detector=ADWIN()
                ),
                "MondrianForestClassifier": lambda: MondrianForestClassifier(seed=seed),
                "AggregatedMondrianForestClassifier": lambda: AggregatedMondrianForestClassifier(seed=seed),
            }
            if learner_name not in catalog:
                return jsonify({"ok": False, "error": f"unknown learner {learner_name}"}), 400

            ds = DriftStream(
                n_samples=n, n_features=4, n_drifts=1,
                kind=stream_kind, seed=seed,
            )
            X, y = ds.data()
            drift_idx = [int(p * n) for p in
                         (getattr(ds, "drift_positions", []) or [])]
            names = [f"x{i}" for i in range(X.shape[1])]

            est = catalog[learner_name]()
            correct = np.zeros(n, dtype=np.int8)
            latency_us = np.empty(n, dtype=np.float64)
            t0 = time.perf_counter()
            for i in range(n):
                x = {k: float(X[i, j]) for j, k in enumerate(names)}
                yi = int(y[i])
                tp = time.perf_counter()
                p_pred = est.predict_one(x)
                est.learn_one(x, yi)
                latency_us[i] = (time.perf_counter() - tp) * 1e6
                if p_pred is not None and int(p_pred) == yi:
                    correct[i] = 1
            elapsed = time.perf_counter() - t0

            # accuracy cumulativa e a finestra, campionate ogni window
            cum = np.cumsum(correct) / np.arange(1, n + 1)
            rolling = np.convolve(correct, np.ones(window, dtype=float) / window, mode="valid")
            step = max(1, n // 120)
            points = [{"i": int(i), "cum": round(float(cum[i]), 4),
                       "win": round(float(rolling[min(i, len(rolling) - 1)]), 4)}
                      for i in range(0, n, step)]

            # accuracy per fetta tra drift consecutivi
            edges = [0] + drift_idx + [n]
            slices = []
            for a, b in zip(edges[:-1], edges[1:]):
                if b > a:
                    slices.append({
                        "from": a, "to": b,
                        "acc": round(float(correct[a:b].mean()), 4),
                    })

            return jsonify({
                "ok": True,
                "learner": learner_name,
                "stream": stream_kind,
                "n": n,
                "seed": seed,
                "window": window,
                "accuracy_final": round(float(cum[-1]), 4),
                "accuracy_window": round(float(rolling[-1]), 4),
                "elapsed_s": round(elapsed, 3),
                "p50_us": round(float(np.percentile(latency_us, 50)), 1),
                "p99_us": round(float(np.percentile(latency_us, 99)), 1),
                "drift_positions": drift_idx,
                "slices": slices,
                "points": points,
            })
        except Exception as e:
            return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500

    @app.get("/api/last-run/decisions")
    def api_last_run_decisions() -> Any:
        from flask import request

        last_run_log = config.paths.log.parent / "log_last_run.jsonl"
        try:
            limit = int(request.args.get("limit", "20"))
        except ValueError:
            limit = 20
        limit = max(1, min(limit, 1000))
        rows = _iter_decisions(last_run_log)
        return jsonify(rows[-limit:])

    @app.post("/api/run")
    def api_run() -> Any:
        import tempfile
        import time

        from flask import request

        try:
            body = request.get_json(silent=True) or {}
            n = int(body.get("n", 1000))
            seed = int(body.get("seed", 0))
        except (TypeError, ValueError) as e:
            return jsonify({"ok": False, "error": f"bad n or seed: {e}"}), 400
        n = max(1, min(n, 10000))
        seed = max(0, seed)

        last_run_log = config.paths.log.parent / "log_last_run.jsonl"
        last_run_log.unlink(missing_ok=True)

        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            feed_path = tmp / "feed.bin"
            feed_path.write_bytes(synthetic_itch(n, seed))
            checker = config.make_checker()
            dlog = DecisionLog(last_run_log)
            records: list[tuple[float, dict]] = []
            ref_price: dict[str, float] = {}
            counts: dict[str, int] = {}
            n_messages = 0
            n_orders = 0
            for m in iter_itch(feed_path):
                n_messages += 1
                if m["type"] not in ("A", "F"):
                    continue
                n_orders += 1
                stock = m["stock"]
                if stock not in ref_price:
                    ref_price[stock] = m["price"]
                t_order = m["timestamp"] / 1e9
                decision = checker.check(
                    account=stock,
                    side="buy" if m["side"] == "B" else "sell",
                    size=m["shares"],
                    price=m["price"],
                    ref_price=ref_price[stock],
                    t=t_order,
                )
                counts[decision.reason] = counts.get(decision.reason, 0) + 1
                records.append((t_order, {
                    "approved": bool(decision.approved),
                    "order_ref": m["order_ref"],
                    "reason": decision.reason,
                    "stock": stock,
                }))
            dlog.append_many(records)
        elapsed = time.perf_counter() - t0
        return jsonify({
            "ok": True,
            "n": n,
            "seed": seed,
            "n_messages": n_messages,
            "n_orders": n_orders,
            "counts": counts,
            "elapsed_s": round(elapsed, 3),
            "log": {
                "records": len(dlog),
                "last_hash": dlog.last_hash,
                "verified": dlog.verify(),
                "path": str(last_run_log),
            },
        })

    @app.get("/emergency")
    def emergency() -> Any:
        return render_template(
            "emergency_page.html", page="emergency", title="Emergency stop",
            **base_ctx(),
        )

    @app.post("/emergency-stop")
    def emergency_stop() -> Any:
        from flask import request

        origin = request.headers.get("Origin")
        if origin and urlparse(origin).netloc != request.host:
            return jsonify({"error": "cross-origin request refused"}), 403
        stop = load_config(config_path).paths.emergency_stop
        stop.parent.mkdir(parents=True, exist_ok=True)
        stop.write_bytes(b"")
        return jsonify({"emergency_stop": True})

    @app.get("/emergency-stop")
    def emergency_stop_get() -> Any:
        return Response("GET not allowed; use POST", status=405)

    # ------------------------------------------------------------ SVG

    @app.get("/decisions.svg")
    def decisions_svg() -> Any:
        rows = sorted(_count_reasons(load_config(config_path).paths.log).items())
        if not rows:
            return Response(
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 100"></svg>',
                mimetype="image/svg+xml",
            )
        n = len(rows)
        w, h = 400, 200
        bar_w = max(4, (w - 40) // max(n, 1) - 10)
        max_count = max(c for _, c in rows) or 1
        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
        for i, (reason, count) in enumerate(rows):
            bh = int((count / max_count) * (h - 60))
            x = 20 + i * (bar_w + 10)
            y = h - 20 - bh
            reason = html.escape(reason, quote=True)
            parts.append(
                f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bh}" '
                f'data-reason="{reason}" data-count="{count}"/>'
            )
            parts.append(
                f'<text x="{x + bar_w // 2}" y="{y - 5}" text-anchor="middle" '
                f'font-size="10">{count}</text>'
            )
            parts.append(
                f'<text x="{x + bar_w // 2}" y="{h - 5}" text-anchor="middle" '
                f'font-size="10">{reason}</text>'
            )
        parts.append("</svg>")
        return Response("".join(parts), mimetype="image/svg+xml")

    return app


def main(argv: list[str] | None = None) -> int:
    """Run the dashboard."""
    parser = argparse.ArgumentParser(
        prog="python -m dashboard.app"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--enable-playground", action="store_true",
        help="allow POST /api/playground/run, which executes the code that is posted (loopback only)",
    )
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return int(e.code) if e.code is not None else 2
    if args.enable_playground and args.host not in LOOPBACK_HOSTS:
        sys.stdout.write(json.dumps({
            "error": "--enable-playground needs --host 127.0.0.1 (or localhost, ::1)", "ok": False,
        }, sort_keys=True) + "\n")
        return 2
    try:
        app = create_app(args.config, enable_playground=args.enable_playground)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        sys.stdout.write(json.dumps({"error": str(e), "ok": False}, sort_keys=True) + "\n")
        return 2
    app.run(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
