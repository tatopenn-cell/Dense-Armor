"""Files that a visitor brings to the dashboard, and the reports the dashboard gives back.

Three jobs, each with its own reader:

* ``parse_prices``: a CSV of prices, one column per asset, for the chain and the portfolio;
* ``parse_orders``: a CSV of orders, for the pre-trade checks;
* ``report_csv``: the result of a run as a long-format CSV (``section,name,index,value``), which a
  spreadsheet opens without a macro.

Every reader raises ``DataError`` with a message that names the row or the column, because the
person who reads it has only the file. The module knows nothing of Flask.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

MAX_BYTES = 5_000_000
MAX_ROWS = 200_000
MAX_ORDERS = 50_000
MIN_ROWS = 30

DATE_NAMES = ("date", "time", "timestamp", "datetime")
OHLCV = ("open", "high", "low", "close", "adj close", "volume")
ACCOUNT_NAMES = ("account", "stock", "symbol", "ticker", "asset")
SIDE_NAMES = ("side", "direction")
SIZE_NAMES = ("size", "shares", "quantity", "qty", "volume")
PRICE_NAMES = ("price", "limit", "limit_price")
REF_NAMES = ("ref_price", "reference", "ref", "mid")
TIME_NAMES = ("t", "time", "timestamp", "datetime", "date")
BUY = ("buy", "b", "bid", "long")
SELL = ("sell", "s", "ask", "short")

_ID = re.compile(r"^up_[0-9a-f]{10}$")
_RUN_ID = re.compile(r"^[0-9a-f]{12}$")


class DataError(ValueError):
    """The file cannot be used; the message says why and where."""


# ---------------------------------------------------------------- reading text


def decode(raw: bytes) -> str:
    """The text of an upload: UTF-8 (with or without BOM), else Latin-1 which never fails."""
    if len(raw) > MAX_BYTES:
        raise DataError(
            f"the file is {len(raw) // 1000} KB; the limit is {MAX_BYTES // 1000} KB"
        )
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _rows(text: str) -> tuple[list[list[str]], str]:
    if not text.strip():
        raise DataError("the file is empty")
    head = text[:4096]
    delim = max((",", ";", "\t"), key=head.count)
    rows = [
        [c.strip() for c in r] for r in csv.reader(io.StringIO(text), delimiter=delim)
    ]
    rows = [r for r in rows if any(r)]
    return rows, delim


def _num(cell: str, delim: str) -> float | None:
    s = cell.strip()
    if delim == ";":
        s = s.replace(",", ".")
    try:
        v = float(s)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


# ---------------------------------------------------------------- prices


def parse_prices(text: str, min_rows: int = MIN_ROWS) -> dict[str, Any]:
    """A table of prices: ``columns``, ``dates`` (or None), ``prices`` (rows by columns), ``dropped``.

    The first row holds the names. A column called Date, Time, Timestamp or Datetime is the date.
    Every other column of numbers is a price series. A file from Yahoo Finance has the columns Open,
    High, Low, Close, Volume: only Close is used, and its second header row (Ticker) names it. A
    row with a missing or non-numeric price is dropped and counted. A price of 0 or less is an
    error: the chain works on log returns.

    Raises:
        DataError: if the file is empty, has no column of numbers, a price of 0 or less, or fewer
            than ``min_rows`` usable rows (or more than ``MAX_ROWS``).
    """
    rows, delim = _rows(text)
    header = rows[0]
    i = 1
    tickers = None
    while i < len(rows) and not any(_num(c, delim) is not None for c in rows[i][1:]):
        if rows[i][0].lower() == "ticker":
            tickers = rows[i]
        i += 1
    data = rows[i:]
    if not data:
        raise DataError("there are no rows of numbers under the header")
    if len(data) > MAX_ROWS:
        raise DataError(f"the file has {len(data)} rows; the limit is {MAX_ROWS}")
    low = [h.lower() for h in header]
    date_idx = next((j for j, h in enumerate(low) if h in DATE_NAMES), None)
    if date_idx is None and data and _num(data[0][0], delim) is None:
        date_idx = 0
    cand = [j for j in range(len(header)) if j != date_idx]
    if "close" in low and any(x in low for x in ("open", "high", "low")):
        cand = [j for j in cand if low[j] == "close"]
    keep = []
    for j in cand:
        ok = sum(1 for r in data if j < len(r) and _num(r[j], delim) is not None)
        if ok >= 0.9 * len(data):
            keep.append(j)
    if not keep:
        raise DataError(
            "no column of numbers was found: the prices must be numbers, one column per asset"
        )
    names = [
        (tickers[j] if tickers and j < len(tickers) and tickers[j] else header[j])
        or f"col{j}"
        for j in keep
    ]
    out, dates, dropped = [], [], 0
    for r in data:
        vals = [_num(r[j], delim) if j < len(r) else None for j in keep]
        if any(v is None for v in vals):
            dropped += 1
            continue
        out.append(vals)
        dates.append(r[date_idx] if date_idx is not None and date_idx < len(r) else "")
    if len(out) < min_rows:
        raise DataError(f"only {len(out)} usable rows; at least {min_rows} are needed")
    arr = np.asarray(out, dtype=float)
    bad = np.argwhere(arr <= 0)
    if len(bad):
        r, c = bad[0]
        raise DataError(
            f"column {names[c]!r} has a price of {arr[r, c]:g} in row {r + 1} of the data: "
            "a price must be above 0 (the chain works on log returns)"
        )
    return {
        "columns": names,
        "dates": dates if date_idx is not None else None,
        "prices": arr,
        "dropped": dropped,
    }


def log_returns(prices: np.ndarray) -> np.ndarray:
    """The log returns, one row fewer than the prices."""
    return np.diff(np.log(prices), axis=0)


# ---------------------------------------------------------------- orders


def _column(low: list[str], names: tuple[str, ...]) -> int | None:
    return next((low.index(n) for n in names if n in low), None)


def _when(cell: str) -> float | None:
    v = _num(cell, ",")
    if v is not None:
        return v
    try:
        return datetime.fromisoformat(cell.strip().replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def parse_orders(text: str) -> list[dict[str, Any]]:
    """The orders of a CSV: ``account``, ``side``, ``size``, ``price``, ``ref_price``, ``t``, ``row``.

    The header names the columns (account, stock, symbol or ticker; side; size, shares, quantity or
    qty; price; optional ref_price; optional time as seconds or an ISO date). Without ``ref_price``
    the first price seen for the account is its reference, as in the demo. Without a time column
    the orders are one second apart, in file order.

    Raises:
        DataError: for a missing column, a value that is not a number, a side that is not buy or
            sell, a size or price of 0 or less, or more than ``MAX_ORDERS`` orders.
    """
    rows, delim = _rows(text)
    low = [h.lower() for h in rows[0]]
    need = {
        "account": _column(low, ACCOUNT_NAMES),
        "side": _column(low, SIDE_NAMES),
        "size": _column(low, SIZE_NAMES),
        "price": _column(low, PRICE_NAMES),
    }
    missing = [k for k, v in need.items() if v is None]
    if missing:
        raise DataError(
            "the header must have a column for each of: account (or stock, symbol, ticker), side, "
            f"size (or shares, quantity, qty) and price; missing: {', '.join(missing)}"
        )
    ref_i = _column(low, REF_NAMES)
    t_i = _column(low, TIME_NAMES)
    body = rows[1:]
    if not body:
        raise DataError("there are no orders under the header")
    if len(body) > MAX_ORDERS:
        raise DataError(f"the file has {len(body)} orders; the limit is {MAX_ORDERS}")
    first_price: dict[str, float] = {}
    orders = []
    for k, r in enumerate(body):
        row = k + 2

        def cell(j: int | None) -> str:
            return r[j] if j is not None and j < len(r) else ""

        side_raw = cell(need["side"]).lower()
        side = "buy" if side_raw in BUY else "sell" if side_raw in SELL else None
        if side is None:
            raise DataError(
                f"row {row}: the side {cell(need['side'])!r} is not buy or sell"
            )
        size, price = _num(cell(need["size"]), delim), _num(cell(need["price"]), delim)
        if size is None or size <= 0:
            raise DataError(
                f"row {row}: the size {cell(need['size'])!r} is not a number above 0"
            )
        if price is None or price <= 0:
            raise DataError(
                f"row {row}: the price {cell(need['price'])!r} is not a number above 0"
            )
        acc = cell(need["account"])
        if not acc:
            raise DataError(f"row {row}: the account is empty")
        ref = _num(cell(ref_i), delim) if ref_i is not None and cell(ref_i) else None
        if ref is not None and ref <= 0:
            raise DataError(
                f"row {row}: the ref_price {cell(ref_i)!r} is not a number above 0"
            )
        first_price.setdefault(acc, price)
        t = _when(cell(t_i)) if t_i is not None and cell(t_i) else None
        if t_i is not None and cell(t_i) and t is None:
            raise DataError(
                f"row {row}: the time {cell(t_i)!r} is not seconds or an ISO date"
            )
        orders.append(
            {
                "row": row,
                "account": acc,
                "side": side,
                "size": size,
                "price": price,
                "ref_price": ref if ref is not None else first_price[acc],
                "t": t if t is not None else float(k),
            }
        )
    return orders


# ---------------------------------------------------------------- storage


def uploads_dir(base: Path) -> Path:
    d = Path(base) / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def store_prices(folder: Path, name: str, raw: bytes) -> dict[str, Any]:
    """Check an upload and keep it; return what the page needs to show (no arrays)."""
    parsed = parse_prices(decode(raw))
    uid = "up_" + hashlib.sha1(raw).hexdigest()[:10]
    d = uploads_dir(folder)
    (d / f"{uid}.csv").write_bytes(raw)
    dates = parsed["dates"]
    meta = {
        "id": uid,
        "name": Path(name or "prices.csv").name[:80],
        "columns": parsed["columns"],
        "n_rows": int(parsed["prices"].shape[0]),
        "dropped": parsed["dropped"],
        "first_date": dates[0] if dates else None,
        "last_date": dates[-1] if dates else None,
        "uploaded": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (d / f"{uid}.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def load_prices(folder: Path, uid: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """The parsed table and the meta of an upload; ``DataError`` if the id is not one of ours."""
    if not _ID.match(uid or ""):
        raise DataError(f"{uid!r} is not an uploaded file")
    d = uploads_dir(folder)
    f = d / f"{uid}.csv"
    if not f.is_file():
        raise DataError(
            f"the uploaded file {uid} is not there any more: upload it again"
        )
    meta = json.loads((d / f"{uid}.json").read_text(encoding="utf-8"))
    return parse_prices(decode(f.read_bytes())), meta


def list_uploads(folder: Path) -> list[dict[str, Any]]:
    out = []
    for f in sorted(uploads_dir(folder).glob("up_*.json")):
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return sorted(out, key=lambda m: m.get("uploaded", ""), reverse=True)


# ---------------------------------------------------------------- reports


def _flat(prefix: str, value: Any, out: list[tuple[str, str]]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _flat(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _flat(f"{prefix}[{i}]", v, out)
    else:
        out.append((prefix, "" if value is None else str(value)))


def report_rows(result: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    """The rows ``(section, name, index, value)`` of a chain result."""
    rows: list[tuple[str, str, str, str]] = []
    for k, v in (result.get("chain") or {}).items():
        rows.append(("chain", str(k), "", str(v)))
    for k, v in (result.get("data_meta") or {}).items():
        rows.append(("data", str(k), "", str(v)))
    flat: list[tuple[str, str]] = []
    metrics = dict(result.get("metrics") or {})
    metrics.pop("role_attributes", None)
    _flat("", metrics, flat)
    for k, v in flat:
        rows.append(("metric", k, "", v))
    for name, pts in (result.get("series") or {}).items():
        for p in pts:
            rows.append(
                (f"series_{name}", name, str(p.get("i", "")), str(p.get("v", "")))
            )
    for i in result.get("alarms") or []:
        rows.append(("alarm", "risk", str(i), ""))
    return rows


def report_csv(result: dict[str, Any]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["section", "name", "index", "value"])
    w.writerows(report_rows(result))
    return buf.getvalue()


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def report_path(folder: Path, rid: str, suffix: str) -> Path:
    """The file ``<rid>.<suffix>`` of a run; ``DataError`` if the id is not a run id or the file is missing."""
    if not _RUN_ID.match(rid or ""):
        raise DataError(f"{rid!r} is not a run id")
    p = Path(folder) / "reports" / f"{rid}.{suffix}"
    if not p.is_file():
        raise DataError(f"the file {rid}.{suffix} is not there")
    return p


def save_report(folder: Path, result: dict[str, Any]) -> str:
    """Keep a result under a new run id, written into the result as ``run_id``, and return the id."""
    rid = new_run_id()
    result["run_id"] = rid
    d = Path(folder) / "reports"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{rid}.json").write_text(json.dumps(result), encoding="utf-8")
    return rid


def load_report(folder: Path, rid: str) -> dict[str, Any]:
    if not _RUN_ID.match(rid or ""):
        raise DataError(f"{rid!r} is not a run id")
    f = Path(folder) / "reports" / f"{rid}.json"
    if not f.is_file():
        raise DataError(f"the run {rid} is not there")
    return json.loads(f.read_text(encoding="utf-8"))


def orders_csv(results: list[dict[str, Any]]) -> str:
    """The checked orders as a CSV, one row per order."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(
        [
            "row",
            "account",
            "side",
            "size",
            "price",
            "ref_price",
            "t",
            "approved",
            "reason",
        ]
    )
    for r in results:
        w.writerow(
            [
                r["row"],
                r["account"],
                r["side"],
                r["size"],
                r["price"],
                r["ref_price"],
                r["t"],
                int(r["approved"]),
                r["reason"],
            ]
        )
    return buf.getvalue()


# ---------------------------------------------------------------- pre-trade checks


LIMIT_KEYS = (
    "max_notional",
    "max_size",
    "price_band",
    "max_orders",
    "window_s",
    "duplicate_window_s",
)


def read_limits(
    base: dict[str, Any], override: dict[str, Any] | None
) -> dict[str, Any]:
    """The limits of the checker: ``base`` with the numbers of ``override`` on top.

    Raises:
        DataError: for a limit that does not exist, or a value that is not a number above 0
            (``max_orders`` must be a whole number).
    """
    out = dict(base)
    for k, v in (override or {}).items():
        if k not in LIMIT_KEYS:
            raise DataError(
                f"unknown limit {k!r}; the limits are {', '.join(LIMIT_KEYS)}"
            )
        if v is None or str(v).strip() == "":
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            raise DataError(f"the limit {k} must be a number, got {v!r}") from None
        if not math.isfinite(x) or x <= 0:
            raise DataError(f"the limit {k} must be above 0, got {v!r}")
        if k == "max_orders":
            if x != int(x):
                raise DataError("the limit max_orders must be a whole number")
            out[k] = int(x)
        else:
            out[k] = x
    return out


def check_orders(
    orders: list[dict[str, Any]], limits: dict[str, Any], log_path: Path
) -> dict[str, Any]:
    """Send every order to a ``PreTradeChecker`` and write the decisions in a hash-chained log.

    The orders are checked in time order (the rate and duplicate controls need it); a row keeps
    its ``row`` number. The log is verified here, with its file in place.
    """
    from dense_armor.core.finance.decision_log import DecisionLog
    from dense_armor.core.finance.pretrade import PreTradeChecker

    try:
        checker = PreTradeChecker(**limits)
    except (TypeError, ValueError) as e:
        raise DataError(f"the limits are not usable: {e}") from None
    results, counts, records = [], {}, []
    for o in sorted(orders, key=lambda x: (x["t"], x["row"])):
        d = checker.check(
            account=o["account"],
            side=o["side"],
            size=o["size"],
            price=o["price"],
            ref_price=o["ref_price"],
            t=o["t"],
        )
        counts[d.reason] = counts.get(d.reason, 0) + 1
        results.append({**o, "approved": bool(d.approved), "reason": d.reason})
        records.append(
            (
                o["t"],
                {
                    "approved": bool(d.approved),
                    "order_ref": o["row"],
                    "reason": d.reason,
                    "stock": o["account"],
                },
            )
        )
    log = DecisionLog(log_path)
    log.append_many(records)
    return {
        "results": results,
        "counts": counts,
        "n": len(results),
        "approved": sum(1 for r in results if r["approved"]),
        "log": {
            "records": len(log),
            "verified": log.verify(),
            "last_hash": log.last_hash,
        },
    }


# ---------------------------------------------------------------- schedules


def parse_history(text: str) -> list[list[float]]:
    """A volume history: one session per line, the volume of each bucket separated by commas.

    A first line that is not numbers is a header and is skipped.
    """
    rows, delim = _rows(text)
    out: list[list[float]] = []
    for k, r in enumerate(rows):
        vals = [_num(c, delim) for c in r]
        if any(v is None for v in vals):
            if k == 0:
                continue
            raise DataError(f"line {k + 1}: every volume must be a number")
        out.append([float(v) for v in vals if v is not None])
    if not out:
        raise DataError("the volume history has no rows of numbers")
    return out


def schedule_plan(
    total: float, n: int, history: list[list[float]] | None = None
) -> dict[str, Any]:
    """The TWAP and VWAP slices of an order, and the plan as a CSV (``bucket,twap,vwap``).

    Without a ``history`` the volume is taken as flat, and the VWAP equals the TWAP.

    Raises:
        DataError: if ``total`` is not a number of 0 or more, ``n`` is not a whole number from 1 to
            1000, or a row of the history has not ``n`` volumes above 0.
    """
    from dense_armor.core.finance.schedule import twap_schedule, vwap_schedule

    try:
        total = float(total)
    except (TypeError, ValueError):
        raise DataError("the total must be a number") from None
    if not math.isfinite(total) or total < 0:
        raise DataError("the total must be 0 or more")
    try:
        n_f = float(n)
    except (TypeError, ValueError):
        raise DataError(
            "the number of buckets must be a whole number from 1 to 1000"
        ) from None
    if (
        isinstance(n, bool)
        or not math.isfinite(n_f)
        or n_f != int(n_f)
        or not 1 <= n_f <= 1000
    ):
        raise DataError("the number of buckets must be a whole number from 1 to 1000")
    n = int(n_f)
    if history is None:
        arr, flat = np.ones((1, n)), True
    else:
        flat = False
        if len(history) > MAX_ROWS:
            raise DataError(
                f"the history has {len(history)} sessions; the limit is {MAX_ROWS}"
            )
        for i, r in enumerate(history):
            if len(r) != n:
                raise DataError(
                    f"session {i + 1} of the history has {len(r)} buckets; the plan has {n}"
                )
        arr = np.asarray(history, dtype=float)
        if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
            raise DataError("every volume of the history must be a number above 0")
    twap = [float(x) for x in twap_schedule(total, n)]
    vwap = [float(x) for x in vwap_schedule(total, arr)]
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["bucket", "twap", "vwap"])
    for i in range(n):
        w.writerow([i, f"{twap[i]:.6f}", f"{vwap[i]:.6f}"])
    return {
        "total": total,
        "n": n,
        "twap": twap,
        "vwap": vwap,
        "flat_history": flat,
        "sessions": int(arr.shape[0]),
        "csv": buf.getvalue(),
    }


# ---------------------------------------------------------------- sample files


def sample_prices_csv(fixtures: Path, n: int = 500) -> str:
    """The last ``n`` common days of the four tickers of the fixtures, as ``Date,GSPC,AAPL,MSFT,TSLA``."""
    tickers = ("GSPC", "AAPL", "MSFT", "TSLA")
    cols, first = {}, []
    for t in tickers:
        p = parse_prices(
            (Path(fixtures) / f"{t}.csv").read_text(encoding="utf-8"), min_rows=1
        )
        cols[t] = dict(zip(p["dates"], p["prices"][:, 0], strict=True))
        if not first:
            first = list(p["dates"])
    common = [d for d in first if all(d in cols[t] for t in tickers)][-n:]
    lines = ["Date," + ",".join(tickers)]
    lines += [d + "," + ",".join(f"{cols[t][d]:.4f}" for t in tickers) for d in common]
    return "\n".join(lines) + "\n"


def sample_volumes_csv(buckets: int = 10, sessions: int = 20, seed: int = 3) -> str:
    """A made-up volume history: ``sessions`` days of ``buckets`` volumes, high at the open and the close."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, buckets)
    shape = 1.0 + 1.5 * x**2
    rows = [",".join(f"b{i}" for i in range(buckets))]
    for _ in range(sessions):
        v = np.round(1000.0 * shape * rng.lognormal(0.0, 0.15, buckets))
        rows.append(",".join(f"{int(a)}" for a in v))
    return "\n".join(rows) + "\n"


def sample_orders_csv(n: int = 60, seed: int = 7) -> str:
    """A made-up list of orders, with a few that break a limit, to try the pre-trade page."""
    rng = np.random.default_rng(seed)
    base = {"AAPL": 150.0, "MSFT": 300.0, "TSLA": 700.0}
    names = list(base)
    t, rows, last = 0.0, ["account,side,size,price,ref_price,t"], None
    for _ in range(n):
        if last is not None and rng.random() < 0.06:
            acc, side, size, price = last
            t += 0.01
        else:
            acc = names[int(rng.integers(len(names)))]
            side = "buy" if rng.random() < 0.5 else "sell"
            size = float(max(1, round(float(rng.lognormal(3.0, 1.0)))))
            drift = float(rng.normal(0, 0.012)) + (0.08 if rng.random() < 0.08 else 0.0)
            price = round(base[acc] * (1 + drift), 2)
            if rng.random() < 0.05:
                size *= 12
            t += float(rng.exponential(0.3))
        last = (acc, side, size, price)
        rows.append(f"{acc},{side},{size:g},{price},{base[acc]},{t:.3f}")
    return "\n".join(rows) + "\n"
