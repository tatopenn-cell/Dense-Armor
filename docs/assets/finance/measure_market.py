"""Run the market pipelines on the fixtures and write results.json.

Run from the repo root: python docs/assets/finance/measure_market.py
Needs a machine with plenty of free RAM (the Orca memory guard asks for 15%).
The numbers on the finance pages come from a run on a 4-core CPU with 31 GB of RAM.
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from pipelines.finance import (
    ExecutionPipeline,
    FinancePipeline,
    OnlineAnalysis,
    RiskPipeline,
    SingleTickerPipeline,
    StaticAnalysis,
)

FIXT = Path("test/fixtures/finance")
OUT = Path(__file__).resolve().parent / "results.json"
R = {}


def save():
    OUT.write_text(json.dumps(R), encoding="utf-8")


def load(ticker):
    dates, px, vol = [], [], []
    with (FIXT / f"{ticker}.csv").open() as fh:
        r = csv.reader(fh)
        next(r)
        next(r)
        next(r)
        for row in r:
            if row and row[0]:
                dates.append(row[0])
                px.append(float(row[1]))
                vol.append(float(row[5]))
    return dates, np.array(px), np.array(vol)


def ranges(prices, vol_rel, n=250):
    rc = np.diff(prices[:n]) / prices[:n][:-1]
    return (
        (float(np.percentile(rc, 0.5)), float(np.percentile(rc, 99.5))),
        (
            float(np.percentile(vol_rel[:n], 0.5)),
            float(np.percentile(vol_rel[:n], 99.5)),
        ),
    )


def say(*a):
    print(*a, flush=True)


def surveillance(ticker):
    d, p, v = load(ticker)
    vr = v / np.nanmean(v)
    rr, rv = ranges(p, vr)
    pl = FinancePipeline(range_ret=rr, range_vol=rv, seed=0)
    flags, ts, disc, cus, orca = [], [], [], [], []
    for i in range(1, len(p)):
        t0 = time.perf_counter()
        e = pl.process_tick(float(p[i]), float(vr[i]))
        ts.append(time.perf_counter() - t0)
        flags.append(bool(e.anomalous))
        disc.append(bool(e.discarded_by_armatura))
        cus.append(bool(e.drift_flagged))
        orca.append(bool(e.orca_dampened))
        if i % 250 == 0:
            say("  ", ticker, "tick", i)
    ret = (p[1:] - p[:-1]) / p[:-1]
    post = np.arange(1, len(p)) >= 250
    an = [i for i in range(len(flags)) if flags[i] and post[i]]
    R["surv_" + ticker] = {
        "dates": d,
        "close": p.tolist(),
        "ret": ret.tolist(),
        "anomalous": flags,
        "post_from": 249,
    }
    say(
        ticker,
        "n_ticks",
        len(flags),
        "armatura_all",
        sum(disc),
        "armatura_post",
        sum(x for x, q in zip(disc, post) if q),
        "cusum_all",
        sum(cus),
        "cusum_post",
        sum(x for x, q in zip(cus, post) if q),
        "anomalous_post",
        len(an),
        "orca_all",
        sum(orca),
        "orca_post",
        sum(x for x, q in zip(orca, post) if q),
        "ms_per_tick median",
        round(float(np.median(ts)) * 1e3, 2),
        "p99",
        round(float(np.quantile(ts, 0.99)) * 1e3, 2),
    )
    say("   anomalous days:", [(d[i + 1], round(ret[i] * 100, 2)) for i in an])
    save()


surveillance("GSPC")

d, p, v = load("GSPC")
r = np.diff(p) / p[:-1]
for th in (0.02, 0.03, 0.04):
    s = StaticAnalysis(threshold=th).analyze(r)
    say("static", th, s.n_anomalous, "of", s.n_days)
    if th == 0.03:
        st_idx = list(s.anomalous_indices)
vol_rel = v / np.mean(v)
rets = [0.0] + list(r)
rr, rv = ranges(p, v / np.nanmean(v))
oa = OnlineAnalysis(range_ret=rr, range_vol=rv, seed=0).analyze(rets, list(vol_rel))
on_idx = list(oa.anomalous_indices)
say(
    "online",
    oa.n_anomalous,
    "thr",
    round(oa.final_loda_threshold, 3),
    round(oa.final_hst_threshold, 3),
    "overlap",
    len(set(st_idx) & set(on_idx)),
    "only static",
    len(set(st_idx) - set(on_idx)),
    "only online",
    len(set(on_idx) - set(st_idx)),
)
say("  index base check (static idx 0 is r[0]?)", st_idx[:3], on_idx[:3])
R["static_online"] = {
    "dates": d,
    "ret": r.tolist(),
    "static_idx": st_idx,
    "online_idx": on_idx,
    "static_note": "static idx refers to r (diff), online idx refers to rets (prepended 0.0)",
}
save()

loss = np.maximum(-r, 0.0)
rp = RiskPipeline(alpha=0.01, loss_limit=0.05, seed=0)
ts, es = [], []
for x in loss:
    t0 = time.perf_counter()
    e = rp.process_day(float(x))
    ts.append(time.perf_counter() - t0)
    es.append(e)
over = [i for i, e in enumerate(es) if e.over_limit]
say(
    "risk var99",
    round(es[-1].var_99 * 100, 3),
    "var999",
    round(es[-1].var_999 * 100, 3),
    "over idx",
    over,
    [d[i + 1] for i in over],
    "vol drift",
    sum(e.volatility_drift for e in es),
    "of",
    len(es),
    "us/day median",
    round(float(np.median(ts)) * 1e6),
)
say(
    "   days with loss > var99_final:",
    int((loss > es[-1].var_99).sum()),
    " > var999_final:",
    int((loss > es[-1].var_999).sum()),
)
R["risk"] = {
    "dates": d,
    "loss": loss.tolist(),
    "var99": es[-1].var_99,
    "var999": es[-1].var_999,
    "over": over,
}
save()

da, pa, va = load("AAPL")
var_ = va / np.nanmean(va)
rra, rva = ranges(pa, var_)
t = SingleTickerPipeline(range_ret=rra, range_vol=rva, seed=0)
fl, ts = [], []
for i in range(1, len(pa)):
    t0 = time.perf_counter()
    e = t.process_day(float((pa[i] - pa[i - 1]) / pa[i - 1]), float(var_[i]))
    ts.append(time.perf_counter() - t0)
    fl.append(bool(e.anomalous))
ra = ((pa[1:] - pa[:-1]) / pa[:-1]).tolist()
say(
    "single AAPL anomalies (all days)",
    sum(fl),
    "of",
    len(fl),
    "ms/day",
    round(float(np.median(ts)) * 1e3, 2),
    "| after day 250:",
    sum(fl[249:]),
)
say(
    "   flagged:", [(da[i + 1], round(ra[i] * 100, 2)) for i in range(len(fl)) if fl[i]]
)
R["single_AAPL"] = {"dates": da, "ret": ra, "anomalous": fl}
save()
for tk in ("TSLA", "MSFT"):
    dt_, pt_, vt_ = load(tk)
    vv = vt_ / np.nanmean(vt_)
    rrt, rvt = ranges(pt_, vv)
    tt = SingleTickerPipeline(range_ret=rrt, range_vol=rvt, seed=0)
    ff = []
    for i in range(1, len(pt_)):
        ff.append(
            bool(
                tt.process_day(
                    float((pt_[i] - pt_[i - 1]) / pt_[i - 1]), float(vv[i])
                ).anomalous
            )
        )
    say(
        "single",
        tk,
        "anomalies",
        sum(ff),
        "of",
        len(ff),
        "| after day 250:",
        sum(ff[249:]),
        [
            (dt_[i + 1], round(100 * (pt_[i + 1] - pt_[i]) / pt_[i], 2))
            for i in range(len(ff))
            if ff[i]
        ][:12],
    )

sizes = []
with (FIXT / "coinbase_btcusd_trades.csv").open() as fh:
    for row in csv.DictReader(fh):
        sizes.append(float(row["size"]))
ep = ExecutionPipeline(seed=0)
ts, res = [], []
for s in sizes:
    t0 = time.perf_counter()
    e = ep.process_order(float(s))
    ts.append(time.perf_counter() - t0)
    res.append(e)
say(
    "execution orders",
    ep.n_orders,
    "direct",
    ep.n_passed_directly,
    "orca",
    ep.n_through_orca,
    "refused",
    ep.n_refused,
    "us/order median",
    round(float(np.median(ts)) * 1e6, 1),
    "p99",
    round(float(np.quantile(ts, 0.99)) * 1e6, 1),
    "min",
    min(sizes),
    "max",
    max(sizes),
    "dampened",
    sum(
        1
        for e in res
        if e.reduction > 0 and e.approved and e.sent_size < e.original_size
    ),
)
R["execution"] = {
    "size": sizes,
    "approved": [bool(e.approved) for e in res],
    "reason": [e.reason for e in res],
}
save()
say("FINE")
