import argparse
import json
import os
import random
import statistics
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).parent.parent))
from dense_armor.utility.data import SharedList, SharedStrings

_handle = None


def _init_worker(handle):
    global _handle
    _handle = handle


def _build_data(n):
    return [f"/path/to/file/{i}" for i in range(n)]


def _read_worker(args):
    n_reads, seed = args
    pid = os.getpid()
    proc = psutil.Process(pid)
    h = _handle
    _ = h[0]
    before = proc.memory_full_info()
    rng = random.Random(seed)
    idx = rng.sample(range(len(h)), n_reads)
    lat = []
    for i in idx[:1000]:
        t0 = time.perf_counter_ns()
        _ = h[i]
        t1 = time.perf_counter_ns()
        lat.append(t1 - t0)
    for i in idx:
        _ = h[i]
    after = proc.memory_full_info()
    return {
        "uss": after.uss - before.uss,
        "pss": after.pss - before.pss,
        "lat_median_ns": statistics.median(lat),
    }


def _run_once(n_items, n_reads, contender, start_method):
    data = _build_data(n_items)
    if contender == "plain_list":
        handle = data
    elif contender == "numpy_object":
        handle = np.array(data, dtype=object)
    elif contender == "manager_list":
        ctx = get_context(start_method)
        handle = ctx.Manager().list(data)
    elif contender == "shareable_list":
        from multiprocessing.shared_memory import ShareableList

        handle = ShareableList(data)
    elif contender == "shared_strings":
        handle = SharedStrings(data)
    elif contender == "shared_list":
        handle = SharedList(data)
    else:
        raise ValueError(contender)

    ctx = get_context(start_method)
    tasks = [(n_reads, i) for i in range(4)]
    global _handle
    if start_method == "fork":
        _handle = handle
        with ctx.Pool(4) as pool:
            results = pool.map(_read_worker, tasks)
        _handle = None
    else:
        with ctx.Pool(4, initializer=_init_worker, initargs=(handle,)) as pool:
            results = pool.map(_read_worker, tasks)

    uss = [r["uss"] for r in results]
    pss = [r["pss"] for r in results]
    lat = [r["lat_median_ns"] for r in results]

    if contender == "shareable_list":
        handle.shm.close()
        handle.shm.unlink()
    elif contender in ("shared_strings", "shared_list"):
        handle.unlink()
        handle.close()

    return {
        "uss_median": statistics.median(uss),
        "pss_median": statistics.median(pss),
        "lat_median_us": statistics.median(lat) / 1000.0,
    }


def _iqr(vals):
    s = sorted(vals)
    n = len(s)
    if n < 2:
        return 0.0
    return s[int(0.75 * (n - 1))] - s[int(0.25 * (n - 1))]


def _run_5(n_items, n_reads, contender, start_method):
    runs = [_run_once(n_items, n_reads, contender, start_method) for _ in range(5)]
    uss = [r["uss_median"] for r in runs]
    pss = [r["pss_median"] for r in runs]
    lat = [r["lat_median_us"] for r in runs]
    return {
        "uss_median": statistics.median(uss),
        "uss_iqr": _iqr(uss),
        "pss_median": statistics.median(pss),
        "pss_iqr": _iqr(pss),
        "lat_median_us": statistics.median(lat),
    }


def _evaluate_claims(results):
    plain = results["plain_list"]["fork_1.0"]
    ss = results["shared_strings"]["fork_1.0"]
    sl = results["shared_list"]["fork_1.0"]
    mgr = results["manager_list"]["fork_1.0"]
    ss_spawn = results["shared_strings"].get("spawn_1.0", {})
    sl_spawn = results["shared_list"].get("spawn_1.0", {})

    c1_ss = ss["uss_median"] / plain["uss_median"]
    c1_sl = sl["uss_median"] / plain["uss_median"]
    claim1 = c1_ss < 0.5 and c1_sl < 0.5

    c2_ss = ss["lat_median_us"] <= 10 * plain["lat_median_us"]
    c2_sl = sl["lat_median_us"] * 10 <= mgr["lat_median_us"]
    claim2 = c2_ss and c2_sl

    claim3 = True
    if ss_spawn and sl_spawn:
        c3_ss = abs(ss_spawn["uss_median"] - ss["uss_median"]) / ss["uss_median"] <= 0.2
        c3_sl = abs(sl_spawn["uss_median"] - sl["uss_median"]) / sl["uss_median"] <= 0.2
        claim3 = c3_ss and c3_sl

    return {
        "claim1_uss_0.5x": "holds" if claim1 else "does not hold",
        "claim2_latency": "holds" if claim2 else "does not hold",
        "claim3_spawn_20pct": "holds" if claim3 else "does not hold",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", default="benchmarks/data_shared_memory.json")
    args = parser.parse_args()

    n_items = 100_000 if args.smoke else 1_000_000
    runs = 1 if args.smoke else 5
    fractions = [0.1, 0.5, 1.0]
    contenders = [
        "plain_list",
        "numpy_object",
        "manager_list",
        "shareable_list",
        "shared_strings",
        "shared_list",
    ]

    results = {}
    for cont in contenders:
        results[cont] = {}
        for frac in fractions:
            n_reads = int(n_items * frac)
            if runs == 1:
                results[cont][f"fork_{frac}"] = _run_once(
                    n_items, n_reads, cont, "fork"
                )
            else:
                results[cont][f"fork_{frac}"] = _run_5(n_items, n_reads, cont, "fork")
            if (
                cont
                in ("shareable_list", "shared_strings", "shared_list", "plain_list")
                and frac == 1.0
            ):
                if runs == 1:
                    results[cont]["spawn_1.0"] = _run_once(
                        n_items, n_reads, cont, "spawn"
                    )
                else:
                    results[cont]["spawn_1.0"] = _run_5(n_items, n_reads, cont, "spawn")

    results["claims"] = _evaluate_claims(results)

    print(json.dumps(results, indent=2))
    if not args.smoke:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
