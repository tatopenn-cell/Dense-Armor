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


def _build_data(n):
    return [f"/path/to/file/{i}" for i in range(n)]


def _read_worker(args):
    handle, indices = args
    pid = os.getpid()
    proc = psutil.Process(pid)
    _ = handle[0]
    before = proc.memory_full_info()
    for i in indices:
        _ = handle[i]
    after = proc.memory_full_info()
    return {
        "uss": after.uss - before.uss,
        "pss": after.pss - before.pss,
    }


def run_benchmark(n_items, fraction, contender, start_method):
    ctx = get_context(start_method)
    data = _build_data(n_items)

    if contender == "plain_list":
        handle = data
    elif contender == "numpy_object":
        handle = np.array(data, dtype=object)
    elif contender == "manager_list":
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

    n_reads = int(n_items * fraction)
    random.seed(42)
    indices = random.sample(range(n_items), n_reads)
    chunk_size = max(1, len(indices) // 4)
    chunks = [indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)]

    with ctx.Pool(4) as pool:
        results = pool.map(_read_worker, [(handle, c) for c in chunks])

    uss_vals = [r["uss"] for r in results]
    pss_vals = [r["pss"] for r in results]

    if contender == "shareable_list":
        handle.shm.close()
        handle.shm.unlink()
    elif contender in ("shared_strings", "shared_list"):
        handle.unlink()
        handle.close()

    return {
        "uss_median": statistics.median(uss_vals),
        "pss_median": statistics.median(pss_vals),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", default="benchmarks/data_shared_memory.json")
    args = parser.parse_args()

    n_items = 100_000 if args.smoke else 1_000_000
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
            try:
                start = time.time()
                res = run_benchmark(n_items, frac, cont, "fork")
                res["time_s"] = time.time() - start
                results[cont][f"fork_{frac}"] = res
            except Exception as e:
                results[cont][f"fork_{frac}"] = {"error": str(e)}

            if (
                cont
                in ("shareable_list", "shared_strings", "shared_list", "plain_list")
                and frac == 1.0
            ):
                start = time.time()
                try:
                    res = run_benchmark(n_items, frac, cont, "spawn")
                    res["time_s"] = time.time() - start
                    results[cont][f"spawn_{frac}"] = res
                except Exception as e:
                    results[cont][f"spawn_{frac}"] = {"error": str(e)}

    print(json.dumps(results, indent=2))
    if not args.smoke:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
