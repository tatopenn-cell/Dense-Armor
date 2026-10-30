"""Render the float32 cost line of each page from the benchmark JSON.

Usage: python tools/render_float32_sections.py [--check]

Without --check, rewrites the "Cost, median of ..." paragraph in each
page from the numbers in benchmarks/float32_t4.json. With --check,
prints OK or DIFF for each page and exits non-zero on any DIFF.

The page paths and the class names are fixed in PAGES; every other
paragraph of the page is untouched.
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JSON = ROOT / "benchmarks" / "float32_t4.json"

PAGES = [
    ("docs/proba/index.md", "Gaussian"),
    ("docs/learn/online_classifiers.md", "OnlineGaussianNB"),
    ("docs/preprocessing/features.md", "SelectKBest"),
    ("docs/metrics/regression.md", "R2Score"),
]

PREFIX = "Cost, median of"
PATTERN = re.compile(
    r"Cost,\s+median\s+of\s+\d+\s+runs\s+of\s+one\s+update\s+over\s+"
    r"[\d,]+\s+samples,\s+one\s+CPU\s+thread:\s+default\s+"
    r"[-0-9.]+\s+ms\s+\(IQR\s+[-0-9.]+\)\s*,\s+float32\s+"
    r"[-0-9.]+\s+ms\s+\(IQR\s+[-0-9.]+\)\."
)


def render(n_runs, n_samples, default_ms, default_iqr, f32_ms, f32_iqr):
    return (
        f"Cost, median of {n_runs} runs of one update over "
        f"{n_samples:,} samples, one CPU thread: default "
        f"{default_ms:.6f} ms (IQR {default_iqr:.6f}), float32 "
        f"{f32_ms:.6f} ms (IQR {f32_iqr:.6f})."
    )


def main(check):
    data = json.loads(JSON.read_text())
    rows = {r["class"]: r for r in data["rows"]}
    n_runs = data["runs"]
    n_samples = data["n"]

    diffs = 0
    for rel_path, cls in PAGES:
        p = ROOT / rel_path
        src = p.read_text()
        row = rows[cls]
        expected = render(
            n_runs,
            n_samples,
            row["default_median_ms"],
            row["default_iqr_ms"],
            row["f32_median_ms"],
            row["f32_iqr_ms"],
        )
        matches = PATTERN.findall(src)
        if not matches:
            print(f"{rel_path}: NO MATCH")
            diffs += 1
            continue
        if check:
            if expected in src:
                print(f"{rel_path}: OK")
            else:
                print(f"{rel_path}: DIFF")
                diffs += 1
        else:
            new = PATTERN.sub(expected, src)
            if new != src:
                p.write_text(new)
                print(f"{rel_path}: rewritten")
            else:
                print(f"{rel_path}: already current")
    if check and diffs:
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    main(args.check)
