"""Kaggle kernel: full video_quality measurement at 1280x720 and 1920x1080.

Item E of task Z2. The full measurement at 1280x720 and 1920x1080 does
not fit in one Colab session (2 cores at 1.6 GHz); this kernel runs on
the Kaggle CPU (4 cores by default, or a GPU if the user chooses one)
and writes the JSON of every size to /kaggle/working.

The kernel clones the repository at a branch given at the top, installs
it, runs ``benchmarks/video_quality.py --size`` for each size in
``SIZES``, and copies the JSON to ``/kaggle/working``. A ``SMOKE``
switch at the top runs ``--smoke`` first (a few frames per stream), so
the user can check the kernel end-to-end in two minutes before paying
for the full run.

Command-line flags: none. Edit ``REPO_BRANCH`` and ``SMOKE`` at the top.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---- configuration (edit here) ----
REPO_URL = "https://github.com/tatopenn-cell/Dense-Armor.git"
REPO_BRANCH = "taskZ2-close"
WORK_DIR = Path("/kaggle/working")
CLONE_DIR = WORK_DIR / "Dense-Armor"
SIZES = ("1280x720", "1920x1080")
SMOKE = False
# ------------------------------------


def run(cmd, **kwargs):
    print(f"$ {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    dt = time.perf_counter() - t0
    if r.stdout:
        print(r.stdout[-3000:], flush=True)
    if r.returncode != 0:
        print(f"  exit={r.returncode} after {dt:.1f} s", flush=True)
        if r.stderr:
            print(r.stderr[-1500:], flush=True)
        raise SystemExit(f"command failed: {cmd}")
    print(f"  exit=0 after {dt:.1f} s", flush=True)
    return dt


def machine():
    import platform
    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    return {
        "cpu": cpu,
        "cores": os.cpu_count(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def main():
    print("=== machine ===", flush=True)
    print(json.dumps(machine(), indent=2), flush=True)
    print(f"branch: {REPO_BRANCH}", flush=True)
    print(f"sizes: {SIZES}", flush=True)
    print(f"smoke: {SMOKE}", flush=True)

    if CLONE_DIR.exists():
        shutil.rmtree(CLONE_DIR)
    run(["git", "clone", "--depth", "1", "-b", REPO_BRANCH, REPO_URL, str(CLONE_DIR)])
    run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[vision]"],
        cwd=str(CLONE_DIR))

    results = []
    for size in SIZES:
        cmd = [sys.executable, "-u", "benchmarks/video_quality.py",
               "--size", size]
        if SMOKE:
            cmd.append("--smoke")
        dt = run(cmd, cwd=str(CLONE_DIR))
        w, h = size.split("x")
        produced = CLONE_DIR / f"benchmarks/video_quality_{w}x{h}.json"
        if not produced.exists():
            print(f"  WARN: {produced} non creato", flush=True)
            continue
        dst = WORK_DIR / f"video_quality_{w}x{h}.json"
        shutil.copy(produced, dst)
        data = json.loads(dst.read_text())
        results.append({"size": size, "wall_clock_s": round(dt, 1),
                        "json": str(dst), "n_frames": data.get("n_frames"),
                        "runs": {k: v for k, v in data.get("runs", {}).items()}})

    summary = {
        "machine": machine(),
        "branch": REPO_BRANCH,
        "smoke": SMOKE,
        "sizes": results,
    }
    (WORK_DIR / "video_quality_summary.json").write_text(
        json.dumps(summary, indent=2))
    print("\n=== summary ===", flush=True)
    for r in results:
        print(f"  {r['size']}: {r['wall_clock_s']} s, "
              f"{r['n_frames']} frames, json={Path(r['json']).name}",
              flush=True)
    print(f"summary: {WORK_DIR / 'video_quality_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
