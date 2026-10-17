"""Synthetic 2-D lidar next to a robot arm: false alarms and detection delay.

There is no lidar recording in the CASPER dataset, so this benchmark is a SIMULATION and
every number it prints belongs to the simulated sensor, not to a real robot. The scene is a
6 m by 4 m room with two pillars and a robot arm that sweeps one beam sector with a period
of 38 s (the CASPER cycle); the sensor has 180 beams at 10 Hz, range noise of 1 cm and 0.2%
beam dropouts. Four faults start 30 minutes into a 60 minute stream: an obstacle enters the
room, the range drifts by 1 cm per minute, a sector of 20 beams is covered by dirt, and 30%
of the beams drop out in bursts of 10 s.

Two detectors of the library run on every scan with fixed settings, none tuned on this data:
robust Mahalanobis distance of a random projection of the scan to 8 numbers (alarm after 5
scans in a row above the threshold 7.0), and ADWIN on the change of the mean range against one
cycle earlier. The script prints false alarms per hour on faultless time, the detection delay
of every fault over the seeds, and the time per scan.

Command, from the repository root:

    python benchmarks/lidar_sim.py --seeds 5

    python benchmarks/lidar_sim.py --smoke
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from dense_armor.utility.anomaly.mahalanobis import OnlineRobustMahalanobis
from dense_armor.utility.drift.adwin import ADWIN
from dense_armor.utility.vision.reduce import RandomProjection

BEAMS = 180
HZ = 10.0
MAX_R = 10.0
CYCLE_S = 38.0
NOISE = 0.01
DROPOUT = 0.002
ARM_LEN = 0.9
ROOM = (3.0, 2.0)
PILLARS = ((-1.8, 1.2, 0.3), (1.5, -1.1, 0.3))
ANGLES = np.deg2rad(np.arange(BEAMS) * 360.0 / BEAMS)
COS, SIN = np.cos(ANGLES), np.sin(ANGLES)
K = 8
THRESHOLD = 7.0
CONSECUTIVE = 5
WARMUP_S = 600.0
SHIFT = int(CYCLE_S * HZ)
FAULTS = ("obstacle", "drift", "dirt", "dropout")


def room_ranges():
    with np.errstate(divide="ignore"):
        tx = np.where(np.abs(COS) > 1e-9, ROOM[0] / np.abs(COS), np.inf)
        ty = np.where(np.abs(SIN) > 1e-9, ROOM[1] / np.abs(SIN), np.inf)
    return np.minimum(tx, ty)


def circle_ranges(cx, cy, radius):
    t = cx * COS + cy * SIN
    h = np.abs(cx * SIN - cy * COS)
    hit = (t > 0.0) & (h < radius)
    out = np.full(BEAMS, np.inf)
    out[hit] = t[hit] - np.sqrt(radius**2 - h[hit] ** 2)
    return out


BASE = room_ranges()
for _cx, _cy, _r in PILLARS:
    BASE = np.minimum(BASE, circle_ranges(_cx, _cy, _r))
OBSTACLE = circle_ranges(2.0, 1.2, 0.25)


def scan(step, rng, fault=None, onset=None):
    t = step / HZ
    r = BASE.copy()
    phi = 1.0 * np.sin(2.0 * np.pi * t / CYCLE_S)
    d = np.angle(np.exp(1j * (ANGLES - phi)))
    r[np.abs(d) < 0.03] = np.minimum(r[np.abs(d) < 0.03], ARM_LEN)
    active = fault is not None and step >= onset
    if active and fault == "obstacle":
        r = np.minimum(r, OBSTACLE)
    r = r + rng.normal(0.0, NOISE, BEAMS)
    if active and fault == "drift":
        r = r + 0.01 * (step - onset) / (HZ * 60.0)
    if active and fault == "dirt":
        r[20:40] = 0.3
    drop = rng.random(BEAMS) < DROPOUT
    if active and fault == "dropout" and int((step - onset) / (HZ * 10.0)) % 2 == 0:
        drop |= rng.random(BEAMS) < 0.3
    r[drop] = MAX_R
    return np.clip(r, 0.05, MAX_R)


class Pipeline:
    def __init__(self, seed):
        self.rp = RandomProjection(k=K, seed=seed, input_dim=BEAMS)
        self.keys = [f"p{j}" for j in range(K)]
        self.maha = OnlineRobustMahalanobis(feature_keys=self.keys, threshold=THRESHOLD, seed=seed)
        self.adwin = ADWIN()
        self.run = 0
        self.mean_hist = []
        self.maha_alarm = False
        self.adwin_alarm = False

    def step(self, r):
        p = self.rp.transform_one({f"r{i}": float(v) for i, v in enumerate(r)})
        score = self.maha.score_one(p)
        self.maha.learn_one(p)
        self.run = self.run + 1 if score > THRESHOLD else 0
        maha_alarm = self.run == CONSECUTIVE
        m = float(np.mean(r))
        self.mean_hist.append(m)
        adwin_alarm = False
        if len(self.mean_hist) > SHIFT:
            self.adwin.update(abs(m - self.mean_hist[-1 - SHIFT]))
            adwin_alarm = bool(self.adwin.drift_detected)
        return maha_alarm, adwin_alarm


def run_stream(seed, minutes, fault=None, onset_min=30.0):
    rng = np.random.default_rng(seed)
    pipe = Pipeline(seed)
    n = int(minutes * 60 * HZ)
    onset = int(onset_min * 60 * HZ) if fault else None
    alarms = {"mahalanobis": [], "adwin": []}
    t0 = time.perf_counter()
    for step in range(n):
        a, b = pipe.step(scan(step, rng, fault, onset))
        if a:
            alarms["mahalanobis"].append(step)
        if b:
            alarms["adwin"].append(step)
    return alarms, onset, (time.perf_counter() - t0) / n * 1e6


def summarise(alarms, onset, minutes):
    warm = int(WARMUP_S * HZ)
    end = onset if onset is not None else int(minutes * 60 * HZ)
    out = {}
    for name, steps in alarms.items():
        quiet = [s for s in steps if warm <= s < end]
        out[name] = {"quiet_alarms": len(quiet), "quiet_hours": (end - warm) / HZ / 3600.0}
        if onset is not None:
            after = [s for s in steps if s >= onset]
            out[name]["delay_s"] = (after[0] - onset) / HZ if after else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--normal-hours", type=float, default=4.0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="benchmarks/lidar_sim.json")
    a = ap.parse_args()
    seeds = 1 if a.smoke else a.seeds
    normal_min = 12.0 if a.smoke else a.normal_hours * 60.0
    stream_min = 14.0 if a.smoke else 60.0
    onset_min = 12.0 if a.smoke else 30.0
    res = {"config": {"beams": BEAMS, "hz": HZ, "noise_m": NOISE, "dropout": DROPOUT, "k": K,
                      "threshold": THRESHOLD, "consecutive": CONSECUTIVE, "seeds": seeds, "smoke": a.smoke,
                      "simulated": True}}
    alarms, _, us = run_stream(0, normal_min)
    res["faultless"] = summarise(alarms, None, normal_min)
    res["microseconds_per_scan_median_stream"] = us
    print(f"faultless stream of {normal_min / 60:.2f} h, {us:.0f} us per scan (both detectors and the generator)")
    for name, v in res["faultless"].items():
        print(f"  {name:12s} false alarms {v['quiet_alarms']} in {v['quiet_hours']:.2f} h")
    res["faults"] = {}
    for fault in FAULTS:
        per = {"mahalanobis": [], "adwin": [], "quiet_alarms": {"mahalanobis": 0, "adwin": 0}, "quiet_hours": 0.0}
        for seed in range(seeds):
            alarms, onset, _ = run_stream(100 + seed, stream_min, fault, onset_min)
            s = summarise(alarms, onset, stream_min)
            for name in ("mahalanobis", "adwin"):
                per[name].append(s[name]["delay_s"])
                per["quiet_alarms"][name] += s[name]["quiet_alarms"]
            per["quiet_hours"] += s["mahalanobis"]["quiet_hours"]
        res["faults"][fault] = per
        for name in ("mahalanobis", "adwin"):
            d = [x for x in per[name] if x is not None]
            med = float(np.median(d)) if d else None
            print(f"{fault:9s} {name:12s} detected {len(d)}/{seeds}  median delay {med} s  "
                  f"false alarms before onset {per['quiet_alarms'][name]} in {per['quiet_hours']:.2f} h")
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
