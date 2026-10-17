"""CASPER: can the IMU file be tied to the arm by cross-correlation?

The file nicla_fixed.csv (accelerometer, gyroscope, magnetometer of the Nicla sensor) has no
timestamp column. This script counts the rows of both files and cross-correlates the speed
of the arm joints (right_arm.csv) with the norm of the gyroscope and of the accelerometer,
and with the absolute change of each, over lags of plus or minus 16,000 rows on the first
150,000 rows, and prints the three strongest lags. It reads the Kaggle copy of the dataset.

Command, with the dataset in place (on Kaggle: attached as a data source):

    python benchmarks/casper_imu_alignment.py
"""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

root = next(Path("/kaggle/input").rglob("right_arm.csv")).parent
arm = root / "right_arm.csv"
imu = next(Path("/kaggle/input").rglob("nicla_fixed.csv"))
res = {"imu_columns": list(pd.read_csv(imu, nrows=1).columns)}
res["arm_columns"] = list(pd.read_csv(arm, nrows=1).columns)[:30]

n_imu = 0
for ch in pd.read_csv(imu, usecols=[0], chunksize=500_000):
    n_imu += len(ch)
n_arm, ts_first, ts_last = 0, None, None
for ch in pd.read_csv(arm, usecols=["Timestamp"], chunksize=500_000):
    n_arm += len(ch)
    if ts_first is None:
        ts_first = float(ch["Timestamp"].iloc[0])
    ts_last = float(ch["Timestamp"].iloc[-1])
res.update(rows_imu=n_imu, rows_arm=n_arm, arm_seconds=ts_last - ts_first, arm_hz=n_arm / (ts_last - ts_first),
           imu_hz_if_same_span=n_imu / (ts_last - ts_first))

N = 200_000
a = pd.read_csv(arm, nrows=N, usecols=["Timestamp", "Actual Joint Velocities", "Anomaly State"])
v = np.array([np.linalg.norm(ast.literal_eval(s)) if isinstance(s, str) else np.nan for s in a["Actual Joint Velocities"]])
v = np.nan_to_num(v)
g = pd.read_csv(imu, nrows=N)
gy = np.linalg.norm(g[["GyroX", "GyroY", "GyroZ"]].to_numpy(float), axis=1)
ac = np.linalg.norm(g[["AccX", "AccY", "AccZ"]].to_numpy(float), axis=1)


def smooth(x, w=20):
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[w:] - c[:-w]) / w


sv, sg, sa = smooth(v), smooth(gy), smooth(ac)


def zs(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def xcorr_peaks(x, y, max_lag=16000, n=150000, top=3):
    x, y = zs(x[:n]), zs(y[:n])
    m = len(x) + len(y) - 1
    f = np.fft.rfft(x, 2 * m) * np.conj(np.fft.rfft(y, 2 * m))
    c = np.fft.irfft(f)[: m] 
    c = np.concatenate([c[-(len(y) - 1):], c[: len(x)]]) / min(len(x), len(y))
    lags = np.arange(-(len(y) - 1), len(x))
    sel = (lags >= -max_lag) & (lags <= max_lag)
    cs, ls = c[sel], lags[sel]
    order = np.argsort(-np.abs(cs))[: top * 40]
    picked = []
    for i in order:
        if all(abs(ls[i] - q[0]) > 200 for q in picked):
            picked.append((int(ls[i]), float(cs[i])))
        if len(picked) == top:
            break
    return picked, float(cs[np.where(ls == 0)[0][0]])


dv = np.abs(np.diff(sv))
for name, y in (("gyro_norm", sg), ("acc_norm", sa), ("gyro_abs_diff", np.abs(np.diff(sg))), ("acc_abs_diff", np.abs(np.diff(sa)))):
    x = sv if "diff" not in name else dv
    peaks, c0 = xcorr_peaks(x, y)
    res[name] = {"top_lags_rows_and_corr": peaks, "corr_at_0": c0}
res["first_rows_same_start"] = {"arm_timestamp_first": ts_first}
Path("benchmarks/casper_imu_alignment.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
