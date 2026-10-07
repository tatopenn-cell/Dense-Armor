"""CASPER benchmark, final assembly.

Sources: Kayan et al., CASPER, IEEE PerCom Workshops 2023 (dataset, task).
CUSUM on the squared residual as a variance-change detector: Basseville &
Nikiforov, Detection of Abrupt Changes, Prentice Hall, 1993, Ch. 2.

The arm repeats one cycle of T = 760 samples (38 s at 20 Hz). The cycle
residual e(t) = x(t) - x(t - T) is computed by loading T extra samples
before each slice so the first residual sample is real, and a causal
trailing mean is used wherever the signal is smoothed. Every number in
docs/api/real_robot_benchmark.md is printed here.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from dense_armor.drift.detector import CUSUMDriftDetector
from dense_armor.anomaly.deviation import StreamingDeviationScorer
from dense_armor.anomaly.filters import (
    HampelScorer, TukeyScorer, ChauvenetScorer, SigmaClipScorer,
)
from dense_armor.anomaly.mahalanobis import OnlineRobustMahalanobis


FS = 20.0
T = 760
PERIOD_SEARCH_S = 120.0
CHANGE_HOUR = 15.0
CHANGE_WINDOW = (14.0, 16.0)
FAR_NORMAL_WINDOW = (8.0, 9.0)
CASPER_WINDOW_SAMPLES = 500
CUSUM_RADIUS = T // 10
CUSUM_REF_1C = 10
CUSUM_REF_4C = 40
CUSUM_K_DEFAULT = 0.5
CUSUM_H_DEFAULT = 20.0
SCORER_RADIUS = T // 4
POINT_WINDOW = (14.5, 16.0)
H_SWEEP = (20.0, 50.0, 100.0, 200.0, 400.0, 1000.0)
K_SWEEP = (0.5, 2.0, 5.0)


def find_dataset():
    for base in (
        Path("/kaggle/input/industrial-robotic-arm-anomaly-detection"),
        Path("/root/.cache/kagglehub/datasets/hkayan/"
             "industrial-robotic-arm-anomaly-detection/versions/3"),
    ):
        if base.exists():
            arm = next(base.rglob("right_arm.csv"), None)
            imu = next(base.rglob("nicla_fixed.csv"), None)
            return base, arm, imu
    raise FileNotFoundError("CASPER dataset not found")


def load_slice(path, t_start_abs, t_end_abs, cols, extra_T=False):
    """t_start_abs, t_end_abs are absolute Timestamp values (seconds).

    If extra_T is True, reads T extra samples before t_start_abs so a cycle
    residual computed on the loaded series has a real value at t_start_abs.
    The caller must then drop the first T samples of the returned arrays.
    """
    if extra_T:
        t_start_abs = t_start_abs - (T / FS)
    parts = []
    for chunk in pd.read_csv(path, chunksize=200_000,
                              usecols=["Timestamp"] + cols):
        ts = chunk["Timestamp"].to_numpy()
        m = (ts >= t_start_abs) & (ts <= t_end_abs)
        if m.any():
            parts.append(chunk.loc[m])
        if ts[-1] > t_end_abs:
            break
    if not parts:
        return pd.DataFrame(columns=["Timestamp"] + cols)
    return pd.concat(parts, ignore_index=True)


def parse_joint(df, col, j):
    s = df[col].astype(str).str.strip("[]").str.split(",").str[j]
    return s.astype(float).to_numpy()


def strongest_period(x, fs=FS, min_lag_s=0.5, max_lag_s=PERIOD_SEARCH_S,
                     min_acf=0.3):
    y = x - x.mean()
    n = len(y)
    if n < 4 * int(max_lag_s * fs):
        return None, None
    F = np.fft.rfft(y, n=2 * n)
    acf = np.fft.irfft(F * np.conj(F))[:n]
    acf = acf / (acf[0] + 1e-12)
    lo = int(min_lag_s * fs)
    hi = min(n - 2, int(max_lag_s * fs))
    best_lag, best_acf = None, 0.0
    for i in range(lo + 1, hi):
        if acf[i] > acf[i-1] and acf[i] > acf[i+1] and acf[i] > min_acf:
            if acf[i] > best_acf:
                best_lag, best_acf = i, float(acf[i])
    return best_lag, best_acf


def shift_residual(x, T_=T):
    e = x.copy()
    e[T_:] = x[T_:] - x[:-T_]
    return e


def shift_residual_fixed(x_load, T_=T):
    """Return e = x - x(t - T) for the slice, dropping the first T raw."""
    e = shift_residual(x_load, T_)
    return e[T_:]


def trailing_mean(x, w):
    c = np.concatenate([[0], np.cumsum(x)])
    n = len(x)
    out = np.zeros(n)
    for i in range(n):
        lo = max(0, i - w + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


def run_cusum(series, change_idx, k=CUSUM_K_DEFAULT, h=CUSUM_H_DEFAULT,
              reference="fixed", ref_mult=CUSUM_REF_1C,
              two_sided=False):
    det = CUSUMDriftDetector(reference=reference, radius=CUSUM_RADIUS,
                             ref_mult=ref_mult, k=k, h=h,
                             two_sided=two_sided)
    first_after = None
    fa = 0
    for i, x in enumerate(series):
        det.update(float(x))
        if det.drift_detected:
            if i < change_idx:
                fa += 1
            elif first_after is None:
                first_after = i
    delay = (first_after - change_idx) if first_after is not None else None
    hours = change_idx / FS / 3600.0
    fa_h = fa / hours if hours > 0 else float("nan")
    return delay, fa_h


def point_scores(x, radius):
    scorers = {
        "deviation": StreamingDeviationScorer(radius=radius, ref_mult=3),
        "hampel": HampelScorer(radius=radius, n_sigmas=3.0),
        "tukey": TukeyScorer(radius=radius),
        "chauvenet": ChauvenetScorer(radius=radius),
        "sigmaclip": SigmaClipScorer(radius=radius, n_sigmas=3.0),
    }
    scores = {k: np.zeros(len(x)) for k in scorers}
    t0_t = time.perf_counter()
    for i, v in enumerate(x):
        d = {"v": float(v)}
        for name, s in scorers.items():
            scores[name][i] = s.score_one(d)
            s.learn_one(d)
    dt = time.perf_counter() - t0_t
    return scores, dt / len(x) * 1e6


def predict(scores, name, thr):
    return scores < thr if name == "chauvenet" else scores > thr


def f1_binary(preds, labels):
    tp = int((preds & labels).sum())
    fp = int((preds & ~labels).sum())
    fn = int((~preds & labels).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return f1, p, r


def f1_best(scores, labels, name, n_grid=200):
    if name == "chauvenet":
        grid = np.quantile(scores, np.linspace(0.001, 0.999, n_grid))
    else:
        grid = np.quantile(scores, np.linspace(0.5, 0.9999, n_grid))
    best = (0.0, 0.0, 0.0)
    for thr in grid:
        preds = predict(scores, name, thr)
        f1, p, r = f1_binary(preds, labels)
        if f1 > best[0]:
            best = (f1, p, r)
    return best


def majority(sig, window):
    cs = np.concatenate([[0], np.cumsum(sig.astype(float))])
    n = len(sig)
    half = window // 2
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + half + 1, 0, n)
    counts = cs[hi] - cs[lo]
    sizes = hi - lo
    return counts > sizes / 2.0


def median_template(x, T_):
    n_full = len(x) // T_
    if n_full < 2:
        return None
    return np.median(x[:n_full * T_].reshape(n_full, T_), axis=0)


def template_residual(x, tmpl, phase0):
    T_ = len(tmpl)
    return x - tmpl[(np.arange(len(x)) + phase0) % T_]


def phase_jitter(x_1h, T_, search=40):
    tmpl = median_template(x_1h, T_)
    if tmpl is None:
        return None
    n_cycles = len(x_1h) // T_ - 1
    offsets = []
    for k in range(1, n_cycles):
        seg = x_1h[k * T_:(k + 1) * T_]
        best_d, best_off = np.inf, 0
        for off in range(-search, search + 1):
            lo = max(0, off)
            hi = T_ - max(0, -off)
            ref_lo = max(0, -off)
            ref_hi = ref_lo + (hi - lo)
            d = float(np.mean((seg[lo:hi] - tmpl[ref_lo:ref_hi]) ** 2))
            if d < best_d:
                best_d, best_off = d, off
        offsets.append(best_off)
    return np.array(offsets)


def main():
    base, arm, imu = find_dataset()
    print(f"dataset root: {base}")
    print(f"  right_arm:   {arm}")
    print(f"  nicla_fixed: {imu}")

    head = pd.read_csv(arm, nrows=1, usecols=["Timestamp"])
    t0 = float(head["Timestamp"].iloc[0])
    print(f"  t0 = {t0}  change at hour {CHANGE_HOUR}")

    def hours_to_ts(h):
        return t0 + h * 3600.0

    # ---------- 1. period ----------
    print("\n" + "=" * 70)
    print("1. strongest autocorrelation peak on the first hour of normal data")
    print("=" * 70)
    df_p = load_slice(arm, hours_to_ts(0.0), hours_to_ts(1.0),
                      ["Actual Joint Velocities", "Actual Joint Positions"])
    print(f"  first-hour rows: {len(df_p):,}")
    if len(df_p) == 0:
        raise RuntimeError("first-hour slice is empty; check Timestamp units")
    per_joint = {}
    for j in range(6):
        v = parse_joint(df_p, "Actual Joint Velocities", j)
        lag, acf_v = strongest_period(v)
        per_joint[j] = (lag, acf_v, float(np.var(v)))
        if lag is not None:
            print(f"  joint {j}: T = {lag:>5d} samples ({lag/FS:6.2f} s), "
                  f"acf = {acf_v:.4f}, var = {np.var(v):.5f}")
    valid = [(j, lag, acf_v) for j, (lag, acf_v, _) in per_joint.items()
             if lag is not None]
    if not valid:
        print("  no valid period. abort.")
        return
    j_ref, T_est, acf_T = max(valid, key=lambda t: t[2])
    j_move, _, _ = max(valid, key=lambda t: per_joint[t[0]][2])
    print(f"  adopted T = {T_est} samples ({T_est/FS:.2f} s) from joint {j_ref}")
    print(f"  moving joint (highest variance among valid): {j_move}")

    v_1h = parse_joint(df_p, "Actual Joint Velocities", j_move)
    tmpl_1h = median_template(v_1h, T)

    # ---------- 2. three-window RMS (fixed residual) ----------
    print("\n" + "=" * 70)
    print("2. RMS raw / shift / template in three windows (fixed residual)")
    print("=" * 70)
    windows = [
        ("normal [8, 9]", FAR_NORMAL_WINDOW),
        ("pre-change [14, 15]", (CHANGE_HOUR - 1.0, CHANGE_HOUR)),
        ("post-change [15, 16]", (CHANGE_HOUR, CHANGE_HOUR + 1.0)),
    ]
    for wname, (ta, tb) in windows:
        df_w = load_slice(arm, hours_to_ts(ta), hours_to_ts(tb),
                          ["Actual Joint Velocities"], extra_T=True)
        v_load = parse_joint(df_w, "Actual Joint Velocities", j_move)
        v_win = v_load[T:]
        sh_win = shift_residual_fixed(v_load, T)
        phase0 = int(ta * 3600 * FS)
        tm_win = template_residual(v_win, tmpl_1h, phase0)
        print(f"  {wname:<22} rows={len(v_win):>7,}  "
              f"RMS raw={rms(v_win):.4f}  shift={rms(sh_win):.4f}  "
              f"template={rms(tm_win):.4f}")

    # ---------- 3. change window ----------
    print("\n" + "=" * 70)
    print("3. change window [14, 16]")
    print("=" * 70)
    df_w = load_slice(arm, hours_to_ts(CHANGE_WINDOW[0]),
                      hours_to_ts(CHANGE_WINDOW[1]),
                      ["Actual Joint Velocities", "Anomaly State"],
                      extra_T=True)
    v_load = parse_joint(df_w, "Actual Joint Velocities", j_move)
    lbl_load = df_w["Anomaly State"].to_numpy()
    v_win = v_load[T:]
    lbl_win = lbl_load[T:]
    sh_win = shift_residual_fixed(v_load, T)
    phase0_w = int(CHANGE_WINDOW[0] * 3600 * FS)
    tm_win = template_residual(v_win, tmpl_1h, phase0_w)
    e2_win = sh_win ** 2
    abs_win = np.abs(sh_win)
    tr_T = trailing_mean(e2_win, T)
    tr_2T = trailing_mean(e2_win, 2 * T)

    change_idx = int(np.argmax(lbl_win == 1))
    print(f"  rows={len(v_win):,}  change_idx={change_idx} "
          f"({change_idx/FS/3600:.3f} h into window)")
    print(f"  fraction positive labels: {lbl_win.mean():.3f}")
    print(f"  RMS raw={rms(v_win):.4f}  shift={rms(sh_win):.4f}  "
          f"template={rms(tm_win):.4f}")
    pre_n = change_idx
    post_n = min(len(v_win), change_idx + 72000)
    print(f"  RMS shift  pre={rms(sh_win[:pre_n]):.4f}  "
          f"post={rms(sh_win[change_idx:post_n]):.4f}  "
          f"ratio={rms(sh_win[change_idx:post_n])/(rms(sh_win[:pre_n])+1e-12):.2f}")

    # ---------- 4. CUSUM baseline ----------
    print("\n" + "=" * 70)
    print("4. CUSUM one-sided, reference span 1 cycle (radius=T/10, ref_mult=10)")
    print("=" * 70)
    print(f"  {'signal':<28} {'ref':<10} {'delay':>8} {'delay_s':>9} {'FA/h':>10}")
    for name, s in (("raw", v_win),
                    ("shift residual", sh_win),
                    ("template residual", tm_win),
                    ("shift^2 (energy)", e2_win),
                    ("abs(shift)", abs_win),
                    ("trailing mean(e^2, T)", tr_T),
                    ("trailing mean(e^2, 2T)", tr_2T)):
        for ref_mode in ("fixed", "adaptive"):
            d, fa = run_cusum(s, change_idx, reference=ref_mode)
            ds = f"{d/FS:.2f}" if d is not None else "-"
            print(f"  {name:<28} {ref_mode:<10} {str(d):>8} {ds:>9} {fa:>10.1f}")

    # ---------- 5. h x k sweep ----------
    print("\n" + "=" * 70)
    print("5. h x k sweep on trailing mean(e^2, 2T), span 1c")
    print("=" * 70)
    best_zero = None
    print(f"  {'h':>6} {'k':>4} {'ref':<10} {'delay':>8} {'delay_s':>9} {'FA/h':>10}")
    for ref_mode in ("fixed", "adaptive"):
        for k in K_SWEEP:
            for h in H_SWEEP:
                d, fa = run_cusum(tr_2T, change_idx, k=k, h=h,
                                  reference=ref_mode)
                ds = f"{d/FS:.2f}" if d is not None else "-"
                print(f"  {h:>6.0f} {k:>4.1f} {ref_mode:<10} {str(d):>8} "
                      f"{ds:>9} {fa:>10.1f}")
                if fa == 0.0 and d is not None:
                    if best_zero is None or d < best_zero[0]:
                        best_zero = (d, h, k, ref_mode)
    if best_zero is None:
        print("\n  No (h,k,ref) with zero FA and a detection in the sweep.")
    else:
        d, h, k, ref_mode = best_zero
        print(f"\n  Best zero-FA config: h={h}, k={k}, ref={ref_mode}, "
              f"delay={d} samples ({d/FS:.2f} s)")

    # ---------- 6. point scorers ----------
    print("\n" + "=" * 70)
    print(f"6. point scorers on [14.5, 16], radius={SCORER_RADIUS} "
          f"(window 2r={2*SCORER_RADIUS})")
    print("=" * 70)
    df_pw = load_slice(arm, hours_to_ts(POINT_WINDOW[0]),
                       hours_to_ts(POINT_WINDOW[1]),
                       ["Actual Joint Velocities", "Anomaly State"],
                       extra_T=True)
    v_p_load = parse_joint(df_pw, "Actual Joint Velocities", j_move)
    lbl_p_load = df_pw["Anomaly State"].to_numpy()
    v_p = v_p_load[T:]
    lbl_p = lbl_p_load[T:]
    sh_p = shift_residual_fixed(v_p_load, T)
    e2_p = sh_p ** 2
    tr2T_p = trailing_mean(e2_p, 2 * T)
    y_pt = lbl_p.astype(bool)
    y_win = majority(y_pt, CASPER_WINDOW_SAMPLES)
    base_pt = f1_binary(np.ones_like(y_pt), y_pt)[0]
    base_win = f1_binary(np.ones_like(y_win), y_win)[0]
    print(f"  window rows: {len(v_p):,}  positives: {y_pt.mean():.3f}")
    print(f"  trivial F1 (all anomalous): pt = {base_pt:.3f}  "
          f"win = {base_win:.3f}")

    thr = {"deviation": 3.0, "hampel": 3.0, "tukey": 0.0,
           "chauvenet": 0.5, "sigmaclip": 3.0}

    for tag, s in (("shift residual", sh_p),
                   ("shift energy  e^2", e2_p),
                   ("trailing mean(e^2, 2T)", tr2T_p)):
        scores, us = point_scores(s, SCORER_RADIUS)
        print(f"\n  {tag}  (us/sample = {us:.1f})")
        print(f"    {'estimator':<12} {'F1 pt':>8} {'P pt':>8} {'R pt':>8} "
              f"{'F1 best':>9} {'F1 win':>8}")
        for name in scores:
            preds = predict(scores[name], name, thr[name])
            f1f, pf, rf = f1_binary(preds, y_pt)
            f1b, _, _ = f1_best(scores[name], y_pt, name)
            pw = majority(preds, CASPER_WINDOW_SAMPLES)
            f1w, _, _ = f1_binary(pw, y_win)
            print(f"    {name:<12} {f1f:>8.3f} {pf:>8.3f} {rf:>8.3f} "
                  f"{f1b:>9.3f} {f1w:>8.3f}")

    # ---------- 7. phase jitter ----------
    print("\n" + "=" * 70)
    print("7. phase jitter over the first hour")
    print("=" * 70)
    offs = phase_jitter(v_1h, T)
    if offs is not None and len(offs):
        print(f"  cycles measured: {len(offs)}")
        print(f"  offsets: mean={offs.mean():.2f}  std={offs.std():.2f}  "
              f"min={offs.min()}  max={offs.max()}  "
              f"frac_zero={float(np.mean(offs == 0)):.3f}")
    else:
        print("  could not measure")

    # ---------- 8. IMU ----------
    print("\n" + "=" * 70)
    print("8. OnlineRobustMahalanobis on IMU (acc + gyro, 6 channels)")
    print("=" * 70)
    if imu is not None:
        try:
            head_i = pd.read_csv(imu, nrows=1)
            print(f"  columns: {list(head_i.columns)}")
            cols6 = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"]
            df_i = pd.read_csv(imu, usecols=cols6)
            X = df_i[cols6].to_numpy(dtype=float)
            print(f"  rows: {len(X):,}")
            print("  no timestamp column: cannot align to arm label")
            m = OnlineRobustMahalanobis(feature_keys=cols6, threshold=7.0)
            scores = np.empty(len(X))
            t0_t = time.perf_counter()
            for i in range(len(X)):
                d = {c: float(X[i, j]) for j, c in enumerate(cols6)}
                scores[i] = m.score_one(d)
                m.learn_one(d)
            dt = time.perf_counter() - t0_t
            print(f"  per-sample: {dt/len(X)*1e6:.1f} us")
            print(f"  median={np.median(scores):.3f}  "
                  f"p99={np.quantile(scores, 0.99):.3f}  "
                  f"max={scores.max():.3f}")
        except Exception as e:
            print(f"  ERROR: {e}")
    else:
        print("  no nicla_fixed.csv")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
