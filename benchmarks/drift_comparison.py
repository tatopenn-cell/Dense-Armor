"""Compare four drift detectors on the CASPER stream.

Run from the repository root. Requires the Kaggle dataset used by
``casper_benchmark.py``. Detector parameters are the module defaults,
the same for both signals (no tuning on the test change). The CUSUM
fixed reference uses the same span as ``casper_benchmark.run_cusum``
(radius=T//10, ref_mult=10, one-sided as ``run_cusum``). Two signals,
both at the native 20 Hz: the squared shift residual e² and
``trailing_mean(e², 2T)``, the series ``run_cusum`` is tuned on.
"""

import time

import numpy as np
from casper_benchmark import (
    CHANGE_WINDOW,
    FS,
    T,
    find_dataset,
    load_slice,
    parse_joint,
    shift_residual_fixed,
    trailing_mean,
)

from dense_armor.utility.drift.adwin import ADWIN
from dense_armor.utility.drift.detector import CUSUMDriftDetector
from dense_armor.utility.drift.kswin import KSWIN
from dense_armor.utility.drift.page_hinkley import PageHinkley
from dense_armor.utility.evaluate import evaluate_events


def _stream_for_comparison():
    import pandas as pd
    _, arm, _ = find_dataset()
    head = pd.read_csv(arm, nrows=1, usecols=["Timestamp"])
    t0 = float(head["Timestamp"].iloc[0])
    ta, tb = CHANGE_WINDOW
    df = load_slice(
        arm, t0 + ta * 3600, t0 + tb * 3600,
        ["Actual Joint Velocities", "Anomaly State"], extra_T=True,
    )
    v_load = parse_joint(df, "Actual Joint Velocities", 5)
    lbl_load = df["Anomaly State"].to_numpy()
    lbl_win = lbl_load[T:]
    e2 = shift_residual_fixed(v_load, T) ** 2
    change_idx = int(np.argmax(lbl_win == 1))
    n_anom = int((lbl_win == 1).sum())
    t_start = ta * 3600.0
    dt = 1.0 / FS
    stream = [(t_start + i * dt, float(e2[i])) for i in range(len(e2))]
    event_start = t_start + change_idx * dt
    event_end = event_start + max(1, n_anom) * dt
    print(f"joint 5, change_idx {change_idx}, anomaly samples {n_anom}")
    return stream, [(event_start, event_end)], n_anom


def _latency_us(make_det, values, warmup=50, n=2000):
    det = make_det()
    xs = [float(v) for v in values[:warmup + n]]
    for v in xs[:warmup]:
        det.update(v)
    samples = []
    for v in xs[warmup:]:
        t0 = time.perf_counter_ns()
        det.update(v)
        samples.append(time.perf_counter_ns() - t0)
    arr = np.asarray(samples, dtype=float) / 1000.0
    return float(np.percentile(arr, 50)), float(np.percentile(arr, 99))


def main():
    stream, events, n_anom = _stream_for_comparison()
    normal_time_s = events[0][0] - stream[0][0]
    print(f"stream: {len(stream)} samples, "
          f"event {events[0]}, anomaly samples {n_anom}")
    print(f"normal_time_s = event_start - stream_start = "
          f"{normal_time_s:.1f} s = {normal_time_s / 3600.0:.3f} h")
    print(f"{'detector':<22} {'delay_s':>9} {'missed':>7} "
          f"{'FA/h':>9} {'rangeF1':>9} {'NAB':>8} {'p50us':>9} {'p99us':>9}")

    makers = [
        ("CUSUM fixed",
         lambda: CUSUMDriftDetector(reference="fixed", radius=T // 10,
                                    ref_mult=10, two_sided=False)),
        ("PageHinkley", lambda: PageHinkley()),
        ("ADWIN", lambda: ADWIN()),
        ("KSWIN", lambda: KSWIN(seed=0)),
    ]
    raw = np.asarray([v for _, v in stream], dtype=float)
    ts = [t for t, _ in stream]
    for label, vals in (("e2", raw), ("trailing_mean(e2, 2T)", trailing_mean(raw, 2 * T))):
        print(f"--- signal: {label}")
        s = list(zip(ts, (float(v) for v in vals)))
        for name, make in makers:
            report = evaluate_events(
                s, make(), events, dt=1.0 / FS, normal_time_s=normal_time_s,
            )
            delay = report["mean_delay_s"]
            p50, p99 = _latency_us(make, list(vals))
            print(f"{name:<22} "
                  f"{(delay if delay is not None else float('nan')):>9.3f} "
                  f"{report['missed']:>7d} "
                  f"{report['false_alarms_per_hour']:>9.3f} "
                  f"{report['range_f1']:>9.3f} "
                  f"{report['nab_score']:>8.2f} "
                  f"{p50:>9.2f} {p99:>9.2f}", flush=True)


if __name__ == "__main__":
    main()
