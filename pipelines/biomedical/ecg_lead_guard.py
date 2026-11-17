"""ECG lead guard: loose-lead shield and anomaly score on a real ECG stream.

Runs the five minutes of MIT-BIH ECG of scipy (108,000 samples at 360 Hz),
injects three loose-lead faults, blocks them with a step shield, and scores
every other sample with Half-Space Trees on a cycle residual and two
spectral bands. Demonstration only: it is not a medical device and the
alerts have no clinical label.

Install: pip install "dense-armor[biomedical]"
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.datasets import electrocardiogram

from dense_armor.core.signals.stats.sdft import SDFT
from dense_armor.core.online.timeseries.cycle import CycleResidual
from dense_armor.core.online.anomaly.hst import HalfSpaceTrees

# 1. Load 5 minutes of real clinical MIT-BIH ECG data (108,000 samples @ 360 Hz)
ecg = electrocardiogram().copy()
fs = 360.0
n_samples = len(ecg)
t = np.arange(n_samples) / fs

# Inject 3 fake loose-lead hardware faults along the 5 minutes (at 38s, 120s, 240s)
for fault_t, val_spike in [(38.0, 4.8), (120.0, -4.6), (240.0, 4.9)]:
    idx = int(fault_t * fs)
    ecg[idx:idx + 4] = val_spike

# 2. Set up the shield, the wave reader, and the unsupervised brain
max_period = 500
step_limit_mv = 2.0
fault_run_max = 10
range_init = 1000
hst_window = 2000
hst_thr_warmup = 1500
template_start = 2 * max_period
score_start = template_start + range_init + hst_window
alert_start = score_start + hst_thr_warmup

cycle_res = CycleResidual(max_period=max_period, search=40)
sdft = SDFT(90)
hst = HalfSpaceTrees(
    n_trees=25,
    depth=8,
    window=hst_window,
    range_init=range_init,
    threshold_quantile=0.992,
    threshold_window=4000,
    threshold_warmup=hst_thr_warmup,
    seed=42
)

sensor_fault_flags = np.zeros(n_samples, dtype=int)
anomaly_scores = np.zeros(n_samples, dtype=float)
hst_alerts = np.zeros(n_samples, dtype=int)
last_clean_v = 0.0
fault_run = 0

t0 = time.perf_counter()

for i in range(n_samples):
    v = float(ecg[i])

    # STAGE 1: Shield: flags a jump above step_limit_mv from the last clean sample (real ECG steps stay below 0.65 mV)
    fault_candidate = abs(v - last_clean_v) > step_limit_mv
    fault_run = fault_run + 1 if fault_candidate else 0
    is_sensor_fault = fault_candidate and fault_run <= fault_run_max
    if is_sensor_fault:
        sensor_fault_flags[i] = 1
        clean_v = last_clean_v
    else:
        clean_v = v
        last_clean_v = v

    res_raw = float(cycle_res.transform_one({"ecg": clean_v})["residual"])
    template_ready = bool(np.isfinite(res_raw))
    res_val = abs(res_raw) if template_ready else 0.0

    # STAGE 2: Sliding DFT spectral extraction on the clean signal
    sdft.update(clean_v)
    spec = np.abs(sdft.spectrum())

    features = {
        "baseline": float(np.mean(spec[0:2])),   # bins 0-1: 0 and 4 Hz (4 Hz per bin)
        "qrs": float(np.mean(spec[3:9])),        # bins 3-8: 12-32 Hz
        "cycle_jump": res_val if not is_sensor_fault else 0.0
    }

    if not template_ready:
        continue

    # STAGE 3: Unsupervised anomaly detection (HalfSpaceTrees)
    score = float(hst.score_one(features))
    anomaly_scores[i] = score

    if not is_sensor_fault:
        is_outlier = bool(hst.is_outlier(features))
        hst.learn_one(features)
        if i >= alert_start and is_outlier:
            hst_alerts[i] = 1

elapsed_s = time.perf_counter() - t0
print(f"Processed {n_samples} real samples in {elapsed_s:.2f} s ({(elapsed_s * 1e6) / n_samples:.1f} us/sample)")
print(f"Estimated period: {cycle_res.period} samples")
print(f"Loose-lead faults blocked:   {sensor_fault_flags.sum()} samples")
print(f"HST alerts (no clinical label): {hst_alerts.sum()} samples ({hst_alerts.mean()*100:.2f}%)")
print(f"Alerts active from sample {alert_start} ({alert_start / fs:.1f} s); HST n_missing: {hst.n_missing}")
alert_idx = np.where(hst_alerts == 1)[0]
alert_t0 = alert_idx[np.r_[True, np.diff(alert_idx) > 1]] / fs if len(alert_idx) else np.array([])
print(f"Alert episodes: {len(alert_t0)}; starting between 35 s and 50 s: {np.round(alert_t0[(alert_t0 >= 35.0) & (alert_t0 <= 50.0)], 2)}")

# 3. Plot
fig, axs = plt.subplots(3, 1, figsize=(14, 9), sharex=False)

axs.item(0).plot(t, ecg, color="black", linewidth=0.5, label="Full Real ECG (300s)")
axs.item(0).fill_between(t, -5, 5, where=(hst_alerts == 1), color="red", alpha=0.4, label="HST alerts (no clinical label)")
axs.item(0).set_ylabel("mV")
axs.item(0).set_title(f"5-Minute Overview ({n_samples:,} samples): ECG Monitoring")
axs.item(0).legend(loc="upper right")

axs.item(1).plot(t[score_start:], anomaly_scores[score_start:], color="purple", linewidth=0.7, label=f"Anomaly Score (HalfSpaceTrees window={hst_window})")
axs.item(1).set_ylabel("Score")
axs.item(1).set_xlabel("Global Time (seconds)")
axs.item(1).legend(loc="upper right")

mask_zoom = (t >= 35.0) & (t <= 50.0)
t_z = t[mask_zoom]
axs.item(2).plot(t_z, ecg[mask_zoom], color="black", linewidth=1.0, label="ECG (Zoom 35s-50s)")
axs.item(2).fill_between(t_z, -2, 5, where=(sensor_fault_flags[mask_zoom] == 1), color="gray", alpha=0.6, label="Blocked: injected loose-lead fault (38s)")
axs.item(2).fill_between(t_z, -2, 5, where=(hst_alerts[mask_zoom] == 1), color="red", alpha=0.5, label="HST alert")
axs.item(2).set_ylabel("mV")
axs.item(2).set_xlabel("Zoom Time (seconds)")
axs.item(2).set_title("Detail (35s - 50s): Hardware Fault Isolation vs HST Alerts")
axs.item(2).legend(loc="upper right")

plt.tight_layout()
out_path = os.path.abspath("ecg_lead_guard.png")
plt.savefig(out_path, dpi=150)
print(f"Saved PNG to: {out_path}")

