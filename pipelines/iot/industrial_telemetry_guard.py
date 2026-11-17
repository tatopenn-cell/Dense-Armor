import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from ucimlrepo import fetch_ucirepo

from dense_armor.core.online.anomaly.filters import HampelScorer
from dense_armor.core.online.anomaly.hst import HalfSpaceTrees
from dense_armor.core.online.drift.page_hinkley import PageHinkley

dataset = fetch_ucirepo(id=601)
df_X = dataset.data.features
df_y = dataset.data.targets

sensor_cols = ["Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear"]
X_base = df_X[sensor_cols].to_numpy(dtype=float)
y_all = df_y["Machine failure"].to_numpy(dtype=int)

X_healthy = X_base[y_all == 0]
X_fault = X_base[y_all == 1]

rng = np.random.default_rng(42)
n_h, n_f = 650, min(350, len(X_fault))
idx_h = rng.choice(len(X_healthy), size=n_h, replace=False)
idx_f = rng.choice(len(X_fault), size=n_f, replace=False)
hst_window = 100
hst_thr_warmup = 100
warmup = hst_window + hst_thr_warmup
feature_keys = ["delta_temp", "overstrain", "power", "rpm", "torque", "wear"]
feature_ranges = [(-4.0, 6.0)] * len(feature_keys)

stream_raw = np.vstack([X_healthy[idx_h], X_fault[idx_f]])

mu0 = stream_raw[:150].mean(axis=0)
std0 = stream_raw[:150].std(axis=0) + 1e-8
stream_norm = (stream_raw - mu0) / std0
n_samples = len(stream_norm)

stream_norm[300, 3] = 9.5

analog_sensors = [(0, "air_temp"), (1, "proc_temp"), (3, "torque")]
edge_guards = {
    name: HampelScorer(radius=10, n_sigmas=7.5, feature=name)
    for _, name in analog_sensors
}

hst = HalfSpaceTrees(
    n_trees=30,
    depth=6,
    window=hst_window,
    feature_keys=feature_keys,
    feature_ranges=feature_ranges,
    threshold_quantile=0.82,
    threshold_window=500,
    threshold_warmup=hst_thr_warmup,
    seed=42,
)
drift_monitor = PageHinkley(delta=0.05, threshold=20.0, two_sided=False, warmup=150)

sensor_spikes = np.zeros(n_samples, dtype=int)
anomaly_scores = np.zeros(n_samples, dtype=float)
alerts = np.zeros(n_samples, dtype=int)
overstrain_history = np.zeros(n_samples, dtype=float)
drift_detected_at = None
false_drifts = 0
spike_run = 0
spike_run_max = 3

rpm_to_rad_s = 2.0 * np.pi / 60.0
p_ref = stream_raw[:150, 3] * (stream_raw[:150, 2] * rpm_to_rad_s)
o_ref = stream_raw[:150, 4] * stream_raw[:150, 3]
d_ref = stream_raw[:150, 1] - stream_raw[:150, 0]
mu_inv = np.array([p_ref.mean(), o_ref.mean(), d_ref.mean()])
std_inv = np.array([p_ref.std(), o_ref.std(), d_ref.std()]) + 1e-8

t0 = time.perf_counter()

for i in range(n_samples):
    raw_row = stream_norm[i]

    spike_detected = False
    for idx, name in analog_sensors:
        val = float(raw_row[idx])
        if i > 50 and bool(edge_guards[name].is_outlier({name: val})):
            spike_detected = True
            break

    spike_run = spike_run + 1 if spike_detected else 0

    if spike_detected and spike_run <= spike_run_max:
        sensor_spikes[i] = 1
        anomaly_scores[i] = anomaly_scores[i - 1] if i > 0 else 0.0
        overstrain_history[i] = overstrain_history[i - 1] if i > 0 else 0.0
        continue

    for idx, name in analog_sensors:
        edge_guards[name].learn_one({name: float(raw_row[idx])})

    phys_row = raw_row * std0 + mu0
    air_t, proc_t, rpm_phys, torque_phys, wear_phys = phys_row

    power_w = torque_phys * (rpm_phys * rpm_to_rad_s)
    overstrain_min_nm = wear_phys * torque_phys
    delta_t_k = proc_t - air_t

    power_z = float((power_w - mu_inv[0]) / std_inv[0])
    overstrain_z = float((overstrain_min_nm - mu_inv[1]) / std_inv[1])
    delta_t_z = float((delta_t_k - mu_inv[2]) / std_inv[2])
    overstrain_history[i] = overstrain_z

    sample = {
        "rpm": float(raw_row[2]),
        "torque": float(raw_row[3]),
        "wear": float(raw_row[4]),
        "power": power_z,
        "overstrain": overstrain_z,
        "delta_temp": delta_t_z,
    }

    score = float(hst.score_one(sample))
    anomaly_scores[i] = score

    is_outlier = (i >= warmup) and bool(hst.is_outlier(sample))
    if is_outlier:
        alerts[i] = 1
    hst.learn_one(sample)

    if i >= hst_window:
        drift_monitor.update(score)
        if drift_monitor.drift_detected:
            if i <= n_h:
                false_drifts += 1
            elif drift_detected_at is None:
                drift_detected_at = i

elapsed_ms = (time.perf_counter() - t0) * 1000
healthy_fp = alerts[warmup:n_h].mean() * 100.0
fault_recall = alerts[n_h:].mean() * 100.0
total_spikes = int(sensor_spikes.sum())

auc_mask = (np.arange(n_samples) >= warmup) & (sensor_spikes == 0)
auc_pos = anomaly_scores[auc_mask & (np.arange(n_samples) >= n_h)]
auc_neg = anomaly_scores[auc_mask & (np.arange(n_samples) < n_h)]
auc = float(
    ((auc_pos[:, None] > auc_neg[None, :]).sum() + 0.5 * (auc_pos[:, None] == auc_neg[None, :]).sum())
    / (len(auc_pos) * len(auc_neg))
)
lag_str = f"+{drift_detected_at - n_h} lag" if drift_detected_at is not None else "Not detected"

print(f"Processed {n_samples} UCI AI4I 2020 samples in {elapsed_ms:.2f} ms ({(elapsed_ms*1000)/n_samples:.1f} us/sample)")
print(f"Injected spike at step 300 isolated:          {bool(sensor_spikes[300] == 1)} (total isolated: {total_spikes})")
print(f"Regime shift flagged at step:                 {drift_detected_at} (onset step: {n_h})")
print(f"Drift triggers before onset (false):  {false_drifts}")
print(f"False alarm rate on healthy window:           {healthy_fp:.2f}%")
print(f"Mechanical failure detection rate:            {fault_recall:.1f}%")
print(f"ROC AUC of HST score (healthy vs failure):    {auc:.3f} (n_pos={len(auc_pos)}, n_neg={len(auc_neg)})")
tp = int(alerts[n_h:].sum())
fp = int(alerts[warmup:n_h].sum())
precision = tp / max(tp + fp, 1)
stream_prev = (n_samples - n_h) / (n_samples - warmup)
tpr = fault_recall / 100.0
fpr = healthy_fp / 100.0
real_prev = float(y_all.mean())
real_precision = tpr * real_prev / max(tpr * real_prev + fpr * (1.0 - real_prev), 1e-12)
print(f"Precision on this stream (failure share {stream_prev*100:.1f}%): {precision*100:.1f}%")
print(f"Expected precision at dataset prevalence ({real_prev*100:.1f}%): {real_precision*100:.1f}%")

plt.style.use("dark_background")
fig = plt.figure(figsize=(10.5, 12.5), facecolor="#0d131a")

fig.text(
    0.5, 0.945, "STREAMING INDUSTRIAL TELEMETRY GUARD",
    ha="center", va="center", fontsize=23, fontweight="bold", color="#e0fbfc"
)
fig.text(
    0.5, 0.912, "UCI AI4I 2020 Synthetic Benchmark • True HST Decision Manifold",
    ha="center", va="center", fontsize=10.5, color="#4cc9f0"
)

cols = [
    ("01", "ANALOG EDGE\nSHIELD", f"Isolates electrical spikes\n(step 300: {'isolated' if sensor_spikes[300] else 'missed'};\n{total_spikes} samples isolated in total)\nbefore physics fusion.", 0.09),
    ("02", "REGIME SHIFT\nTRACKING", f"Flags mechanical drift\nat step {drift_detected_at} ({lag_str})\nwith no batch retraining.", 0.39),
    ("03", "PREDICTIVE\nMAINTENANCE", f"Detects {fault_recall:.1f}% of failures\nwith {healthy_fp:.2f}% false alarms.\nAUC {auc:.3f}; precision {real_precision*100:.0f}%\nat dataset prevalence ({real_prev*100:.1f}%).", 0.69),
]

for num, title, desc, x_pos in cols:
    fig.text(x_pos, 0.865, num, fontsize=13, fontweight="bold", color="#f72585")
    fig.text(x_pos, 0.828, title, fontsize=11, fontweight="bold", color="#ffffff", va="top")
    fig.text(x_pos, 0.770, desc, fontsize=9.5, color="#cbd5e1", va="top", linespacing=1.35)

ax = fig.add_axes([0.04, 0.05, 0.92, 0.67], projection="3d")
ax.set_facecolor("#0d131a")

for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.pane.fill = False
    axis.pane.set_edgecolor("#1e293b")
    axis._axinfo["grid"]["color"] = (0.20, 0.30, 0.42, 0.45)
    axis._axinfo["grid"]["linewidth"] = 0.7

x_vals = np.clip(stream_norm[:, 3], -3.0, 4.5)
y_vals = np.clip(overstrain_history, -3.0, 4.5)
z_vals = anomaly_scores

gx = np.linspace(-3.0, 4.5, 30)
gy = np.linspace(-3.0, 4.5, 30)
GX, GY = np.meshgrid(gx, gy)
GZ = np.full_like(GX, np.nan)

med_rpm = float(np.median(stream_norm[:150, 2]))
rpm_phys_med = med_rpm * std0[2] + mu0[2]
wear_lo, wear_hi = float(stream_raw[:, 4].min()), float(stream_raw[:, 4].max())

for r in range(GX.shape[0]):
    for c in range(GX.shape[1]):
        torque_phys = float(GX[r, c]) * std0[3] + mu0[3]
        overstrain_phys = float(GY[r, c]) * std_inv[1] + mu_inv[1]
        if torque_phys <= 0.0:
            continue
        wear_phys = overstrain_phys / torque_phys
        if not (wear_lo <= wear_phys <= wear_hi):
            continue
        power_phys = torque_phys * rpm_phys_med * rpm_to_rad_s
        grid_sample = {
            "rpm": med_rpm,
            "torque": float(GX[r, c]),
            "wear": (wear_phys - mu0[4]) / std0[4],
            "power": float((power_phys - mu_inv[0]) / std_inv[0]),
            "overstrain": float(GY[r, c]),
            "delta_temp": 0.0,
        }
        GZ[r, c] = float(hst.score_one(grid_sample))

cyan_to_amber = LinearSegmentedColormap.from_list(
    "industrial_risk", ["#00b4d8", "#4cc9f0", "#7209b7", "#f72585", "#ff9e00"]
)

ax.plot_surface(
    GX, GY, GZ, cmap=cyan_to_amber, alpha=0.28,
    edgecolor="#48cae4", linewidth=0.35, rstride=1, cstride=1
)

mask_healthy = (np.arange(n_samples) < n_h) & (sensor_spikes == 0)
mask_fault_caught = (np.arange(n_samples) >= n_h) & (alerts == 1)
mask_spike_pt = (sensor_spikes == 1)

ax.scatter(
    x_vals[mask_healthy], y_vals[mask_healthy], z_vals[mask_healthy],
    color="#4cc9f0", s=22, alpha=0.85, edgecolors="#90e0ef", linewidth=0.3,
    label=f"HEALTHY OPERATION ({healthy_fp:.1f}% FP)"
)
ax.scatter(
    x_vals[mask_fault_caught], y_vals[mask_fault_caught], z_vals[mask_fault_caught],
    color="#ff9e00", s=34, alpha=0.95, edgecolors="#ffffff", linewidth=0.4,
    label=f"MECHANICAL FAILURES DETECTED ({fault_recall:.1f}%)"
)
ax.scatter(
    x_vals[mask_spike_pt], y_vals[mask_spike_pt], np.zeros(int(mask_spike_pt.sum())),
    color="#f72585", s=95, marker="^", edgecolors="#ffffff", linewidth=0.8,
    label=f"ISOLATED SENSOR SPIKES ({total_spikes}, plotted at score 0)"
)

ax.set_xlabel("TORQUE SENSOR [Nm, z-score]", color="#90e0ef", fontsize=9, fontweight="bold", labelpad=10)
ax.set_ylabel("OVERSTRAIN [min·Nm, z-score]", color="#90e0ef", fontsize=9, fontweight="bold", labelpad=10)
ax.set_zlabel("HST ANOMALY SCORE", color="#90e0ef", fontsize=9, fontweight="bold", labelpad=8)
ax.tick_params(colors="#64748b", labelsize=8)
ax.view_init(elev=26, azim=-50)
ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.9), frameon=True, facecolor="#161f2c", edgecolor="#334155", fontsize=8.5)

out_path = os.path.abspath("industrial_iiot_3d_manifold.png")
plt.savefig(out_path, dpi=220, facecolor="#0d131a")
print(f"Saved PNG to: {out_path}")
