"""Generate the three figures for docs/timeseries/.

- ar_convergence.png: OnlineAR coefficient converging to the true AR(1).
- kalman_R_tracking.png: R estimate of the two adaptive filters after a jump.
- cycle_residual.png: cycle residual on a SyntheticArm payload fault.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dense_armor.utility.timeseries import (
    OnlineAR, KalmanFilter, VBAdaptiveKalman, InnovationAdaptiveKalman,
)

out = Path(__file__).parent

# --- 1. OnlineAR convergence ---
rng = np.random.default_rng(0)
n = 500
series = [0.0]
for _ in range(n):
    series.append(0.8 * series[-1] + 0.3 * rng.normal())
ar = OnlineAR(order=1, lam=1.0)
w_hist = []
for v in series[:-1]:
    ar.learn_one({"x": float(v)})
    w_hist.append(float(ar.coef_[0]))

fig, ax = plt.subplots(figsize=(8, 3.2))
ax.plot(w_hist, color="tab:blue", lw=1.2)
ax.axhline(0.8, color="0.5", lw=0.8, ls="--")
ax.text(350, 0.82, "true coefficient 0.8", fontsize=8, color="0.4")
ax.set_xlabel("sample")
ax.set_ylabel("w")
ax.set_ylim(0.0, 1.0)
fig.tight_layout()
fig.savefig(out / "ar_convergence.png", dpi=130)

# --- 2. Kalman adaptive R tracking ---
F = np.array([[1.0, 1.0], [0.0, 1.0]])
H = np.array([[1.0, 0.0]])
Q = 1e-4 * np.eye(2)
R = np.array([[1e-2]])
half = 200
N = 400

fig, ax = plt.subplots(figsize=(8, 3.2))
for cls, color, name in [
    (VBAdaptiveKalman, "tab:blue", "VB-AKF"),
    (InnovationAdaptiveKalman, "tab:orange", "Innovation-adaptive"),
]:
    kf = cls(F=F, H=H, Q=Q, R=R)
    rng = np.random.default_rng(0)
    r_hist = []
    for i in range(N):
        std = 0.1 if i < half else 1.0
        pos = 1.0 + 0.1 * i + std * rng.normal()
        kf.learn_one({"pos": float(pos)})
        if cls is VBAdaptiveKalman:
            r_hist.append(float(kf.R_estimate[0, 0]))
        else:
            r_hist.append(float(kf.R_estimate[0, 0]))
    ax.plot(r_hist, color=color, lw=1.2, label=name)

ax.axvline(half, color="0.3", lw=0.8, ls="--")
ax.axhline(1.0, color="0.5", lw=0.8, ls=":")
ax.text(half + 4, 1.05, "true R after the jump", fontsize=8, color="0.4")
ax.set_xlabel("sample")
ax.set_ylabel("R estimate")
ax.set_ylim(0.0, 1.4)
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(out / "kalman_R_tracking.png", dpi=130)

# --- 3. CycleResidual on a real arm ---
from dense_armor.utility.datasets import SyntheticArm
from dense_armor.utility.timeseries import CycleResidual

ds = SyntheticArm(
    Path("test/fixtures/urdf/panda.urdf"),
    period_s=1.0, rate_hz=50.0, n_cycles=16,
    fault_at_s=12.0, fault="payload",
    payload_mass=5.0, seed=0,
)
taus = np.asarray([float(sig["tau_1"]) for sig, _ in ds.stream()])
cr = CycleResidual(period=200, n_cycles=2)
res = np.asarray([cr.transform_one({"x": float(v)})["residual"] for v in taus])

fig, ax = plt.subplots(figsize=(8, 3.2))
ax.plot(np.arange(len(res)) / 50.0, res, color="tab:blue", lw=1.0)
ax.axvline(12.0, color="0.3", lw=0.8, ls="--")
ax.text(12.1, 300.0, "fault on", fontsize=8, color="0.3")
ax.set_xlabel("time (s)")
ax.set_ylabel("cycle residual (N m)")
fig.tight_layout()
fig.savefig(out / "cycle_residual.png", dpi=130)

print("written:", out / "ar_convergence.png")
print("written:", out / "kalman_R_tracking.png")
print("written:", out / "cycle_residual.png")
