import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dense_armor.dynamics.urdf_dynamics import RigidBodyModel
from dense_armor.utility.linear.payload import PayloadEstimator
from dense_armor.utility.datasets import SyntheticArm

out = Path(__file__).parent
URDF = Path("test/fixtures/urdf/panda.urdf")
model = RigidBodyModel(URDF)

fig, ax = plt.subplots(figsize=(8, 3.2))
for lam, color in [(0.999, "tab:blue"), (0.99, "tab:orange"),
                    (0.95, "tab:red")]:
    ds = SyntheticArm(URDF, period_s=4.0, rate_hz=50.0, n_cycles=2,
                      fault_at_s=2.0, fault="payload", payload_mass=5.0,
                      payload_link="panda_link7", seed=0)
    est = PayloadEstimator(model, link="panda_link7", lam=lam)
    ts, ms = [], []
    for sig, _ in ds.stream():
        est.learn_one(sig)
        ts.append(float(getattr(sig, "t", 0.0)))
        ms.append(float(est.predict_one()))
    ax.plot(ts, ms, color=color, lw=1.2, label=f"lam = {lam}")
    t_arr, m_arr = np.asarray(ts), np.asarray(ms)
    i15 = int(np.argmin(np.abs(t_arr - 1.5)))
    after = np.where((t_arr >= 2.0) & (np.abs(m_arr - 5.0) <= 0.5))[0]
    print(
        f"lam {lam}: t=1.5 -> {m_arr[i15]:.4f} kg | pre-fault max {m_arr[t_arr < 2.0].max():.4f} | "
        f"first within 10% after {None if after.size == 0 else round(float(t_arr[after[0]] - 2.0), 2)} s | "
        f"at t=3.5 {m_arr[int(np.argmin(np.abs(t_arr - 3.5)))]:.3f} | final {m_arr[-1]:.3f}"
    )

ax.axvline(2.0, color="0.5", lw=0.8, ls="--")
ax.axhline(5.0, color="0.5", lw=0.8, ls=":")
ax.text(2.05, 5.15, "true mass = 5.0 kg", fontsize=8, color="0.4")
ax.text(2.05, 0.3, "fault on", fontsize=8, color="0.4")
ax.set_xlabel("time [s]")
ax.set_ylabel("estimated mass [kg]")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(out / "payload_tracking.png", dpi=130)
print("scritto:", out / "payload_tracking.png")
