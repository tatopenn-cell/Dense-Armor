"""Generate the two figures for docs/drift/detectors.md."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dense_armor.utility.drift.adwin import ADWIN
from dense_armor.utility.drift.detector import CUSUMDriftDetector
from dense_armor.utility.drift.kswin import KSWIN
from dense_armor.utility.drift.page_hinkley import PageHinkley

OUT = Path(__file__).parent


def four_detectors_on_a_step():
    rng = np.random.default_rng(0)
    n = 1500
    change = 800
    x = np.concatenate(
        [rng.normal(0.0, 1.0, change), rng.normal(2.0, 1.0, n - change)]
    )
    detectors = [
        ("CUSUM fixed",
         CUSUMDriftDetector(reference="fixed", radius=5, ref_mult=4),
         "crimson"),
        ("Page-Hinkley",
         PageHinkley(delta=0.05, threshold=20.0),
         "darkorange"),
        ("ADWIN",
         ADWIN(delta=0.002, max_window=100),
         "seagreen"),
        ("KSWIN",
         KSWIN(seed=0),
         "royalblue"),
    ]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(x, color="0.6", lw=0.7, label="signal")
    ax.axvline(change, color="black", ls=":", lw=1.0,
               label=f"change at {change}")
    for name, det, color in detectors:
        first = next(
            (i for i, v in enumerate(x) if det.update(v).drift_detected),
            None,
        )
        if first is not None:
            ax.scatter([first], [x[first]], s=70, marker="v",
                       color=color, zorder=5,
                       label=f"{name} (t={first})")
    ax.set_xlabel("sample")
    ax.set_ylabel("value")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "four_detectors.png", dpi=140)
    plt.close(fig)


def adwin_bound():
    delta = 0.002
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ns = np.arange(4, 501)
    n_cuts = ns - 3
    m = 1.0 / (1.0 / (ns // 2) + 1.0 / (ns - ns // 2))
    eps = np.sqrt(np.log(2.0 * n_cuts / delta) / (2.0 * m))
    ax.plot(ns, eps, label=f"cut in the middle, delta = {delta}")
    ax.set_xlabel("window length n (samples)")
    ax.set_ylabel(r"$\varepsilon_\mathrm{cut}$  (units of $R$)")
    ax.set_ylim(0.0, 1.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "adwin_bound.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    four_detectors_on_a_step()
    adwin_bound()
    print("wrote", OUT / "four_detectors.png")
    print("wrote", OUT / "adwin_bound.png")