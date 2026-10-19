"""Generate the PNGs of docs/drones/propeller_health.md.

Two figures, from the library's own functions on a synthetic stream
labelled "synthetic" (pipelines.drones.io.synthetic_flight).

- ``propeller_health_score.png``: the score (absolute difference of the
  band max amplitude from the baseline) with the threshold.
- ``propeller_health_spectrum.png``: the amplitude spectrum before and
  after the fault, showing the 47 Hz peak.

Run from the repository root:

    python benchmarks/drones/make_figure_propeller.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dense_armor.utility.misc import SDFT
from pipelines.drones.io import synthetic_flight
from pipelines.drones.propeller_health import DEFAULT_BAND_HZ, run

HERE = Path(__file__).resolve().parents[2] / "docs" / "assets" / "drones"
HERE.mkdir(parents=True, exist_ok=True)


def _score_plot():
    stream = list(synthetic_flight(n=2000, seed=0, fault="propeller"))
    r = run(stream)
    t = np.array([s["t"] for s in stream])
    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.plot(t, r.score, color="#2c6fa6", label="score (band max minus baseline)")
    ax.axhline(10.0, color="red", linestyle="--", label="threshold 10")
    if r.fault_index >= 0:
        ax.axvline(t[r.fault_index], color="black", linestyle="--",
                   label="fault")
    if r.n_episodes:
        ep_idx = []
        for d in r.delays_s:
            ep_idx.append(r.fault_index + round(d * 100))
        ax.scatter(t[ep_idx], r.score[ep_idx], color="red", marker="x",
                   zorder=5, label=f"episode (n={r.n_episodes})")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("score")
    ax.set_title(
        "Propeller fault, synthetic stream, band [40, 55] Hz",
        fontsize=12, weight="bold",
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "propeller_health_score.png", dpi=140)
    plt.close(fig)
    print(f"  propeller_health_score.png: fault_index={r.fault_index}, "
          f"episodes={r.n_episodes}, "
          f"delays={[round(d, 3) for d in r.delays_s]}")


def _spectrum_plot():
    stream = list(synthetic_flight(n=2000, seed=0, fault="propeller"))
    fault_at = round(0.6 * 2000)
    sdft = SDFT(n=128)
    prev = 0.0
    amps_before, amps_after = [], []
    for i, s in enumerate(stream):
        g = float(np.asarray(s["gyro"])[0])
        sdft.update(g - prev)
        prev = g
        if sdft.is_ready:
            amp = np.abs(sdft.spectrum())
            if 200 <= i < 250:
                amps_before.append(amp)
            elif fault_at + 100 <= i < fault_at + 150:
                amps_after.append(amp)
    b = np.mean(amps_before, axis=0)
    a = np.mean(amps_after, axis=0)
    freqs = np.fft.fftfreq(128, d=0.01)
    lo, hi = DEFAULT_BAND_HZ
    mask = (np.abs(freqs) >= lo) & (np.abs(freqs) <= hi)
    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.plot(np.abs(freqs[mask]), b[mask], label="before fault", color="#2c6fa6")
    ax.plot(np.abs(freqs[mask]), a[mask], label="after fault", color="red")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("|X| of the differenced gyro")
    ax.set_title(
        "Amplitude spectrum in the band, before and after the fault",
        fontsize=12, weight="bold",
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "propeller_health_spectrum.png", dpi=140)
    plt.close(fig)
    k = int(np.argmax(a[mask]))
    peak_hz = float(np.abs(freqs[mask][k]))
    print(f"  propeller_health_spectrum.png: peak_after={peak_hz:.2f} Hz, "
          f"max_before={b[mask].max():.3f}, max_after={a[mask].max():.3f}")


if __name__ == "__main__":
    print("Generating docs/assets/drones/ (synthetic stream)")
    _score_plot()
    _spectrum_plot()
    print("Done.")
