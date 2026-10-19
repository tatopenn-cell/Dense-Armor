"""Propeller health: is a propeller damaged?

Zhang et al. 2023 (section V, page 5) describe the fault as one broken
propeller installed while the other three are healthy, and say that
"the quadrotor's angular accelerations will change" when the propeller
changes (page 3). They train a DCNN on a nine-channel temporal signal;
they do not use a spectrum.

The spectral approach of this pipeline follows Baldini et al. 2026
("Active Propeller Fault Detection and Isolation in Multirotors Via
Vibration Model", arXiv:2604.07281v1, section 3.2, page 5, equation
26):

    r_{Omega, T}(t) = max_{omega in Omega} |F[rect(t-T, t) . v_dot(t)]|

where F is the Fourier transform, rect filters the last T seconds,
v_dot is the measured acceleration, Omega is a frequency band around
the motor speed, and the residual is compared with a fixed threshold
rho_FD. The paper uses Goertzel and a threshold rho_FD = 0.005 (page 6)
chosen via ROC (page 8).

Four deviations from Baldini, each declared in the page:

- the signal is the gyro (differentiated), not the acceleration. The
  synthetic stream puts the vibration on the gyro; the acceleration
  in synthetic_flight does not carry it.
- the spectrum is the SDFT of dense_armor.utility.misc, not Goertzel.
  The SDFT is the library's nearest tool to the DFT of the paper;
  it is O(N) per sample and gives the same amplitude spectrum.
- the band Omega is fixed at [40, 55] Hz. Baldini defines Omega
  around the motor speed; the synthetic stream has no RPM. The band
  is my choice and contains the 47 Hz of the synthetic fault.
- the threshold is 10.0. Baldini uses 0.005 on its own acceleration
  scale; on this synthetic stream the healthy score is below 2.5 and
  the propeller score above 88, so 10.0 separates them with 0 false
  alarms and one episode, with a delay of 0.06 s.

The pipeline:

1. Difference the x component of the gyro (g[t] - g[t-1]), then feed
   the difference to a sliding DFT of N points (SDFT). The difference
   is a choice: the synthetic trajectory has a slow oscillation around
   0.5 Hz that swamps the spectrum; the difference amplifies
   frequencies proportionally, so the 47 Hz vibration survives and
   the 0.5 Hz trajectory is attenuated by a factor of about 100.
2. When the SDFT is ready, take the amplitude spectrum |X| and its
   maximum in the band Omega = [40, 55] Hz.
3. Build the baseline as the mean of the maxima over the first
   ``warmup`` spectra. The baseline is the healthy behaviour.
4. The score is the absolute difference between the current maximum
   and the baseline maximum (a distance in the band, not over the
   whole spectrum: this is the band energy of the spec).
5. The detector is a fixed threshold on the score, not a CUSUM. A
   crossing starts an episode; the episode ends when the score goes
   back below the threshold. This is the paper's mechanism.

The synthetic fault ``propeller`` in ``synthetic_flight`` adds a 47 Hz
sine of amplitude 0.5 rad/s to the x gyro from the fault sample on.
The frequency is my choice; neither Zhang nor Baldini gives one.

Example:
    >>> from pipelines.drones.io import synthetic_flight
    >>> from pipelines.drones.propeller_health import run
    >>> r = run(list(synthetic_flight(n=2000, seed=0, fault="propeller")))
    >>> r.fault_kind
    'propeller'
    >>> r.n_samples
    2000
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dense_armor.utility.misc import SDFT

DEFAULT_N = 128
DEFAULT_WARMUP = 50
DEFAULT_THRESHOLD = 10.0
DEFAULT_BAND_HZ = (40.0, 55.0)


@dataclass
class PropellerHealthResult:
    """Result of one run of the propeller health pipeline.

    Attributes:
        n_samples: number of samples seen.
        fault_kind: kind of the injected fault, or None.
        fault_index: sample index at which the fault starts, or -1.
        n_episodes: number of rising edges of ``drift_detected``.
        delays_s: detection delays in seconds, one per episode after the fault.
        max_delay_s: maximum delay, or None.
        mean_delay_s: mean delay, or None.
        normal_seconds: length of the normal part, in seconds.
        false_alarms: episodes in the normal part.
        false_alarms_per_hour: rate on normal flight.
        score: per-sample L2 distance from the baseline.
        alarms: per-sample ``drift_detected``.
        dominant_freq_hz: frequency of the largest baseline peak.
    """

    n_samples: int
    fault_kind: str | None
    fault_index: int
    n_episodes: int
    delays_s: list[float]
    max_delay_s: float | None
    mean_delay_s: float | None
    normal_seconds: float
    false_alarms: int
    false_alarms_per_hour: float
    score: np.ndarray
    alarms: np.ndarray
    dominant_freq_hz: float


def run(
    stream: list[dict],
    n_sdft: int = DEFAULT_N,
    warmup: int = DEFAULT_WARMUP,
    threshold: float = DEFAULT_THRESHOLD,
    band_hz: tuple[float, float] = DEFAULT_BAND_HZ,
) -> PropellerHealthResult:
    """Run the propeller health pipeline on a list of samples.

    Args:
        stream: list of dicts from
            :func:`pipelines.drones.io.synthetic_flight` or
            :func:`pipelines.drones.io.read_mavlink_log`. Each dict
            needs ``gyro`` (3-vector) and may carry ``t``,
            ``fault_started`` and ``fault_kind``.
        n_sdft: number of points of the sliding DFT. Default 128,
            which gives a bin width of 100 / 128 = 0.78 Hz at 100 Hz.
        warmup: number of spectra averaged into the baseline. Default 50.
        threshold: score above which the sample is flagged. Default 10.0.
        band_hz: frequency band, in Hz, of the max-amplitude feature.
            Default (40.0, 55.0), which contains the 47 Hz synthetic fault.

    Returns:
        :class:`PropellerHealthResult`.
    """
    sdft = SDFT(n=n_sdft)
    sample_rate_hz = 100.0
    freqs = np.fft.fftfreq(n_sdft, d=1.0 / sample_rate_hz)
    lo, hi = band_hz
    band_idx = np.flatnonzero((np.abs(freqs) >= lo) & (np.abs(freqs) <= hi))
    if len(band_idx) == 0:
        raise ValueError(f"band_hz {band_hz} contains no SDFT bin")

    n = len(stream)
    score = np.full(n, np.nan)
    alarms: np.ndarray = np.zeros(n, dtype=bool)
    times: np.ndarray = np.zeros(n, dtype=float)
    fault_index = -1
    fault_kind: str | None = None
    baseline_max: float | None = None
    baseline_sum = 0.0
    baseline_count = 0
    prev_gyro_x = 0.0
    dominant_freq = 0.0

    for i, sample in enumerate(stream):
        times[i] = float(sample.get("t", i * 0.01))
        if fault_index < 0 and sample.get("fault_started"):
            fault_index = i
            fault_kind = sample.get("fault_kind")
        gyro = sample.get("gyro")
        if gyro is None:
            continue
        gx = float(np.asarray(gyro, dtype=float)[0])
        sdft.update(gx - prev_gyro_x)
        prev_gyro_x = gx
        if not sdft.is_ready:
            continue
        amp = np.abs(sdft.spectrum())
        m = float(amp[band_idx].max())
        if baseline_count < warmup:
            baseline_sum += m
            baseline_count += 1
            if baseline_count == warmup:
                baseline_max = baseline_sum / warmup
                k = int(np.argmax(amp[1:]))
                dominant_freq = float(abs(freqs[k + 1]))
                if dominant_freq == 0.0:
                    dominant_freq = float(abs(freqs[1]))
            continue
        if baseline_max is None:
            continue
        s = abs(m - baseline_max)
        score[i] = s
        alarms[i] = bool(s > threshold)

    # episodes: runs of consecutive True in alarms
    if len(alarms) == 0:
        starts = alarms
    else:
        starts = np.r_[alarms[0], alarms[1:] & ~alarms[:-1]]
    n_episodes = int(starts.sum())
    if 0 <= fault_index < n:
        normal_end = fault_index
    else:
        normal_end = n
    if normal_end > 0:
        normal_seconds = float(times[normal_end - 1] - times[0])
    else:
        normal_seconds = 0.0
    false_alarms = int(starts[:normal_end].sum())
    fa_per_hour = (
        false_alarms * 3600.0 / normal_seconds if normal_seconds > 0 else float("nan")
    )
    delays: list[float] = []
    if fault_index >= 0:
        t_fault = times[fault_index]
        for i in np.flatnonzero(starts):
            if i >= fault_index:
                delays.append(float(times[i] - t_fault))
    return PropellerHealthResult(
        n_samples=n,
        fault_kind=fault_kind,
        fault_index=fault_index,
        n_episodes=n_episodes,
        delays_s=delays,
        max_delay_s=max(delays) if delays else None,
        mean_delay_s=float(np.mean(delays)) if delays else None,
        normal_seconds=normal_seconds,
        false_alarms=false_alarms,
        false_alarms_per_hour=fa_per_hour,
        score=score,
        alarms=alarms,
        dominant_freq_hz=dominant_freq,
    )
