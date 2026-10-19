"""GPS vs IMU: is the GPS telling the truth?

SemperFi (Sathaye et al. 2021) describes a GPS spoofer that
"gradually introduces offsets" into a victim's position and
velocity (section 2.2, page 3). The detector in that paper is a
hardware correlator (section 4, page 4) that "compares the track
estimated by IMU sensors and GPS by averaging the Euclidean
distance between each deviation sample". This pipeline does the
same comparison in software, on velocity instead of position:

1. Start the IMU velocity from the first GPS velocity (the same
   anchoring an EKF does), then integrate the accelerometer with
   gravity removed. The integration gives the change in velocity;
   the anchor makes it the absolute velocity.
2. The residual is ``gps_v - imu_v``, one three-vector per sample.
3. The score is the Euclidean norm of the residual, matching the
   "Euclidean distance" of the paper.
4. A CUSUM drift detector on the score marks an episode, with
   ``reset_on_alarm=True``. Episodes are rising edges of
   ``drift_detected``, same convention as B1 and as
   ``benchmarks/casper_hours_episodes.py`` lines 74-76.

The spoofer is injected by the synthetic stream, with the kind
``gps_ramp``: from the fault sample on, the north GPS velocity
drifts away from the true one at 1 m/s^2. The page says the
attack is injected, not reimplemented.

The CUSUM parameters are the same as in B1. The metric is the
same "detection delay" and "false alarms per hour" of ALFA
(Keipour et al. 2020, section 4, page 5).

Example:
    >>> from pipelines.drones.io import synthetic_flight
    >>> from pipelines.drones.sensor_consistency import run
    >>> r = run(list(synthetic_flight(n=300, seed=0, fault="gps_ramp")))
    >>> r.fault_kind
    'gps_ramp'
    >>> r.n_samples
    300
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dense_armor.utility.drift.detector import CUSUMDriftDetector

GRAVITY_NED = np.array([0.0, 0.0, 9.81])
DEFAULT_CUSUM = {
    "reference": "fixed",
    "radius": 20,
    "ref_mult": 5,
    "k": 0.5,
    "h": 20.0,
    "two_sided": True,
    "reset_on_alarm": True,
}


@dataclass
class SensorConsistencyResult:
    """Result of one run of the GPS-vs-IMU pipeline.

    Attributes:
        n_samples: number of samples seen.
        fault_kind: kind of the injected fault, or None.
        fault_index: sample index at which the fault starts, or -1.
        n_episodes: number of rising edges of ``drift_detected``.
        delays_s: detection delays in seconds, one per episode that
            starts after the fault. Empty if no fault.
        max_delay_s: maximum delay, or None.
        mean_delay_s: mean delay, or None.
        normal_seconds: length of the normal part, in seconds.
        false_alarms: episodes in the normal part.
        false_alarms_per_hour: rate on normal flight.
        residual: per-sample ``gps_v - imu_v`` (n, 3).
        score: per-sample Euclidean norm of the residual.
        alarms: per-sample ``drift_detected``.
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
    residual: np.ndarray
    score: np.ndarray
    alarms: np.ndarray


def _episode_starts(alarms: np.ndarray) -> np.ndarray:
    """Boolean array: True at the first True of each run of True."""
    if len(alarms) == 0:
        return alarms
    starts = np.empty_like(alarms, dtype=bool)
    starts[0] = alarms[0]
    starts[1:] = alarms[1:] & ~alarms[:-1]
    return starts


def run(
    stream: list[dict],
    cusum_kwargs: dict | None = None,
) -> SensorConsistencyResult:
    """Run the GPS-vs-IMU consistency pipeline on a list of samples.

    Args:
        stream: list of dicts from
            :func:`pipelines.drones.io.synthetic_flight` or
            :func:`pipelines.drones.io.read_mavlink_log`. Each dict
            needs ``acc`` and ``gps_v`` (3-vectors) and may carry
            ``t``, ``fault_started`` and ``fault_kind``.
        cusum_kwargs: keyword arguments forwarded to
            :class:`CUSUMDriftDetector`.

    Returns:
        :class:`SensorConsistencyResult`.
    """
    cusum_kwargs = {**DEFAULT_CUSUM, **(cusum_kwargs or {})}
    cusum = CUSUMDriftDetector(**cusum_kwargs)

    n = len(stream)
    residual = np.full((n, 3), np.nan)
    score = np.full(n, np.nan)
    alarms: np.ndarray = np.zeros(n, dtype=bool)
    times: np.ndarray = np.zeros(n, dtype=float)
    fault_index = -1
    fault_kind: str | None = None

    imu_v: np.ndarray | None = None
    prev_t: float | None = None

    for i, sample in enumerate(stream):
        t = float(sample.get("t", i * 0.01))
        times[i] = t
        if fault_index < 0 and sample.get("fault_started"):
            fault_index = i
            fault_kind = sample.get("fault_kind")
        acc = sample.get("acc")
        gps_v = sample.get("gps_v")
        if acc is None or gps_v is None:
            prev_t = t
            continue
        acc = np.asarray(acc, dtype=float)
        gps_v = np.asarray(gps_v, dtype=float)
        if imu_v is None:
            imu_v = gps_v.copy()
        elif prev_t is not None:
            dt = t - prev_t
            if dt > 0:
                imu_v = imu_v + (acc - GRAVITY_NED) * dt
        prev_t = t
        r = gps_v - imu_v
        residual[i] = r
        norm = float(np.linalg.norm(r))
        score[i] = norm
        cusum.update(norm)
        alarms[i] = bool(cusum.drift_detected)

    starts = _episode_starts(alarms)
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
    if normal_seconds > 0:
        fa_per_hour = false_alarms * 3600.0 / normal_seconds
    else:
        fa_per_hour = float("nan")

    delays: list[float] = []
    if fault_index >= 0:
        t_fault = times[fault_index]
        for i in map(int, np.flatnonzero(starts)):
            if i >= fault_index:
                delays.append(float(times[i] - t_fault))
    max_delay = max(delays) if delays else None
    mean_delay = float(np.mean(delays)) if delays else None

    return SensorConsistencyResult(
        n_samples=n,
        fault_kind=fault_kind,
        fault_index=fault_index,
        n_episodes=n_episodes,
        delays_s=delays,
        max_delay_s=max_delay,
        mean_delay_s=mean_delay,
        normal_seconds=normal_seconds,
        false_alarms=false_alarms,
        false_alarms_per_hour=fa_per_hour,
        residual=residual,
        score=score,
        alarms=alarms,
    )
