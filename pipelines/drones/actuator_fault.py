"""Actuator fault detection: which part broke, and how many seconds after?

The pipeline has three steps:

1. Residual = commanded rate minus measured rate, three components.
2. CUSUM drift detector on the L2 norm of the residual, with
   ``reset_on_alarm=True`` so a sustained fault gives one *episode*
   (the rising edge of ``drift_detected``), not one alarm per sample.
   The convention for counting episodes is the one of
   ``benchmarks/casper_hours_episodes.py`` lines 74-76: a sample
   with ``drift_detected=True`` whose previous sample had
   ``drift_detected=False``.
3. OnlineRobustMahalanobis on the three residual components, for
   the score threshold.

Two metrics, both from ALFA (Keipour, Mousaei & Scherer 2020,
section 4, page 5):

- **Detection delay** in seconds, for each detected episode, from
  the fault sample to the first sample of the episode. ALFA calls
  it "Maximum Detection Time" and "Average Detection Time". Here
  the pipeline reports both, in ``max_delay_s`` and
  ``mean_delay_s``.
- **False alarms per hour** of normal flight. An hour that holds
  an alarm episode is one false positive; episodes are counted on
  the normal part of the stream, then scaled by
  ``3600 / normal_seconds``. This is the same convention of
  ``benchmarks/casper_hours_episodes.py`` lines 129-132 for the
  CASPER stream.

The synthetic fault ``actuator_loss`` follows Garg and Fan 2023,
section II, equation (4), page 2: a faulty actuator scales the
commanded input by ``Theta_i in [0, 1)``, so the residual is
``commanded * (1 - Theta_i)`` on the faulty axis and noise on the
healthy ones.

On ALFA, the fault is a control-surface failure (Keipour et al.
2020, page 1: aileron, elevator, rudder, engine), not a quadrotor
motor. The residual of the surface deflection against the
commanded one is the same shape as here, but the log comes from
``read_mavlink_log`` (telemetry tlog, page 3 section 3.1) and not
from a PX4 ULog.

Example:
    >>> from pipelines.drones.io import synthetic_flight
    >>> from pipelines.drones.actuator_fault import run
    >>> r = run(list(synthetic_flight(n=300, seed=0, fault="actuator_loss")))
    >>> r.fault_kind
    'actuator_loss'
    >>> r.n_samples
    300
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dense_armor.utility.anomaly.mahalanobis import OnlineRobustMahalanobis
from dense_armor.utility.drift.detector import CUSUMDriftDetector

RESIDUAL_KEYS = ("r0", "r1", "r2")
DEFAULT_CUSUM = {
    "reference": "fixed",
    "radius": 20,
    "ref_mult": 5,
    "k": 0.5,
    "h": 20.0,
    "two_sided": True,
    "reset_on_alarm": True,
}
DEFAULT_MAHA = {
    "n_init": 100,
    "threshold": 7.0,
    "feature_keys": RESIDUAL_KEYS,
    "seed": 0,
}


@dataclass
class ActuatorFaultResult:
    """Result of one run of the actuator-fault pipeline.

    Attributes:
        n_samples: number of samples seen.
        fault_kind: the kind of the injected fault, or None.
        fault_index: sample index at which the fault starts, or -1.
        n_episodes: number of rising edges of ``drift_detected``.
        delays_s: list of detection delays in seconds, one per
            episode that starts after the fault. Empty if no fault.
        max_delay_s: maximum of ``delays_s``, or None.
        mean_delay_s: mean of ``delays_s``, or None.
        normal_seconds: length of the normal part of the stream,
            in seconds, used as the denominator of the rate.
        false_alarms: number of episodes in the normal part.
        false_alarms_per_hour: ``false_alarms * 3600 / normal_seconds``.
        mahalanobis_scores: per-sample Mahalanobis score.
        cusum_alarms: per-sample bool, ``drift_detected``.
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
    mahalanobis_scores: np.ndarray
    cusum_alarms: np.ndarray


def _residual(sample: dict) -> np.ndarray | None:
    """Return the 3-vector ``commanded - gyro``, or None."""
    cmd = sample.get("commanded")
    gyro = sample.get("gyro")
    if cmd is None or gyro is None:
        return None
    return np.asarray(cmd, dtype=float) - np.asarray(gyro, dtype=float)


def _episode_starts(alarms: np.ndarray) -> np.ndarray:
    """Boolean array: True at the first True of each run of True.

    This is the convention of ``casper_hours_episodes.py`` lines
    74-76: a sample counts as the start of an episode when its
    ``drift_detected`` is True and the previous sample's is False.
    """
    if len(alarms) == 0:
        return alarms
    starts = np.empty_like(alarms, dtype=bool)
    starts[0] = alarms[0]
    starts[1:] = alarms[1:] & ~alarms[:-1]
    return starts


def run(
    stream: list[dict],
    cusum_kwargs: dict | None = None,
    maha_kwargs: dict | None = None,
) -> ActuatorFaultResult:
    """Run the actuator-fault pipeline on a list of samples.

    Args:
        stream: list of dicts from :func:`pipelines.drones.io.synthetic_flight`
            or :func:`pipelines.drones.io.read_mavlink_log`. Each dict
            needs ``gyro`` and ``commanded`` (3-vectors), and may
            carry ``t`` (seconds), ``fault_started`` and ``fault_kind``.
        cusum_kwargs: keyword arguments forwarded to
            :class:`CUSUMDriftDetector`. The default is
            ``reference="fixed", radius=20, ref_mult=5, k=0.5, h=20.0,
            two_sided=True, reset_on_alarm=True``.
        maha_kwargs: keyword arguments forwarded to
            :class:`OnlineRobustMahalanobis`. The default is
            ``n_init=100, threshold=7.0, feature_keys=("r0","r1","r2")``.

    Returns:
        :class:`ActuatorFaultResult`.
    """
    cusum_kwargs = {**DEFAULT_CUSUM, **(cusum_kwargs or {})}
    maha_kwargs = {**DEFAULT_MAHA, **(maha_kwargs or {})}

    cusum = CUSUMDriftDetector(**cusum_kwargs)
    maha = OnlineRobustMahalanobis(**maha_kwargs)

    n = len(stream)
    alarms: np.ndarray = np.zeros(n, dtype=bool)
    scores: np.ndarray = np.full(n, np.nan, dtype=float)
    times: np.ndarray = np.zeros(n, dtype=float)
    fault_index = -1
    fault_kind: str | None = None

    for i, sample in enumerate(stream):
        times[i] = float(sample.get("t", i * 0.01))
        if fault_index < 0 and sample.get("fault_started"):
            fault_index = i
            fault_kind = sample.get("fault_kind")
        r = _residual(sample)
        if r is None:
            continue
        d = dict(zip(RESIDUAL_KEYS, r.tolist()))
        scores[i] = maha.score_one(d)
        maha.learn_one(d)
        cusum.update(float(np.linalg.norm(r)))
        alarms[i] = bool(cusum.drift_detected)

    starts = _episode_starts(alarms)
    n_episodes = int(starts.sum())

    if fault_index >= 0 and fault_index < n:
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

    return ActuatorFaultResult(
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
        mahalanobis_scores=scores,
        cusum_alarms=alarms,
    )
