"""Link health: is the radio link about to fail?

A drone sends a packet every 100 ms over a radio link. When the link
degrades, the packets arrive later and their signal is weaker. When the
link drops, packets are lost. The question is: is the link about to
fail, before the link actually fails?

Two online drift detectors run on two signals:

1. the inter-arrival time between consecutive packets, in seconds,
   with a CUSUM set to detect an **increase** (packets slowing down);
2. the RSSI of the packets, in dBm, with a CUSUM on its **negation**
   (signal getting weaker is a decrease in RSSI, which is an increase
   in -RSSI).

If either detector fires, the pipeline raises an alarm for that
packet. An episode is a rising edge of the combined alarm, same
convention as the other pages (benchmarks/casper_hours_episodes.py,
lines 74-76).

The stream is synthetic. The spec of Task D1 does not cite a paper for
this pipeline, and no paper on MQTT link health is in the task folder.
Every design choice below is my choice, declared in Details of the
page:

- nominal inter-arrival 100 ms with Gaussian noise 5 ms,
- nominal RSSI -60 dBm with Gaussian noise 2 dBm,
- ``link_degradation``: from the fault sample on, inter-arrival grows
  linearly to 150 ms over 2 s, RSSI falls to -75 dBm over 2 s,
- ``link_dropout``: from the fault sample on, 50% of the packets are
  lost, so the average inter-arrival doubles and the RSSI stays the
  same,
- CUSUM with ``reference="fixed"`` and ``reset_on_alarm=True`` on both
  signals, the same detector of the other pages.

Example:
    >>> from pipelines.drones.link_health import synthetic_mqtt
    >>> s = list(synthetic_mqtt(n=5, seed=0))
    >>> len(s)
    5
    >>> sorted(s[0].keys())
    ['fault_kind', 'fault_started', 'inter_arrival_s', 'payload_bytes', 'rssi_dbm', 't']
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dense_armor.utility.drift.detector import CUSUMDriftDetector

FAULT_KINDS = (None, "link_degradation", "link_dropout")
FAULT_AT_FRACTION = 0.6
NOMINAL_INTER_ARRIVAL_S = 0.100
NOMINAL_RSSI_DBM = -60.0
DEFAULT_CUSUM = {
    "reference": "fixed",
    "radius": 20,
    "ref_mult": 5,
    "k": 0.5,
    "h": 20.0,
    "two_sided": False,
    "reset_on_alarm": True,
}


@dataclass
class LinkHealthResult:
    """Result of one run of the link health pipeline.

    Attributes:
        n_packets: number of packets seen.
        fault_kind: the injected fault, or None.
        fault_index: packet index at which the fault starts, or -1.
        n_episodes: rising edges of the combined alarm.
        delays_s: detection delays in seconds, one per episode after
            the fault.
        max_delay_s: maximum delay, or None.
        mean_delay_s: mean delay, or None.
        normal_seconds: length of the normal part, in seconds.
        false_alarms: episodes in the normal part.
        false_alarms_per_hour: false alarms per hour of normal flight.
        inter_arrival: per-packet inter-arrival, in seconds.
        rssi: per-packet RSSI, in dBm.
        alarm_inter: per-packet alarm of the inter-arrival detector.
        alarm_rssi: per-packet alarm of the RSSI detector.
        alarm: per-packet combined alarm (either detector).
    """

    n_packets: int
    fault_kind: str | None
    fault_index: int
    n_episodes: int
    delays_s: list[float]
    max_delay_s: float | None
    mean_delay_s: float | None
    normal_seconds: float
    false_alarms: int
    false_alarms_per_hour: float
    inter_arrival: np.ndarray
    rssi: np.ndarray
    alarm_inter: np.ndarray
    alarm_rssi: np.ndarray
    alarm: np.ndarray


def synthetic_mqtt(
    n: int, seed: int, fault: str | None = None,
) -> list[dict]:
    """Return a list of synthetic MQTT packets.

    Args:
        n: number of packets. ``n`` must be non-negative.
        seed: RNG seed.
        fault: one of None, ``"link_degradation"``, ``"link_dropout"``.
            The fault starts at packet ``round(FAULT_AT_FRACTION * n)``.

    Returns:
        List of dicts with keys ``t``, ``inter_arrival_s``,
        ``rssi_dbm``, ``payload_bytes``, ``fault_started`` (bool),
        ``fault_kind`` (the fault name from the fault packet onwards,
        else None).

    Raises:
        ValueError: on a negative ``n`` or an unknown ``fault``.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if fault not in FAULT_KINDS:
        raise ValueError(
            f"fault must be one of {FAULT_KINDS}, got {fault!r}"
        )
    rng = np.random.default_rng(seed)
    fault_at = round(FAULT_AT_FRACTION * n) if fault is not None else n + 1
    out: list[dict] = []
    t = 0.0
    for k in range(n):
        started = k >= fault_at
        ia = NOMINAL_INTER_ARRIVAL_S + rng.normal(0.0, 0.005)
        rssi = NOMINAL_RSSI_DBM + rng.normal(0.0, 2.0)
        if started and fault == "link_degradation":
            dt = (k - fault_at) / max(n - fault_at, 1)
            ia = 0.100 + 0.050 * dt + rng.normal(0.0, 0.005)
            rssi = -60.0 - 15.0 * dt + rng.normal(0.0, 2.0)
        elif started and fault == "link_dropout":
            if rng.random() < 0.5:
                ia = 2.0 * NOMINAL_INTER_ARRIVAL_S + rng.normal(0.0, 0.005)
        ia = max(ia, 0.001)
        t = t + ia
        out.append({
            "t": float(t),
            "inter_arrival_s": float(ia),
            "rssi_dbm": float(rssi),
            "payload_bytes": int(100 + rng.integers(-10, 10)),
            "fault_started": bool(started),
            "fault_kind": fault if started else None,
        })
    return out


def _episode_starts(alarms: np.ndarray) -> np.ndarray:
    """Boolean array: True at the first True of each run of True."""
    if len(alarms) == 0:
        return alarms
    starts = np.empty_like(alarms, dtype=bool)
    starts[0] = alarms[0]
    starts[1:] = alarms[1:] & ~alarms[:-1]
    return starts


def run(
    packets: list[dict],
    cusum_kwargs: dict | None = None,
) -> LinkHealthResult:
    """Run the link health pipeline on a list of packets.

    Args:
        packets: list of dicts from :func:`synthetic_mqtt` or a real
            MQTT capture. Each dict needs ``t``, ``inter_arrival_s``,
            ``rssi_dbm``.
        cusum_kwargs: keyword arguments forwarded to
            :class:`CUSUMDriftDetector`. The default uses
            ``two_sided=False`` because both signals are made
            "increasing when worse": inter-arrival increases when the
            link degrades, and ``-rssi`` increases when the signal
            drops.

    Returns:
        :class:`LinkHealthResult`.
    """
    cusum_kwargs = {**DEFAULT_CUSUM, **(cusum_kwargs or {})}
    det_inter = CUSUMDriftDetector(**cusum_kwargs)
    det_rssi = CUSUMDriftDetector(**cusum_kwargs)

    n = len(packets)
    inter = np.full(n, np.nan)
    rssi = np.full(n, np.nan)
    alarm_inter: np.ndarray = np.zeros(n, dtype=bool)
    alarm_rssi: np.ndarray = np.zeros(n, dtype=bool)
    alarm: np.ndarray = np.zeros(n, dtype=bool)
    times: np.ndarray = np.zeros(n, dtype=float)
    fault_index = -1
    fault_kind: str | None = None

    for i, pkt in enumerate(packets):
        times[i] = float(pkt.get("t", i * 0.1))
        if fault_index < 0 and pkt.get("fault_started"):
            fault_index = i
            fault_kind = pkt.get("fault_kind")
        ia = pkt.get("inter_arrival_s")
        r = pkt.get("rssi_dbm")
        if ia is not None:
            inter[i] = float(ia)
            det_inter.update(float(ia))
            alarm_inter[i] = bool(det_inter.drift_detected)
        if r is not None:
            rssi[i] = float(r)
            det_rssi.update(-float(r))
            alarm_rssi[i] = bool(det_rssi.drift_detected)
        alarm[i] = bool(alarm_inter[i] or alarm_rssi[i])

    starts = _episode_starts(alarm)
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
        false_alarms * 3600.0 / normal_seconds
        if normal_seconds > 0
        else float("nan")
    )
    delays: list[float] = []
    if fault_index >= 0:
        t_fault = times[fault_index]
        for i in np.flatnonzero(starts):
            if i >= fault_index:
                delays.append(float(times[i] - t_fault))
    return LinkHealthResult(
        n_packets=n,
        fault_kind=fault_kind,
        fault_index=fault_index,
        n_episodes=n_episodes,
        delays_s=delays,
        max_delay_s=max(delays) if delays else None,
        mean_delay_s=float(np.mean(delays)) if delays else None,
        normal_seconds=normal_seconds,
        false_alarms=false_alarms,
        false_alarms_per_hour=fa_per_hour,
        inter_arrival=inter,
        rssi=rssi,
        alarm_inter=alarm_inter,
        alarm_rssi=alarm_rssi,
        alarm=alarm,
    )
