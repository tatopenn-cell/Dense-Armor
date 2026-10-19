"""Battery endurance: how many minutes of flight are left, and how sure?

Three online models run together, one per sample:

1. **V-hat**: an online regression of the battery voltage on the
   current, the time, and their product. The interval is the adaptive
   conformal interval of ``AdaptiveConformalRegressor`` (Gibbs and
   Candes 2021, equation 2, cited in the class docstring), wrapping a
   ``LinearRegression`` with the ``AdaGrad`` optimizer.
2. **P**: the measured power ``V * I``. The nine-term model of
   Alyassi et al. 2017 (section IV, equation 5, page 5) is also
   computed, with the coefficients of the 3DR Solo (page 5), and
   reported as ``p_alyassi`` in the result for comparison. On this
   synthetic stream it gives about 436 W, three times the measured
   150 W: the intercept b9 = 433.9 is the hover power of the 3DR
   Solo and not of the synthetic drone, and the drift in the other
   eight features is too small for a fit to correct the intercept.
   Alyassi is cited as the structural reference (equation 5), not as
   a calibration.
3. **t-remaining**: the remaining flight time, computed as
   ``E_residual / P_hat``, where ``E_residual = capacity_wh *
   (SoC - soc_min)`` and ``SoC = (V - V_min) / (V_full - V_min)``.
   The distribution of the values produced is accumulated in a
   ``TDigest`` (Dunning 2019) and read at quantiles 0.1, 0.5, 0.9.

The parameters that the paper does not give are my choice and are
listed in Details of the page: ``capacity_wh = 63`` (a 3S 5000 mAh
LiPo), ``v_full = 12.6``, ``v_min = 9.0``, ``soc_min = 0.10``.

Example:
    >>> from pipelines.drones.io import synthetic_flight
    >>> from pipelines.drones.battery_endurance import run
    >>> r = run(list(synthetic_flight(n=400, seed=0)))
    >>> r.n_samples
    400
    >>> r.capacity_wh
    63.0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dense_armor.roles import AdaptiveConformalRegressor
from dense_armor.utility.linear.glm import LinearRegression
from dense_armor.utility.optim.adagrad import AdaGrad
from dense_armor.utility.stats.quantiles import TDigest

DEFAULT_CAPACITY_WH = 63.0
DEFAULT_V_FULL = 12.6
DEFAULT_V_MIN = 9.0
DEFAULT_SOC_MIN = 0.10

V_FEATURE_KEYS = ["current_a", "t", "current_t", "one"]
BETA_SOLO = {
    "v_xy_norm": -1.526,
    "a_xy_norm": 3.934,
    "v_xy_a_xy": 0.968,
    "v_z_abs": 18.125,
    "a_z_abs": 96.613,
    "v_z_a_z": -1.085,
    "payload_kg": 0.220,
    "v_xy_dot_w_xy": 1.332,
    "one": 433.9,
}


@dataclass
class BatteryEnduranceResult:
    """Result of one run of the battery endurance pipeline.

    Attributes:
        n_samples: number of samples seen.
        capacity_wh: battery capacity in Wh, from the constructor.
        v_full: voltage at full charge, in V.
        v_min: voltage at empty charge, in V.
        soc_min: state of charge below which the drone should land.
        v_pred: predicted voltage, per sample (n,).
        v_lo: lower bound of the voltage interval, per sample.
        v_hi: upper bound of the voltage interval, per sample.
        p_pred: measured power V * I, per sample (n,).
        p_alyassi: the nine-term model of Alyassi et al. 2017 with
            the coefficients of the 3DR Solo, per sample (n,).
        p_alyassi_median: median of p_alyassi over the stream.
        t_remaining: remaining flight time, in seconds, per sample.
        t_remaining_q10: 10th percentile of t_remaining over the stream.
        t_remaining_q50: median.
        t_remaining_q90: 90th percentile.
        coverage: fraction of true voltages inside the interval.
        mean_width: mean width of the interval.
    """

    n_samples: int
    capacity_wh: float
    v_full: float
    v_min: float
    soc_min: float
    v_pred: np.ndarray
    v_lo: np.ndarray
    v_hi: np.ndarray
    p_pred: np.ndarray
    p_alyassi: np.ndarray
    p_alyassi_median: float
    t_remaining: np.ndarray
    t_remaining_q10: float
    t_remaining_q50: float
    t_remaining_q90: float
    coverage: float
    mean_width: float


def _p_features(sample: dict) -> dict:
    """The nine input features of Alyassi et al. 2017, equation 5, page 5."""
    gps_v = np.asarray(sample.get("gps_v", np.zeros(3)), dtype=float)
    acc = np.asarray(sample.get("acc", np.zeros(3)), dtype=float)
    acc = acc - np.array([0.0, 0.0, 9.81])
    wind = np.asarray(sample.get("wind_xy", np.zeros(2)), dtype=float)
    v_xy_norm = float(np.linalg.norm(gps_v[:2]))
    a_xy_norm = float(np.linalg.norm(acc[:2]))
    v_z_abs = float(abs(gps_v[2]))
    a_z_abs = float(abs(acc[2]))
    payload = float(sample.get("payload_kg", 0.0))
    v_xy_dot_w_xy = float(gps_v[0] * wind[0] + gps_v[1] * wind[1])
    return {
        "v_xy_norm": v_xy_norm,
        "a_xy_norm": a_xy_norm,
        "v_xy_a_xy": v_xy_norm * a_xy_norm,
        "v_z_abs": v_z_abs,
        "a_z_abs": a_z_abs,
        "v_z_a_z": v_z_abs * a_z_abs,
        "payload_kg": payload,
        "v_xy_dot_w_xy": v_xy_dot_w_xy,
        "one": 1.0,
    }


def _soc(v: float, v_full: float, v_min: float) -> float:
    """State of charge from voltage, linear between v_min and v_full."""
    return float(np.clip((v - v_min) / (v_full - v_min), 0.0, 1.0))


def run(
    stream: list[dict],
    capacity_wh: float = DEFAULT_CAPACITY_WH,
    v_full: float = DEFAULT_V_FULL,
    v_min: float = DEFAULT_V_MIN,
    soc_min: float = DEFAULT_SOC_MIN,
) -> BatteryEnduranceResult:
    """Run the battery endurance pipeline on a list of samples.

    The voltage model is an AdaptiveConformalRegressor wrapping a
    LinearRegression; the power is the measured V * I; the remaining
    time is E_residual / P.

    Args:
        stream: list of dicts from
            :func:`pipelines.drones.io.synthetic_flight`. Each dict
            needs ``t``, ``battery_v``, ``current_a``, ``gps_v``,
            ``acc``, and may carry ``payload_kg`` and ``wind_xy``.
        capacity_wh: battery capacity in Wh. Default 63.
        v_full: voltage at full charge, in V. Default 12.6.
        v_min: voltage at empty charge, in V. Default 9.0.
        soc_min: state of charge below which the drone should land.

    Returns:
        :class:`BatteryEnduranceResult`.
    """
    v_model = AdaptiveConformalRegressor(
        LinearRegression(optimizer=AdaGrad(lr=0.01), feature_keys=V_FEATURE_KEYS),
        alpha=0.1, gamma=0.005, window=100,
    )
    t_digest = TDigest(delta=100.0)

    n = len(stream)
    v_pred = np.full(n, np.nan)
    v_lo = np.full(n, np.nan)
    v_hi = np.full(n, np.nan)
    p_pred = np.full(n, np.nan)
    p_alyassi = np.full(n, np.nan)
    t_remaining = np.full(n, np.nan)
    in_interval = 0
    total_width = 0.0
    n_valid = 0

    for i, sample in enumerate(stream):
        t = float(sample.get("t", i * 0.01))
        v_true = float(sample.get("battery_v", float("nan")))
        current = float(sample.get("current_a", 0.0))

        x_v = {
            "current_a": current,
            "t": t,
            "current_t": current * t,
            "one": 1.0,
        }
        v_hat = v_model.predict_one(x_v)
        lo, hi = v_model.predict_interval(x_v)
        v_pred[i] = v_hat
        v_lo[i] = lo
        v_hi[i] = hi
        if np.isfinite(v_true):
            v_model.learn_one(x_v, v_true)
            if lo <= v_true <= hi:
                in_interval += 1
            total_width += hi - lo
            n_valid += 1

        p_measured = v_true * current if np.isfinite(v_true) else float("nan")
        p_pred[i] = p_measured

        f = _p_features(sample)
        p_alyassi[i] = float(sum(BETA_SOLO[k] * f[k] for k in BETA_SOLO))

        soc = _soc(v_true if np.isfinite(v_true) else v_full, v_full, v_min)
        e_residual_wh = capacity_wh * max(soc - soc_min, 0.0)
        p_safe = max(p_measured, 50.0) if np.isfinite(p_measured) else 300.0
        t_rem = (e_residual_wh / p_safe) * 3600.0
        t_remaining[i] = t_rem
        if np.isfinite(t_rem) and t_rem > 0:
            _ = t_digest.update({"x": t_rem})

    valid_t = t_remaining[np.isfinite(t_remaining) & (t_remaining > 0)]
    if len(valid_t) > 0:
        q10 = float(t_digest.quantile(0.1))
        q50 = float(t_digest.quantile(0.5))
        q90 = float(t_digest.quantile(0.9))
    else:
        q10 = q50 = q90 = float("nan")

    valid_a = p_alyassi[np.isfinite(p_alyassi)]
    p_alyassi_med = float(np.median(valid_a)) if len(valid_a) > 0 else float("nan")

    coverage = in_interval / n_valid if n_valid > 0 else float("nan")
    mean_width = total_width / n_valid if n_valid > 0 else float("nan")

    return BatteryEnduranceResult(
        n_samples=n,
        capacity_wh=capacity_wh,
        v_full=v_full,
        v_min=v_min,
        soc_min=soc_min,
        v_pred=v_pred,
        v_lo=v_lo,
        v_hi=v_hi,
        p_pred=p_pred,
        p_alyassi=p_alyassi,
        p_alyassi_median=p_alyassi_med,
        t_remaining=t_remaining,
        t_remaining_q10=q10,
        t_remaining_q50=q50,
        t_remaining_q90=q90,
        coverage=coverage,
        mean_width=mean_width,
    )
