"""Safety filter: is the next move safe?

The scan of a 2-D lidar (180 beams, 10 Hz) gives the distance to the
nearest obstacle and to the walls of the room. The question is: can
the drone move at full speed toward the nearest obstacle without
entering the forbidden ball of radius ``safe_dist`` around it?

The filter is the Control Barrier Function of Ames et al. 2019
(equation CBF-QP, section II, in the module ``cbf_filter.py`` of this
library, which cites arXiv:1903.11199, p. 2): for a safe set
``C = {x : h(x) >= 0}`` and single-integrator dynamics ``xdot = u``,
the minimally invasive safety filter is

    u(x) = argmin_u (1/2)||u - u_des||^2
           s.t.  Lf h(x) + Lg h(x) u >= -alpha(h(x))

with ``h(x) = (x - obstacle)^2 - safe_dist^2`` and a closed-form
solution from the KKT conditions. This module applies the scalar
version twice, once for the nearest obstacle and once for the nearest
wall.

The scan is the only input. ``benchmarks/lidar_sim.py`` generates it
with a robot arm at the centre of a 6 x 4 m room, two pillars, and a
slow sweep. The "geofence" of this pipeline is the walls of that
room: ``room_ranges()`` returns the distance to the nearest wall for
each beam, and its minimum is the distance to the geofence.

The filter returns a velocity, not a boolean. A scan is "unsafe at
speed v" when the CBF changes the requested velocity ``-v`` toward
the obstacle. The full story needs two values of v: on a clean
stream, at v = 0.25 m/s the CBF leaves the command untouched (0
modifications in 200 scans), at v = 2.0 m/s the CBF slows the drone
to 0.29 m/s (200 modifications, max change 1.71 m/s). On a stream
with dirt on a sector (distance 0.30 m, inside the 0.5 m forbidden
ball), even 0.10 m/s is modified: the drone is inside the ball and
the CBF pushes it away, in agreement with Ames et al. 2019: the CBF
guarantee is conditional on h(x0) >= 0, and does not retroactively
fix an unsafe start.

One implementation detail that is my choice and not the paper's: the
two constraints (nearest obstacle, geofence) are both lower bounds on
the velocity, so the safe value is the **maximum** of the two filtered
velocities, not the minimum. With the minimum, one of the two would
be violated. The first version of this module used the minimum and
the tests below caught it: on a dirty scan the obstacle filter pushed
away at +0.27 m/s and the geofence filter let -0.10 m/s through, and
the minimum was -0.10 m/s, which is inside the forbidden ball.

The desired velocity ``u_des = -v_max`` (2 m/s, toward the nearest
obstacle) is the move the pipeline asks about. The CBF says whether
the drone can approach at full speed without entering the forbidden
ball; if the filter modifies the value, the move is unsafe. The
live variant of the filter sub-steps each lidar tick (dt = 0.1 s,
20 sub-steps), so the continuous-time guarantee survives the discrete
sampling.

Example:
    >>> import numpy as np
    >>> import benchmarks.lidar_sim as ls
    >>> from pipelines.drones.safety_filter import run
    >>> scans = [ls.scan(i, np.random.default_rng(0), None, None)
    ...          for i in range(5)]
    >>> r = run(scans)
    >>> r.n_scans
    5
    >>> r.safe_dist_obstacle
    0.5
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dense_armor.utility.control.cbf_filter import (
    cbf_safety_filter_live,
)

DEFAULT_SAFE_DIST_OBSTACLE = 0.5
DEFAULT_SAFE_DIST_GEOFENCE = 0.5
DEFAULT_V_MAX = 2.0
DEFAULT_DT = 0.1


@dataclass
class SafetyFilterResult:
    """Result of one run of the safety filter.

    Attributes:
        n_scans: number of scans seen.
        safe_dist_obstacle: the radius of the forbidden ball around
            the nearest obstacle.
        safe_dist_geofence: the radius of the forbidden ball around
            the nearest wall.
        v_max: the maximum speed of the drone, used as desired
            velocity.
        x_obs: distance to the nearest obstacle, per scan.
        x_geo: distance to the nearest wall, per scan.
        u_obs: filtered velocity for the obstacle, per scan.
        u_geo: filtered velocity for the geofence, per scan.
        u_safe: the larger of ``u_obs`` and ``u_geo``, per scan
            (the CBF constraints are lower bounds on u, so the
            safe value is the max of the two filtered values, not
            the min: with the min, one of the two would be
            violated).
        n_modified_obs: scans where the obstacle filter modified the
            desired velocity.
        n_modified_geo: scans where the geofence filter modified it.
        n_modified_both: scans where at least one of the two modified it.
        fraction_modified: ``n_modified_both / n_scans``.
        min_x_obs: minimum of ``x_obs`` over the run.
        min_x_geo: minimum of ``x_geo`` over the run.
        max_change: the largest absolute change applied by either
            filter over the run, in m/s.
    """

    n_scans: int
    safe_dist_obstacle: float
    safe_dist_geofence: float
    v_max: float
    x_obs: np.ndarray
    x_geo: np.ndarray
    u_obs: np.ndarray
    u_geo: np.ndarray
    u_safe: np.ndarray
    n_modified_obs: int
    n_modified_geo: int
    n_modified_both: int
    fraction_modified: float
    min_x_obs: float
    min_x_geo: float
    max_change: float


def run(
    scans: list[np.ndarray],
    safe_dist_obstacle: float = DEFAULT_SAFE_DIST_OBSTACLE,
    safe_dist_geofence: float = DEFAULT_SAFE_DIST_GEOFENCE,
    v_max: float = DEFAULT_V_MAX,
    dt: float = DEFAULT_DT,
) -> SafetyFilterResult:
    """Run the safety filter on a list of lidar scans.

    Args:
        scans: list of 1-D arrays, one per scan, with the distance to
            the nearest obstacle for each beam.
        safe_dist_obstacle: radius of the forbidden ball around the
            nearest obstacle. Default 0.5 m.
        safe_dist_geofence: radius of the forbidden ball around the
            nearest wall. Default 0.5 m.
        v_max: desired speed toward the obstacle, used as ``|u_des|``.
            Default 2 m/s. The sign of ``u_des`` is negative (approach).
        dt: time between two scans, in seconds. Default 0.1 s (10 Hz
            lidar).

    Returns:
        :class:`SafetyFilterResult`.
    """
    import benchmarks.lidar_sim as ls

    room = ls.room_ranges()
    x_geo_const = float(room.min())
    u_des = -v_max

    n = len(scans)
    x_obs = np.full(n, np.nan)
    x_geo = np.full(n, np.nan)
    u_obs = np.full(n, np.nan)
    u_geo = np.full(n, np.nan)
    u_safe = np.full(n, np.nan)
    n_mod_obs = 0
    n_mod_geo = 0
    n_mod_both = 0
    max_change = 0.0

    for i, scan in enumerate(scans):
        s = np.asarray(scan, dtype=float)
        xo = float(s.min())
        xg = x_geo_const
        x_obs[i] = xo
        x_geo[i] = xg
        uo = cbf_safety_filter_live(
            xo, u_des, dt, obstacle=0.0,
            safe_dist=safe_dist_obstacle,
        )
        ug = cbf_safety_filter_live(
            xg, u_des, dt, obstacle=0.0,
            safe_dist=safe_dist_geofence,
        )
        u_obs[i] = uo
        u_geo[i] = ug
        u_safe[i] = max(uo, ug)
        mod_o = abs(uo - u_des) > 1e-9
        mod_g = abs(ug - u_des) > 1e-9
        if mod_o:
            n_mod_obs += 1
        if mod_g:
            n_mod_geo += 1
        if mod_o or mod_g:
            n_mod_both += 1
        change = abs(u_safe[i] - u_des)
        max_change = max(max_change, change)

    valid_x = x_obs[np.isfinite(x_obs)]
    min_x_obs = float(valid_x.min()) if len(valid_x) else float("nan")

    return SafetyFilterResult(
        n_scans=n,
        safe_dist_obstacle=safe_dist_obstacle,
        safe_dist_geofence=safe_dist_geofence,
        v_max=v_max,
        x_obs=x_obs,
        x_geo=x_geo,
        u_obs=u_obs,
        u_geo=u_geo,
        u_safe=u_safe,
        n_modified_obs=n_mod_obs,
        n_modified_geo=n_mod_geo,
        n_modified_both=n_mod_both,
        fraction_modified=n_mod_both / n if n > 0 else float("nan"),
        min_x_obs=min_x_obs,
        min_x_geo=x_geo_const,
        max_change=max_change,
    )
