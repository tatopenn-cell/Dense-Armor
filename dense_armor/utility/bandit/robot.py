"""Robot test: choose the gain ``kp`` of a kinematic controller.

The bandit sees a stream of tracking episodes of the same reference
trajectory, each with a different ``kp``, and learns the gain that gives
the smallest mean absolute tracking error. The controller and the
dataset are the library's own: :class:`SyntheticArm` for the reference
and :func:`kinematic_tracking_controller` for the control law.
"""

from collections.abc import Sequence

import numpy as np

from dense_armor.roles import Root
from dense_armor.utility.bandit.ucb import UCB1
from dense_armor.utility.control.kinematic_controller import (
    kinematic_tracking_controller,
)
from dense_armor.utility.datasets import SyntheticArm


def _episode(
    ds: SyntheticArm,
    kp: float,
    dt_factor: int = 100,
) -> float:
    """Run one tracking episode with gain ``kp``; return the mean error.

    Args:
        ds: the dataset.
        kp: the proportional gain.
        dt_factor: sub-steps per sample of the reference.

    Returns:
        The mean absolute tracking error over the episode, negated
        (so that a higher number means a better gain, as the bandit
        expects a reward).
    """
    refs = []
    for sig, _ in ds.stream():
        t = float(sig.t or 0.0)
        q_ref, qd_ref, _ = ds.trajectory(t)
        refs.append((t, np.asarray(q_ref), np.asarray(qd_ref)))
    q_sim = refs[0][1].copy()
    errs = []
    for i in range(len(refs) - 1):
        t0, q0, qd0 = refs[i]
        t1, q1, qd1 = refs[i + 1]
        dt = (t1 - t0) / dt_factor
        for k in range(dt_factor):
            a = k / dt_factor
            q_ref = (1.0 - a) * q0 + a * q1
            qd_ref = (1.0 - a) * qd0 + a * qd1
            u = kinematic_tracking_controller(q_sim, q_ref, qd_ref, kp)
            q_sim = q_sim + np.asarray(u) * dt
            errs.append(float(np.mean(np.abs(q_sim - q_ref))))
    return -float(np.mean(errs))


class RobotKpSelector(Root):
    """Choose ``kp`` among a set of values by UCB1 on a robot stream.

    Each round the policy picks a gain, runs a tracking episode with that
    gain on the library's ``SyntheticArm``, and updates the bandit with
    the negated mean tracking error (a higher reward is a lower error).

    Args:
        arms: the candidate gains, positive floats.
        urdf_path: a URDF readable by ``SyntheticArm``.
        period_s: period of the reference trajectory, in seconds.
        rate_hz: sampling rate of the reference, in Hz.
        n_cycles: number of cycles of the reference.
        seed: seed of the bandit.

    Raises:
        ValueError: if ``arms`` is empty or contains a non-positive
            value.

    References:
        Auer, P., Cesa-Bianchi, N., Fischer, P. (2002). Finite-time
            analysis of the multiarmed bandit problem. Machine Learning
            47(2-3), 235-256. UCB1, Figure 1 and Theorem 1, page 3.
    """

    budget_s = 1.0
    memory_class = "O(n_arms)"

    def __init__(
        self,
        arms: Sequence[float],
        urdf_path: str = "test/fixtures/urdf/panda.urdf",
        period_s: float = 0.5,
        rate_hz: float = 50.0,
        n_cycles: int = 2,
        seed: int = 0,
    ) -> None:
        """Initialise the selector.

        Args:
            arms: the candidate gains.
            urdf_path: path to a URDF file.
            period_s: period of the reference trajectory.
            rate_hz: sampling rate of the reference.
            n_cycles: number of cycles.
            seed: seed of the bandit.

        Raises:
            ValueError: if ``arms`` is empty or contains a non-positive
                value.
        """
        super().__init__()
        if not arms:
            raise ValueError("arms must be non-empty")
        for a in arms:
            if float(a) <= 0.0:
                raise ValueError("gains must be positive")
        self.arms = tuple(float(a) for a in arms)
        self.urdf_path = urdf_path
        self.period_s = float(period_s)
        self.rate_hz = float(rate_hz)
        self.n_cycles = int(n_cycles)
        self.seed = int(seed)
        self.bandit_ = UCB1(arms=list(self.arms), seed=self.seed)
        self.ds_ = SyntheticArm(
            self.urdf_path,
            period_s=self.period_s,
            rate_hz=self.rate_hz,
            n_cycles=self.n_cycles,
            seed=self.seed,
        )
        self.counts_: dict[float, int] = {a: 0 for a in self.arms}
        self.rewards_: dict[float, float] = {a: 0.0 for a in self.arms}
        self.n_seen_ = 0

    def run_round(self) -> float:
        """Choose a gain, run an episode and update the bandit.

        Returns:
            The chosen gain.
        """
        kp = float(self.bandit_.pull())
        reward = _episode(self.ds_, kp)
        self.bandit_.update(kp, reward)
        self.counts_[kp] = self.counts_.get(kp, 0) + 1
        n = self.counts_[kp]
        m = self.rewards_.get(kp, 0.0)
        self.rewards_[kp] = m + (reward - m) / n
        self.n_seen_ += 1
        return kp

    def best_gain(self) -> float:
        """Return the gain with the highest running mean reward.

        Returns:
            The best gain seen so far.
        """
        return max(self.arms, key=lambda a: self.rewards_.get(a, -np.inf))
