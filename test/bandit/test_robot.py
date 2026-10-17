"""Robot test: UCB chooses the gain kp on SyntheticArm."""


from dense_armor.utility.bandit import RobotKpSelector


def test_ucb_finds_the_best_gain():
    """UCB should pick kp=8.0 (the best of 0.5, 8.0, 30.0) on most seeds."""
    arms = [0.5, 8.0, 30.0]
    picks = []
    for seed in range(5):
        sel = RobotKpSelector(arms=arms, period_s=0.5, n_cycles=2, seed=seed)
        for _ in range(30):
            sel.run_round()
        picks.append(sel.best_gain())
    counts = {a: picks.count(a) for a in arms}
    assert counts[8.0] >= 3, f"UCB picks: {counts}"


def test_robot_counts_sum_to_rounds():
    sel = RobotKpSelector(arms=[0.5, 8.0], n_cycles=1, seed=0)
    for _ in range(6):
        sel.run_round()
    assert sum(sel.counts_.values()) == 6


def test_robot_rejects_bad_arms():
    import pytest
    with pytest.raises(ValueError):
        RobotKpSelector(arms=[])
    with pytest.raises(ValueError):
        RobotKpSelector(arms=[-1.0])
