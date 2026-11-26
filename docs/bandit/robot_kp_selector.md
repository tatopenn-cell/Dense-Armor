# RobotKpSelector: choose a controller gain with UCB1

[← Bandits](index.md)

A robot arm follows a trajectory with a proportional gain `kp`. Too low and it lags, too high and it overshoots. `RobotKpSelector` treats the candidate gains as the arms of a bandit: each round it picks a gain with UCB1, runs a tracking episode on a simulated arm with that gain and rewards the policy with the negated mean tracking error, so a lower error is a higher reward.

## 1. A first example: three gains

Three candidate gains, nine rounds. Each round runs a short episode on the simulated arm of the library.

```python
from dense_armor.core.online.bandit import RobotKpSelector

sel = RobotKpSelector(arms=[0.5, 8.0, 30.0], period_s=0.5, n_cycles=2, seed=0)
for _ in range(9):
    sel.run_round()
print(sel.best_gain(), dict(sel.counts_))
```

```
8.0 {0.5: 3, 8.0: 3, 30.0: 3}
```

After the nine rounds the best gain is 8.0, and each gain was tried 3 times, because UCB1 plays every arm once before it compares them and the rounds are few. `run_round()` runs one episode and updates the policy, `best_gain()` returns the gain with the lowest mean error so far and `counts_` is how many rounds each gain had.

## 2. What it does

Each round the policy picks a gain, runs a tracking episode with that gain on the library's `SyntheticArm`, and updates the bandit with the negated mean tracking error (a higher reward is a lower error). The bandit is UCB1 (see [UCB1 and UCB1-Tuned](ucb.md)).

On the benchmark of the index, UCB1 picked kp = 8.0 on all 20 seeds, with a median regret against the best fixed gain of 0.0 and a median time per decision of 0.54 s, dominated by the simulation.

**Source:** Auer, P., Cesa-Bianchi, N., Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. Machine Learning 47(2-3), 235-256. UCB1, Figure 1 and Theorem 1, page 3.

## 3. The parameters

| Parameter | Meaning | Default |
|---|---|---|
| `arms` | the candidate gains, positive floats | none |
| `urdf_path` | a URDF readable by `SyntheticArm` | `test/fixtures/urdf/panda.urdf` |
| `period_s` | period of the reference trajectory, in seconds | 0.5 |
| `rate_hz` | sampling rate of the reference, in Hz | 50.0 |
| `n_cycles` | number of cycles of the reference | 2 |
| `seed` | seed of the bandit | 0 |

`RobotKpSelector` raises `ValueError` if `arms` is empty or contains a non-positive value. The methods that every policy shares are on the [shared methods page](policy.md).

## API reference

::: dense_armor.core.online.bandit.robot.RobotKpSelector

---

## Details

### Choices

- **No finance example and no 3D figure on this page:** the class runs a simulation of a robot arm, and one round takes about half a second, so a surface of many runs is not drawn. The behaviour on a stream of success rates is on the other pages.
- **Where it lives:** `dense_armor.core.online.bandit.robot`.
