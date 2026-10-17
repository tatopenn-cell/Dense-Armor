# Bandits: try one option, learn from the reward

A robot that tunes a controller gain has a small set of options and a reward
after every try. Which gain to try next is a bandit problem: try the one that
looks best, or try the ones you know less about, and learn from both.

This page shows the seven bandit policies of the library, from the simplest
(explore with probability epsilon) to the contextual one (use the context to
pick a different arm for a different situation), plus the offline evaluation
of a policy on logged data.

## 1. The problem, in one line

You have K arms. At each step you pick one arm, see a reward, and update what
you know. You want the sum of rewards to be close to the sum of rewards of the
best arm in hindsight.

## 2. Epsilon-greedy: explore with a fixed probability

With probability `1 - epsilon` play the arm with the highest running mean;
otherwise play a uniformly random arm. The policy plays every arm once first,
so an arm has at least one sample before the comparison.

```python
import numpy as np
from dense_armor.utility.bandit import EpsilonGreedy

rng = np.random.default_rng(0)
p = EpsilonGreedy(arms=["cold", "warm", "hot"], epsilon=0.1, seed=0)
for _ in range(500):
    a = p.pull()
    r = float(rng.binomial(1, {"cold": 0.2, "warm": 0.5, "hot": 0.8}[a]))
    _ = p.update(a, r)
print(p.pull())
