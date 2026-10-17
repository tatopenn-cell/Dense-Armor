# Active learning: ask for the label only when unsure

A robot observes a stream of samples but the true label of a sample is
expensive to obtain: an operator, a test bench, a slow measurement. Active
learning decides which samples to label, so a classifier can learn on a
stream with a limited labelling budget. The library provides four
strategies, from the naive (label at random) to the one that balances the
budget while asking the most uncertain samples.

## 1. The budget

A labelling budget `B` is a fraction between 0 and 1: `B = 0.1` means that at
most 10 % of the incoming samples can be labelled. The strategies differ in
*which* 10 % to pick.

```python
import numpy as np
from dense_armor.utility.active import RandomStrategy

rng = np.random.default_rng(0)
strategy = RandomStrategy(budget=0.1, seed=0)
for _ in range(1000):
    strategy.should_label({"x": float(rng.random())}, None)
print(round(strategy.n_labelled_ / strategy.n_seen_, 3))
