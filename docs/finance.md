# Finance: assembly instructions

**In plain words:** Dense-Armor is like a box of building blocks. Each block does one small job with numbers, for example fixing a wrong price or warning you when the market changes. This page is the instruction sheet: it shows which blocks to take for finance and how to snap them together, first a small build, then bigger ones.

## What is in the box

Each block has one job. Pick only the ones you need.

| Block | Its one job | Where it goes in finance |
|---|---|---|
| [`Armatura`](shields/armatura.md) | Replace a broken number with a sensible one | A wrong price tick, a missing quote |
| [CUSUM](drift/cusum.md) | Add up small differences and ring a bell when the total is too big to be chance | The average return has changed: the market is in a new mood |
| [Quantile sketch](stats/streaming.md) | Remember how big a "big" value is, without keeping all the values | How large a move is on the worst 1% of days |
| [LODA](anomaly/loda.md) | Give a score to a day described by several numbers: the higher, the stranger | A day with an unusual mix of return and volume |
| [Half-Space Trees](anomaly/hst.md) | Same job as LODA, done another way | A second opinion on a strange day |
| [Moments](stats/moments.md) | Keep the running average and spread | Rolling average and volatility |
| [Arbiter](protect/arbiter.md) | Tell a one-time glitch from a change that stays | A one-tick spike versus a price gap that stays |
| [Orca](shields/orca.md) | Clean what goes into a model and check what comes out | A protected pricing or signal model |

The four builds below use the first four blocks. Every example uses a made-up market
(random returns with a fixed seed), so you get exactly the same numbers when you run it.
Nothing here is trading advice.

## Build 1: clean the prices

**You need:** `Armatura`.

**The situation:** 500 price ticks. At position 120 the feed sent 9999 by mistake. At position 300 a quote is missing.

**Do this:**

```python
import numpy as np
from dense_armor import Armatura

rng = np.random.default_rng(7)
r = 0.0005 + 0.01 * rng.standard_normal(500)
p = 100 * np.exp(np.cumsum(r))
p[120], p[300] = 9999.0, float("nan")
clean, K, bad = Armatura(livello_ia=0.0).analizza(p)
print(bad, round(float(clean[120]), 2))
```

**You get:**

```
[120, 300] 88.96
```

`bad` is the list of the two positions that were fixed. The 9999 became 88.96, next to the 89.07 of the tick before it. From now on use `clean` and forget `p`.

## Build 2: notice that the market changed

**You need:** `CUSUMDriftDetector`. It needs the streaming extra: `pip install dense-armor[river]`.

**The situation:** 500 calm days, then the average return turns negative. No single day looks wrong. The block reads losses (the return with its sign flipped), so a rise in losses is what it watches for.

**Do this:**

```python
import numpy as np
from dense_armor.utility.drift.detector import CUSUMDriftDetector

rng = np.random.default_rng(7)
calm = 0.0005 + 0.01 * rng.standard_normal(500)
sell = -0.012 + 0.01 * rng.standard_normal(500)
loss = -np.concatenate([calm, sell])
det = CUSUMDriftDetector(reference="fixed")
print(next(i for i, v in enumerate(loss) if det.update(v).drift_detected))
```

**You get:**

```
512
```

The change starts at day 500. The bell rings at day 512. The block did not wait for one big day: it kept adding small differences until the total was too big to be chance.

## Build 3: know how big a bad day is

**You need:** `DDSketch`.

**The situation:** 2000 daily moves. You want the size that 99% of days stay below, but you do not want to keep all 2000 days.

**Do this:**

```python
import numpy as np
from dense_armor.utility.stats.quantiles import DDSketch

rng = np.random.default_rng(7)
move = np.abs(0.01 * rng.standard_normal(2000))
sk = DDSketch(alpha=0.01)
for v in move:
    sk.learn_one({"x": float(v)})
print(round(sk.quantile(0.99), 4), round(float(np.quantile(move, 0.99)), 4))
```

**You get:**

```
0.0255 0.0252
```

The first number comes from the sketch, the second from keeping every day and computing exactly. They differ by 1.2%. The sketch took one day at a time and did not store the days.

## Build 4: spot a strange day

**You need:** `LODA`.

**The situation:** each day is described by two numbers: the return and the volume (1.0 means a usual amount). The block first learns 400 ordinary days. Then you show it two new days.

**Do this:**

```python
import numpy as np
from dense_armor.utility.anomaly.loda import LODA

rng = np.random.default_rng(7)
det = LODA(n_projections=50, n_bins=10, window=200, range_init=50, seed=0)
for _ in range(400):
    det.learn_one({"ret": float(rng.normal(0, 0.01)), "vol": float(rng.normal(1.0, 0.1))})
print(round(det.score_one({"ret": 0.001, "vol": 1.0}), 3))
print(round(det.score_one({"ret": -0.05, "vol": 3.0}), 3))
```

**You get:**

```
1.367
4.837
```

The first day is ordinary and scores 1.367. The second day, a 5% drop with three times the usual volume, scores 4.837. The higher the score, the stranger the day.

## Put the builds together

The four builds work as one chain. For every new tick:

1. `Armatura` cleans the tick.
2. The clean return goes to `CUSUM` (did the average change?) and to the sketch (how big is a bad day?).
3. The clean return and the volume go to `LODA` (is today strange?).

To run all of this inside one pipeline with a time limit for each step, see the [Estimator foundation](api/roles.md) page.

## Details

- The made-up market has returns with a mean of 0.0005 and a spread of 0.01. For the change of mood the mean falls to -0.012. Real markets have heavier tails and a spread that moves, so set thresholds on real history before any use.
- Each code block is complete on its own and was run alone; the lines under "You get" are its output.
- `Armatura` also replaces some ordinary-looking values with the local median. [Armatura](shields/armatura.md) explains how many and why.
- Still to do on this page: a real price file, Half-Space Trees and Arbiter as builds, and a protected model with Orca.
