# Finance: assembly instructions

**In plain words:** Dense-Armor is a box of building pieces that watch a stream of numbers one reading at a time, and a stream of prices and trades is exactly that kind of stream. This page is the instruction sheet: it shows which pieces to pick for prices and trades, and how to snap them together, from the simplest build to a bigger one.

Every piece in this library does one small job on numbers that arrive one by one: clean a
broken value, notice a slow change, give a strange reading a score, keep a running summary.
Finance produces such numbers all day: prices, returns, volumes, order sizes. The pieces
need no changes; only the names of the signals change. An algorithm page tells you how a
piece works inside. This page tells you where it goes.

All the numbers below come from a synthetic market (random returns with fixed seeds), so
that every result can be re-run. Nothing here is trading advice.

## 1. The pieces, and what each does for finance

| Piece | What it does | In finance it can be |
|---|---|---|
| [`Armatura`](shields/armatura.md) | replaces broken values in one series | a bad tick, a missing quote |
| [CUSUM](drift/cusum.md) | adds up small differences until the total is too big to be chance | a change of regime: the average return has moved |
| [Quantile sketch](stats/streaming.md) | keeps a percentile of a stream without storing it | the size of a move that 99% of days stay below |
| [LODA](anomaly/loda.md) | scores how strange a reading with several numbers is | a day with an odd mix of return and volume |
| [Half-Space Trees](anomaly/hst.md) | the same job as LODA with a different method | the same, as a second opinion |
| [Moments](stats/moments.md) | running mean and spread | a rolling average and volatility |
| [Arbiter](protect/arbiter.md) | tells a single glitch from a real change | a one-tick spike versus a gap that stays |
| [Orca](shields/orca.md) | cleans what goes into a model and checks what comes out | a protected pricing or signal model |

## 2. Build one: clean the price feed

A price series of 500 ticks with one absurd tick (9999) at position 120 and one missing
quote at position 300.

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

```
[120, 300] 88.96
```

`bad` lists the two replaced positions, and the 9999 became 88.96, next to the 89.07 of the
tick before it. Every later piece in this page should read `clean`, not `p`.

## 3. Build two: notice that the market changed

Returns are calm for 500 days, then the average turns negative. The piece reads the losses
(minus the return), so a rise in losses is what fills it up.

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

```
512
```

The change starts at day 500 and the alarm comes at day 512. No single day looked wrong; the
piece noticed that the small differences kept adding up in the same direction. The detector
needs the streaming extra: `pip install dense-armor[river]`.

## 4. Build three: a daily risk level without storing history

The 99th percentile of the size of a daily move, kept by a sketch that never stores the days.

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

```
0.0255 0.0252
```

The first number is the sketch, the second is the exact percentile of the 2000 values: they
differ by 1.2%. The sketch is updated one day at a time and does not store the days.

## 5. Build four: a strange day, judged on several numbers together

A day is described by its return and its volume (traded amount relative to normal). The
detector learns 400 ordinary days, then scores two new ones. The higher the score, the stranger.

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

```
1.367
4.837
```

An ordinary day scores 1.367. A day with a 5% drop and three times the usual volume scores
4.837. The detector judges the return and the volume together, in one score.

## 6. How the builds fit together

The four builds are one chain, read from left to right on every new tick:

1. `Armatura` cleans the tick.
2. The cleaned return goes to CUSUM (has the average moved?) and to the sketch (how big is a bad day?).
3. The cleaned return and volume go to LODA (is today's pair strange?).

A pipeline that asks all three questions per tick, with a time limit per step, is described
in the [Estimator foundation](api/roles.md) page.

## Details

- Data: synthetic returns, normal with a mean of 0.0005 and a spread of 0.01, and a fall to a
  mean of -0.012 for the regime change. Real markets have heavier tails and changing
  volatility, so thresholds should be set on real history before any use.
- Each block is self-contained and was run alone; the printed lines are its output.
- `Armatura` also replaces some ordinary-looking samples by the local median, see
  [Armatura](shields/armatura.md) for how many and why.
- The next steps for this page (not done yet): a real price file, Half-Space Trees and
  Arbiter shown as builds, and a protected model with Orca.
