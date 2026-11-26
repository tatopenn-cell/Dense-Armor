# Uncertainty: scores for a model that says how sure it is

[← Metrics](../index.md)

## The map of uncertainty

Click a page to open it, or a job on the left (or the button at the top) to go back to the [map of the library](../../index.md#the-library-folder-by-folder).

<object type="image/svg+xml" data="../../assets/maps/uncertainty.svg" style="width:100%"></object>

A model that only gives a number is hard to trust: it does not say how sure it is. A model that also gives an interval ("tomorrow's return will be between −1.5% and +1.5%"), or a mean and a variance, can be checked: after each day the true value arrives, and an uncertainty metric updates its score with that one pair and forgets it. An interval is honest if the truth falls inside it as often as promised, and useful if it is not wider than it needs to be.

## A first example: the daily return of a stock index

A risk desk quotes for tomorrow's return the band from −1.5% to +1.5% (as fractions, −0.015 to 0.015). Six days later the true returns are known. Two metrics read the six pairs of a return and its band, and a third reads the same returns against a forecast with mean 0 and variance 0.0001, that is a standard deviation of 1%.

```python
from dense_armor.core.evaluation.metrics import GaussianNLL, IntervalCoverage, MeanIntervalWidth

stock_return = [0.004, -0.012, 0.008, 0.021, -0.003, -0.018]
cov, wid, nll = IntervalCoverage(), MeanIntervalWidth(), GaussianNLL()
for r in stock_return:
    cov.update(r, (-0.015, 0.015))
    wid.update(r, (-0.015, 0.015))
    nll.update(r, (0.0, 0.0001))
print(round(cov.get(), 3), round(wid.get(), 3), round(nll.get(), 4))
```

```
0.667 0.03 -2.8546
```

Four of the six returns fell inside the band, 0.004, −0.012, 0.008 and −0.003, and two fell outside, 0.021 and −0.018: the coverage is 4/6 = 0.667. The band is 0.03 wide every day, so the mean width is 0.03. The Gaussian negative log-likelihood is the mean of $\tfrac12 \ln(2\pi\sigma^2) + (y-\mu)^2 / (2\sigma^2)$: the first term is −3.6862 for every day and the second has the mean 0.8317, so the score is −2.8546. Each page below explains one of the metrics with the same six returns.

## The uncertainty metrics, one page each

| page | classes | what it is for |
|---|---|---|
| [Interval coverage](coverage.md) | `IntervalCoverage` | how often the truth falls inside the interval |
| [Interval width](width.md) | `MeanIntervalWidth` | how wide the intervals are |
| [Gaussian NLL](gaussian_nll.md) | `GaussianNLL` | a mean and a variance scored at once |

The metrics that read a single number are in [Regression](../regression/index.md).

---

## Details

### Choices

- **The example** uses the same six returns on the three pages.
- **The map** is drawn with the library by `docs/assets/maps/make_uncertainty_map.py`; the classes are the names of the catalog of `usable`.
