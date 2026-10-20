# Regime: has the market moved to a different volatility regime?

Volatility is not constant. A market alternates calm periods and turbulent periods, and a risk model or an execution schedule tuned for one regime is wrong in the other. This page answers two questions with a streaming detector: **when** did the market change regime, and **how many false alarms** does the detector raise per year of calm data.

The detector is the Bayesian online changepoint of Adams and MacKay (2007, arXiv:0710.3742). It keeps, for every possible length of the current run of stable data, the probability that the run has that length. When the probability mass moves to short runs, a changepoint has just happened. The pipeline turns the daily close of an index into log returns and feeds them to the detector.

## 1. Read a close series and run the detector

```python
from pathlib import Path
from pipelines.finance.regime import run_pipeline

gspc = Path("test/fixtures/finance/GSPC.csv")
r = run_pipeline(gspc)
print(r.n_returns, r.n_alarms, round(r.alarm_rate_per_year, 2))
```

```
1253 5 0.81
```

The file is the S&P 500 daily close of the fixtures, 1,253 log returns. The detector raises 5 alarms over the whole series; 2 of them fall in the first 2.48 years, the calm part, and that is 0.81 alarms per year of calm data. `alarm_rate_per_year` counts the alarms before the first half of the series, so a real series and a synthetic one are compared on the same footing.

## 2. A planted change, and the delay in samples

```python
from pipelines.finance.regime import RegimePipeline, synthetic_two_regime

x, change = synthetic_two_regime(n=1000, seed=0, change_at=500)
r = RegimePipeline().run(x, change_at=change)
print(r.first_alarm_after_change, r.delay_samples)
```

```
505 5
```

The synthetic stream has 500 returns at volatility 0.01 and 500 at 0.03. The first alarm after the change is at sample 505, five samples after the true change at 500. The detector does not know the true change: it sees the returns and reports the first sample whose short-run mass crosses the threshold.

## 3. Twenty seeds, median and interquartile range

```python
from pipelines.finance.regime import run_synthetic

out = run_synthetic(n_seeds=20, n=1000, change_at=500)
print(out["delay_median"], out["delay_iqr"], round(out["rate_median"], 2))
```

```
5.0 0.0 0.5
```

Over 20 seeds the detection delay is 5 samples median, 0 IQR: the detector finds the change on the fifth sample after it, every seed. The alarm rate on the calm part is 0.5 per year median. A single seed is not a comparison; the twenty seeds are the comparison.

## 4. The detector, in one paragraph

The detector keeps a run length distribution, the probability that the current stable run has each length. With a constant hazard `H(tau) = 1 / lam` (equation 4, page 2 of the paper), the growth and changepoint probabilities of Algorithm 1 (pages 3-4) are

```text
growth:      P(r_t = r + 1, x_1:t) = P(r_t-1 = r, x_1:t-1) * pi_t(r) * (1 - H)
changepoint: P(r_t = 0, x_1:t)     = sum_r P(r_t-1 = r, x_1:t-1) * pi_t(r) * H
```

where `pi_t(r)` is the predictive probability of the new sample given the data since the last changepoint, a Student-t under a Normal-Inverse-Gamma model (section 2.3, pages 3-4). The alarm is the mass on the shortest `mass_window` run lengths, not the exact mass on run length zero: with a constant hazard, the mass on run length zero is about `1 / lam` on every step, before and after a changepoint, because the growth term moves the mass to run length one. The `mass_window` is this implementation's choice.

## Details

### References

- Adams, R. P. and MacKay, D. J. C. (2007). *Bayesian Online Changepoint Detection.* arXiv:0710.3742. Hazard function, equation 4, page 2. Conjugate-exponential predictive, section 2.3, pages 3-4. Algorithm 1, pages 3-4. Run length truncation, section 2.4, page 4.

### Choices stated explicitly

- The hazard is constant, `H(tau) = 1 / lam`, with `lam = 252.0` in the pipeline (about one year of trading days). The paper uses a discrete exponential prior (section 3, page 4) and does not fix a value for the daily data of this page.
- The run length distribution is truncated by **length** at `max_run_length = 252`, not by mass as the paper suggests (section 2.4, page 4). The choice bounds the memory at `O(max_run_length)` per step.
- The alarm is the cumulative mass on the shortest `mass_window = 10` run lengths, not the exact mass on run length zero. The reason is the one in section 4: with a constant hazard, the mass on run length zero stays at `1 / lam` on every step, and the signal is the mass that moves to short runs one to ten.
- The detection threshold is `0.5` cumulative mass, and the warm-up is 20 returns. The first alarm happens while the run length is one sample long, before the detector has seen a normal run.
- The synthetic stream is labelled "synthetic" in the code, in the page and in the figure. The volatility levels (`0.01` and `0.03`) and the change at sample 500 are this implementation's choice, not a paper's.

### Measured cost

`RegimePipeline.run` on the 1,253 returns of `GSPC.csv` takes about 0.4 ms per sample median on this host. The 99th percentile moves between runs on the same host (0.5 ms to 1.9 ms on three runs), because it is one sample on 1,253; the median is the stable number and it is the one stated. The budget of the detector is 14.9 ms, about ten times the p99 of 1.5 ms measured on a 12,000-sample stream, where the long stream is the one the budget is for. The budget of `BayesianOnlineChangepoint` is set to 2 ms, measured on this host.

### Development diary

The first version of the detector raised an alarm on the first sample of every stream, because the paper boundary condition puts all the mass at run length zero for the first sample (section 2.2, page 3). The detector now starts with the previous step marked as an alarm, so the first sample is not an alarm, and the alarm only fires on the step where the mass crosses the threshold after having been below it. The second version used the mass on run length zero, which on every stream is about `1 / lam` and never crosses `0.5`; the alarm is now the mass on the first `mass_window` run lengths, and the two shifts of the test (mean and variance) are both detected on the fifth sample after the change.
