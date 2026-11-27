# two_sided_arl: how long will the two-direction CUSUM wait?

[← CUSUM](index.md)

`two_sided_arl(mu, k, h, corrected=True)` gives the average run length of the CUSUM that watches both directions, the one of `CUSUMDriftDetector` and `cusum_detector`, when the true mean has moved by `mu` noise-widths.

## First example

```python
from dense_armor.core.online.drift.cusum import one_sided_arl, two_sided_arl

for mu in (0.0, 1.0, 2.0):
    print(mu, round(float(two_sided_arl(mu, 0.5, 5.0)), 1))
up = one_sided_arl(1.0 - 0.5, 5.0)
down = one_sided_arl(-1.0 - 0.5, 5.0)
print(round(float(1 / (1 / up + 1 / down)), 1))
```

```
0.0 469.1
1.0 10.3
2.0 3.9
10.3
```

With $k = 0.5$ and $h = 5$: when nothing has changed ($\mu = 0$) a false alarm comes on average every 469 samples; after a shift of one noise-width ($\mu = 1$) the real alarm comes after about 10 samples; after a shift of two, after about 4. These are the classic values of CUSUM tables, a useful sanity check. The last line gets the same 10.3 for $\mu = 1$ from the two one-sided values.

## How it works

The two sides use the same $k$ and $h$. The upward side has drift $\mu - k$ and the downward side has drift $-\mu - k$, each one given by [one_sided_arl](one_sided_arl.md). They combine as

$$\frac{1}{\mathrm{ARL}} = \frac{1}{\mathrm{ARL}^+} + \frac{1}{\mathrm{ARL}^-},$$

the standard way to get a two-sided run length from two one-sided ones (Reynolds 1975, equation 13): an alarm on either side stops the wait.

## The parameters

| Parameter | What it is |
|---|---|
| `mu` | true standardized mean shift, in noise-widths; 0 means no change |
| `k` | the slack, in robust-sigma units |
| `h` | the threshold, in robust-sigma units |
| `corrected` | `True` (default) applies the correction of Siegmund to both sides |

The function returns the approximate run length, in samples. The same estimate in the units of your own series is on the page [detectability_report](detectability_report.md), and the numbers for a detector with its own settings are on [expected_false_alarm_run](expected_false_alarm_run.md) and [expected_detection_delay](expected_detection_delay.md).

## Details

- **Sources.** Reynolds, M. R. Jr. (1975), *Technometrics* 17(1), equation 13 for the combination of the two sides; the one-sided values are those of [one_sided_arl](one_sided_arl.md).
