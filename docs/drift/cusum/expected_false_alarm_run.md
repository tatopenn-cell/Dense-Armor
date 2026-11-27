# expected_false_alarm_run: how long between false alarms

[← CUSUM](index.md)

`CUSUMDriftDetector.expected_false_alarm_run()` returns the expected number of samples between false alarms when nothing has changed, for the settings of the detector. It takes no argument.

## First example

```python
from dense_armor.core.online.drift.detector import CUSUMDriftDetector

for kw in ({}, {"h": 10.0}, {"k": 1.0}):
    print(kw, round(CUSUMDriftDetector(**kw).expected_false_alarm_run(), 1))
```

```
{} 1556958360.1
{'h': 10.0} 70673.6
{'k': 1.0} 6.060298510226602e+17
```

The default `h` = 20 gives about 1.6 billion samples, `h` = 10 gives 70,673.6, and a larger slack `k` = 1.0 makes false alarms much rarer. The number is the ARL of [two_sided_arl](two_sided_arl.md) with $\mu = 0$, and it depends only on `k` and `h`: `two_sided`, `reference` and `radius` do not change it.

```python
from dense_armor.core.online.drift.detector import CUSUMDriftDetector

vals = {round(float(CUSUMDriftDetector(**kw).expected_false_alarm_run()), 1)
        for kw in ({}, {"two_sided": False}, {"reference": "adaptive"}, {"radius": 20})}
print(vals)
```

```
{1556958360.1}
```

## The formula against the detector

The formula takes the mean and the noise width as known exactly. The detector estimates them on a window of `radius * ref_mult` samples (30 by default), so on a stream of 1000 stable samples it gives more false alarms than the formula says. `benchmarks/cusum_false_alarm.py` runs 1000 stable series of 1000 samples for each threshold:

| $h$ | formula: samples between false alarms | formula: chance of an alarm in 1000 samples | measured, `adaptive` | measured, `fixed` |
|---|---|---|---|---|
| 5 | 469.1 | 0.881 | 1.000 | 0.841 |
| 10 | 70,674 | 0.014 | 0.505 | 0.389 |
| 20 | $1.6 \times 10^{9}$ | 0.000 | 0.033 | 0.126 |

At $h = 5$ the formula already predicts a false alarm in 88% of the series, and the detector gives more. At $h = 20$ the formula predicts none, and the detector raises one in 3.3% (`adaptive`) or 12.6% (`fixed`) of them: that is the cost of estimating the reference on a short window. Read the measured columns for the cost of a threshold on a real stream.

## The parameters

The method has no parameters: it reads `k` and `h` from the detector, which are described on the page [CUSUMDriftDetector](detector.md). For the delay of a real change see [expected_detection_delay](expected_detection_delay.md).

## Details

- **Source.** The number is the two-sided ARL with $\mu = 0$ of Reynolds (1975), see [two_sided_arl](two_sided_arl.md).
- **Why the run is shorter in practice.** The docstring says that the detector estimates the mean and the noise width on a window, so the run it gives is shorter than the formula; the measure is in `benchmarks/cusum_false_alarm.json`.
