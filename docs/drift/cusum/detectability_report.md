# detectability_report: the estimate in the units of your series

[← CUSUM](index.md)

`detectability_report(local_noise_scale, k, h, candidate_shift)` turns the average run length into numbers for a real deployment: the noise level of your signal and the size of the shift you care about, both in the units of the series. It is a pre-flight estimate, computed before running any benchmark.

## First example

```python
from dense_armor.core.online.drift.cusum import detectability_report

r = detectability_report(local_noise_scale=0.02, k=0.5, h=20.0, candidate_shift=0.04)
for key, val in r.items():
    print(key, float(val))
```

```
false_alarm_arl 1556958360.0943098
detection_arl 13.888444444444445
shift_in_sigma 2.0
```

A series whose noise is 0.02 and a shift of 0.04, both in the units of the series, is a shift of 2 noise-widths (`shift_in_sigma`). With the default threshold $h = 20$ the alarm comes about 14 samples after the shift (`detection_arl`), and a false alarm is expected only once in about 1.6 billion samples (`false_alarm_arl`).

## How it works

The function divides the shift by the noise (`shift_in_sigma = candidate_shift / local_noise_scale`) and gives that number to the formula of [two_sided_arl](two_sided_arl.md): with a shift of 0 for the false alarms and with the shift in noise-widths for the detection.

### The floor at 1.0

When the noise is tiny against the shift, the raw formula returns a fractional ARL below 1, which has no physical meaning: a detector cannot flag in under one sample. Both numbers are floored at 1.0:

```python
from dense_armor.core.online.drift.cusum import detectability_report

r = detectability_report(local_noise_scale=0.0001, k=0.5, h=20.0, candidate_shift=1.0)
print({key: float(val) for key, val in r.items()})
```

```
{'false_alarm_arl': 1556958360.0943098, 'detection_arl': 1.0, 'shift_in_sigma': 10000.0}
```

Here the shift is 10,000 noise-widths and the detection is 1.0. Treat a value that needed the floor as "near-instant detection", not as a precise number: the formula is an approximation that is least trustworthy exactly in this regime.

## The parameters

| Parameter | What it is |
|---|---|
| `local_noise_scale` | the noise of the signal, in real units (for example the median absolute distance times 1.4826 of a causal window, as the detector computes it) |
| `k`, `h` | the slack and the threshold of the detector, in robust-sigma units |
| `candidate_shift` | the shift to evaluate, in real units |

The function returns a dict: `false_alarm_arl` (samples between false alarms, floored at 1.0), `detection_arl` (samples to detect the shift, floored at 1.0) and `shift_in_sigma`.

## Details

- **Real-world check.** `detectability_report` was validated on two real physical domains before being promoted from Dense-Evolution-Discovery. On real lidar (Sydney Urban Objects, 7 independent points) the real detection was always faster than the predicted mean ARL, a consistent bias in one direction. On a real accelerometer (UCI HAR, 5 points) the result was mixed: 2 points faster than predicted, 3 slower. The direction of the gap is not the same across domains, so no single correction factor transfers between them.
- **What it is.** The estimate assumes independent Gaussian noise: use it to reason before a benchmark, then measure the real rates on the real deployment.
