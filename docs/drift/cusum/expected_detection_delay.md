# expected_detection_delay: how long until a real change is noticed

[← CUSUM](index.md)

`CUSUMDriftDetector.expected_detection_delay(shift)` returns the expected number of samples needed to detect a standardized mean shift `shift`, in noise-widths, for the settings of the detector.

## First example

```python
from dense_armor.core.online.drift.detector import CUSUMDriftDetector

for kw in ({}, {"h": 10.0}, {"k": 1.0}):
    d = CUSUMDriftDetector(**kw)
    print(kw, round(d.expected_detection_delay(1.0), 1), round(d.expected_detection_delay(2.0), 1))
```

```
{} 40.3 13.9
{'h': 10.0} 20.3 7.2
{'k': 1.0} 448.0 20.7
```

For a shift of one noise-width the default detector predicts 40.3 samples, and for two noise-widths 13.9. A smaller `h` is faster (20.3 and 7.2 with `h` = 10) and a larger slack `k` is slower (448.0 and 20.7 with `k` = 1.0): the slack eats the small surprises, so a small shift takes much longer. The number is the ARL of [two_sided_arl](two_sided_arl.md) with $\mu$ = `shift`.

## The estimate against the stream

The formula takes the mean and the noise width as known exactly; the detector estimates them on a window of `radius * ref_mult` samples. For a shift of 1 noise-width the detector predicts 40.3 samples; measured on a stream with `reference="fixed"`, the delay is 41.

## The parameters

| Parameter | What it is |
|---|---|
| `shift` | the standardized mean shift to detect, in noise-widths |

The method returns a float, the expected number of samples; `k` and `h` come from the detector, see [CUSUMDriftDetector](detector.md). For the time between false alarms see [expected_false_alarm_run](expected_false_alarm_run.md), and for the same numbers in the units of your series [detectability_report](detectability_report.md).

## Details

- **Source.** The number is the two-sided ARL of Reynolds (1975) for a shift `shift`, see [two_sided_arl](two_sided_arl.md).
