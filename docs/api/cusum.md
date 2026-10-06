# CUSUM (slow-drift detection) + ARL theory

A drift too small, at any single step, to cross [Arbiter](arbiter.md)'s instantaneous
threshold still accumulates: `cusum_detector` sums small deviations over time instead of
judging each point in isolation, so a slow sustained shift eventually trips it even when no
single point ever would. `one_sided_arl`/`two_sided_arl`/`detectability_report` are a
pre-flight estimate of how long that takes -- given a detector's real local noise level and a
candidate shift, how many samples until detection, or until a false alarm -- computable
before running a benchmark. Promoted from Dense-Evolution-Discovery after validation on two
independent real physical domains (lidar, accelerometer); see `detectability_report`'s own
docstring for the honest, mixed real-world result.

::: dense_armor.utility.cusum

---

**See also**: [Arbiter](arbiter.md) -- the instantaneous per-point detector this module is
the slow-drift complement to.

## Streaming CUSUM for river

`CUSUMDriftDetector` (`dense_armor.utility.river_drift`, `pip install dense-armor[river]`)
is `cusum_detector` one sample at a time, as a river `DriftDetector`: `update(x)` returns
the detector and sets `drift_detected`. On the same series it fires at exactly the same
indices as `cusum_detector`, for both `reference="adaptive"` and `reference="fixed"`.

```python
import numpy as np
from dense_armor.utility.river_drift import CUSUMDriftDetector

rng = np.random.default_rng(0)
stream = np.concatenate([rng.normal(0, 1, 500), rng.normal(1.5, 1, 500)])
det = CUSUMDriftDetector(reference="fixed")
print(next(i for i, x in enumerate(stream) if det.update(x).drift_detected))
```

```
511
```

The shift starts at sample 500; it is flagged 11 samples later.

`expected_detection_delay(shift)` and `expected_false_alarm_run()` return the ARL
predictions of `one_sided_arl` / `two_sided_arl` for the detector's own `k` and `h`; on a
1-sigma shift with `reference="fixed"` the measured delay is 41 samples against a predicted 40.3.

Against river's own detectors (20 seeded streams of 1,000 N(0,1) samples per row, mean shift
at sample 500; recall = fraction of streams where the shift was detected, delay in samples,
false alarms per stream):

| shift (σ) | CUSUM adaptive | CUSUM fixed | PageHinkley | ADWIN |
|---|---|---|---|---|
| 0.5 | 0.00 / — / 0.00 | 0.55 / 96.0 / 0.10 | 1.00 / 81.0 / 0.05 | 1.00 / 127.8 / 0.00 |
| 1.0 | 0.00 / — / 0.00 | 0.90 / 42.3 / 0.50 | 1.00 / 41.2 / 0.10 | 1.00 / 54.2 / 0.00 |
| 1.5 | 0.15 / 15.0 / 0.05 | 1.00 / 20.9 / 0.40 | 1.00 / 42.9 / 0.10 | 1.00 / 43.0 / 0.00 |
| 2.0 | 0.05 / 6.0 / 0.00 | 1.00 / 12.6 / 0.35 | 1.00 / 14.9 / 0.05 | 1.00 / 41.4 / 0.00 |
| 3.0 | 0.85 / 9.8 / 0.00 | 1.00 / 7.8 / 0.05 | 1.00 / 11.2 / 0.05 | 1.00 / 11.0 / 0.00 |

Each cell is recall / delay / false alarms. "fixed" detects a sustained shift fastest from
1.5σ upward, at the price of more false alarms than PageHinkley and ADWIN. "adaptive" is
built for something else: its sliding reference catches up with a new level, so it reacts
to the leading edge of a drift and misses most small step changes. ADWIN never raised a
false alarm here but is the slowest on small shifts.
