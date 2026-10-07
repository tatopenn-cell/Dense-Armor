# CUSUM (slow-drift detection) + ARL theory

A drift too small, at any single step, to cross an anomaly detector's instantaneous
threshold still accumulates. The **CUSUM** (cumulative sum) detector sums the small
deviations over time instead of judging each point in isolation, so a slow sustained
shift eventually trips it even when no single point ever would.

This page shows how to run the detector, how to read its one tuning knob, and — before
running anything — how long it will take to fire (or to false-alarm).

## The signal

The same 100 Hz joint velocity: nominal 0.5 rad/s, a collision spike at sample 1500,
and a worn gear that starts shifting the average at sample 2000.

```python
import numpy as np
rng = np.random.default_rng(42)
fs = 100
t = np.arange(3000) / fs
v = 0.5 + 0.1 * rng.standard_normal(len(t))
v[1500] = 2.5
v[2000:] += 0.2 * (np.arange(1000) / 1000)
```

## 1. Feed the detector, one sample at a time

```python
from dense_armor.drift.cusum import cusum_detector

alarms = cusum_detector(v, k=0.5, h=5.0, reference="fixed",
                        radius=20, ref_mult=3)
```

`cusum_detector` takes the whole series and returns the list of indices where the
detector fired.

`k` (the **slack**) is the amount a single sample has to exceed, in robust sigmas, to
contribute to the sum. A sample below `k` contributes nothing.

`h` (the **threshold**) is how large the accumulated sum has to be before an alarm. It
is expressed in the same robust-sigma units.

`reference="fixed"` fits the reference median and MAD on the first `radius * ref_mult`
samples and never updates. `reference="adaptive"` recomputes them from a sliding window
every step.

On the running example, with `k=0.5`, `h=5.0`, the detector fires somewhere just after
sample 2000, once the accumulated drift has built up past 5 sigmas.

## 2. The formula, symbol by symbol

Let `z_t` be the standardized deviation of sample `t` from the reference, in robust
sigmas:

```
S_0 = 0
S_t = max(0, S_{t−1} + z_t − k)
alarm iff S_t > h, then S_t ← 0
```

- `z_t = (x_t − med) / S`, where `med` and `S` come from the reference window.
- `k` is the slack: how much a single sample must exceed to count.
- `h` is the threshold: how large the accumulated evidence must be to fire.

The `max(0, ·)` means the sum cannot go negative. If the process drifts back down, the
accumulated evidence is discarded, and the detector starts over from zero.

**Hand case.** Seven standardized samples:
`z = [0.2, −0.4, 0.1, 1.8, 2.1, 1.9, 2.3]`, `k = 0.5`, `h = 5`:

```
S_0 = 0
S_1 = max(0, 0 + 0.2 − 0.5) = 0.0
S_2 = max(0, 0 − 0.4 − 0.5) = 0.0
S_3 = max(0, 0 + 0.1 − 0.5) = 0.0
S_4 = max(0, 0 + 1.8 − 0.5) = 1.3
S_5 = max(0, 1.3 + 2.1 − 0.5) = 2.9
S_6 = max(0, 2.9 + 1.9 − 0.5) = 4.3
S_7 = max(0, 4.3 + 2.3 − 0.5) = 6.1   →  6.1 > 5: alarm at the 7th sample
```

The first three samples (below `k` in magnitude, or negative) contributed nothing; the
four consecutive samples above `k` built up enough evidence in four steps.

## 3. How long will it take: ARL

Two numbers describe a CUSUM's behaviour before you run it:

- **ARL₀**: how many samples pass, on average, between false alarms when nothing is
  drifting. Larger is better: a nuisance-free detector.
- **ARL₁**: how many samples pass, on average, between the start of a real drift and
  the alarm. Smaller is better: a fast detector.

For the one-sided detector there is a closed-form approximation
([Reynolds 1975](https://doi.org/10.1080/00401706.1975.10489284), equation 12):

```
ARL(δ) = ( exp(−2·δ·h') − 1 + 2·δ·h' ) / (2·δ²)
```

where `h' = h + 1.166` (the Reynolds correction for the discrete-time case), and `δ` is
the true mean shift minus the slack: `δ = μ − k`. For the two-sided case, Kemp's identity
gives `1 / ARL = 1 / ARL⁺ + 1 / ARL⁻`.

`detectability_report` computes both numbers for you, without needing a stream to
calibrate on:

```python
from dense_armor.drift.cusum import detectability_report

report = detectability_report(local_noise_scale=1.0, k=0.5, h=5.0, candidate_shift=1.0)
```

`shift=1.0` means "a real drift of one sigma": the report gives both the predicted
false-alarm run length (in control) and the predicted detection delay (after the shift),
computed from the two closed-form formulas.

**Hand case, `k=0.5`, `h=5`.** With `shift=0` (nothing is drifting),
`ARL ≈ 469 samples`: about one false alarm every 4.7 seconds at 100 Hz. With
`shift=1.0` (a real drift of one sigma above the slack),
`ARL ≈ 10.3 samples`: the detector fires about a tenth of a second after the drift
starts. These are the classic CUSUM tables' values for these parameters; the report
prints them, no calibration run needed.

## 4. Reading the trade-off

`k` and `h` are two knobs, and every setting is a trade-off:

- **Smaller `k`**: fires on smaller drifts, but also false-alarms more often on noise.
- **Larger `k`**: ignores small persistent deviations (good if you know the drift you
  care about is bigger than `k`), fewer false alarms, but misses the small drifts.
- **Larger `h`**: needs more accumulated evidence; fewer false alarms, later detection.
- **Smaller `h`**: faster detection, more false alarms.

The classic choice is `k = μ₁ / 2` where `μ₁` is the smallest drift you care about. At
that setting, `h` is chosen to hit a desired ARL₀ (a false-alarm rate), and the detector
is then optimal in the sense that no other `k` gives a shorter ARL₁ at the same ARL₀.
Reynolds 1975 proves this for the one-sided test (Section 4 of the paper); the same
result holds numerically for the two-sided case.

## 5. Streaming CUSUM for river

`CUSUMDriftDetector` (`dense_armor.drift.detector`, `pip install dense-armor[river]`)
is the same detector, one sample at a time, as a river `DriftDetector`: `update(x)`
returns the detector and sets `drift_detected`.

```python
import numpy as np
from dense_armor.drift.detector import CUSUMDriftDetector

det = CUSUMDriftDetector(reference="fixed")
i = next(i for i, x in enumerate(v) if det.update(x).drift_detected)
```

On the running example, `i` is the index of the first sample after the drift that pushes
the accumulated sum over `h`.

Because `reference="fixed"` never updates, the detector fires at the *same index* as the
batch `cusum_detector` on the same series. That equivalence is checked in the test suite
for both `reference="fixed"` and `reference="adaptive"`.

### ARL as methods on the detector

```python
det.expected_detection_delay(shift=1.0)
det.expected_false_alarm_run()
```

Same numbers as `detectability_report`, computed from the detector's own `k` and `h`.

## 6. Against river's own detectors

20 seeded streams of 1,000 N(0, 1) samples each, mean shift at sample 500. Each cell
is **recall / delay in samples / false alarms per stream**:

| shift (σ) | CUSUM adaptive | CUSUM fixed | PageHinkley | ADWIN |
|---|---|---|---|---|
| 0.5 | 0.00 / — / 0.00 | 0.55 / 96.0 / 0.10 | 1.00 / 81.0 / 0.05 | 1.00 / 127.8 / 0.00 |
| 1.0 | 0.00 / — / 0.00 | 0.90 / 42.3 / 0.50 | 1.00 / 41.2 / 0.10 | 1.00 / 54.2 / 0.00 |
| 1.5 | 0.15 / 15.0 / 0.05 | 1.00 / 20.9 / 0.40 | 1.00 / 42.9 / 0.10 | 1.00 / 43.0 / 0.00 |
| 2.0 | 0.05 / 6.0 / 0.00 | 1.00 / 12.6 / 0.35 | 1.00 / 14.9 / 0.05 | 1.00 / 41.4 / 0.00 |
| 3.0 | 0.85 / 9.8 / 0.00 | 1.00 / 7.8 / 0.05 | 1.00 / 11.2 / 0.05 | 1.00 / 11.0 / 0.00 |

`fixed` detects a sustained shift fastest from 1.5σ upward, at the price of more false
alarms than PageHinkley and ADWIN. `adaptive` is built for something else: its sliding
reference catches up with a new level, so it reacts to the *leading edge* of a drift and
misses most small step changes. ADWIN never raised a false alarm here but is the slowest
on small shifts.

## API reference

::: dense_armor.drift.cusum

---

## Details

`CUSUMDriftDetector` (`dense_armor.drift.detector`) is the river wrapper. On a 1-sigma
shift with `reference="fixed"` the measured delay is 41 samples against a predicted 40.3
from `one_sided_arl`. The batch and streaming versions of the detector, on the same
series, fire at the same indices — that equivalence is what the test suite asserts, for
both `reference` modes.

The `1.166` correction in `h' = h + 1.166` comes from Reynolds 1975, Table 1: without it
the Brownian-motion approximation underestimates ARL⁺ for small `h`, and the correction
is the value that makes the approximation match the tabulated exact values for
`h ∈ {2, 5, 10}` across a range of `k`.
