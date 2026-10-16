# Dense-Armor vs a Kalman filter

A comparison between Dense-Armor and a Kalman filter. They solve different problems: the Kalman filter needs a model of how the system moves, while Dense-Armor's tools check each reading against its recent neighbours and a reference. Neither is always better.

Dense-Armor does not replace a Kalman filter. They solve different problems:
the Kalman filter reconstructs a signal through a **known dynamic model**, the
shields reconstruct a signal through **local coherence** and a reference.
Neither is strictly better; they lose in different regimes.

## 1. The two settings

A Kalman filter knows the dynamics of the system: how the state evolves, how
the sensors see it, how much noise each has. Given that model, it produces
the minimum-variance estimate — the best possible, in the statistical sense.

Dense-Armor knows nothing about the dynamics. It knows the shape the signal
is *supposed* to have: smooth in time, coherent in space, close to a
reference. It builds a robust blind estimate from that shape alone.

The two answers agree when the dynamics are linear and the noise is
Gaussian. They diverge the moment the signal carries a spike, a NaN, or a
missing sample that the Kalman model has not been told about.

## 2. A measurement

A random walk, `15%` missing samples, `3%` huge spikes. Both filters see
the same corrupted stream; the mean squared error against the clean signal:

| method | MSE |
|---|---|
| no protection | ~21000 |
| Kalman, no anti-outlier gating (the common case) | ~7300 |
| Kalman, with anti-outlier gating and known dynamics | ~0.12 |
| Dense-Armor, blind mode | ~0.23 |

```python
import numpy as np
from dense_armor.utility.anomaly.mahalanobis import OnlineRobustMahalanobis

rng = np.random.default_rng(0)
est = OnlineRobustMahalanobis()
for _ in range(150):
    est.learn_one({"x": float(1.0 + 0.05 * rng.normal())})
print(round(float(est.score_one({"x": 1e6})), 2))
```

```
25265805.35
```

A single spike throws off the Kalman gain — the filter trusts the
measurement and drags the state behind itself. Dense-Armor scores the
spike as astronomically far from the centre it has learned (`25 million`
against a training stream around `1.0`), which a downstream detector
flags as an anomaly instead of feeding it into the reconstruction. In
blind mode it does not know the true dynamics, so it cannot beat the
*correctly configured* Kalman filter (`0.23` vs `0.12` MSE) — but it does
not need to be configured either.

## 3. When to use which

Use a **Kalman filter** when:

- you know the dynamics (a robot model, a physics simulation)
- the noise is Gaussian, or close
- you can spend the time to calibrate `Q` and `R`
- you want the minimum-variance estimate for that specific model

Use **Dense-Armor** when:

- you do not know the dynamics (a new sensor, an LLM embedding, an image)
- the signal carries spikes, NaN, missing data, adversarial perturbations
- you need something that works as-is on any tensor, temporal or not
- you want the same interface (`learn_one`, `transform_one`) as the rest of
  the library

The two compose: a Kalman filter for the clean stream, a shield in front of
it for the corrupted one.

## Details

### References

The comparison is measured by `test/experiments/testKalman.py` and the
`benchmarks/anomaly_streaming.py` script on the same random walk.

The Kalman formulation is Kalman, R. E. (1960). *A new approach to linear
filtering and prediction problems*. Journal of Basic Engineering, 82(1),
35-45. The robust blind estimator is `OnlineRobustMahalanobis` of the anomaly
section, see [anomaly/mahalanobis.md](anomaly/mahalanobis.md).

### Choices stated explicitly

- **The comparison is on a random walk, not on a robot joint.** The point
  is to have a signal where the Kalman model is exactly known, so the
  comparison is favorable to the Kalman filter. On a robot joint the
  dynamic model is itself an approximation.
- **The Kalman "no gating" case is the common case.** Most Kalman
  deployments in practice do not add anti-outlier gating; the number
  ~7300 is the one a user is likely to see without extra work.
- **"Blind mode" means no reference.** With `reference=historical_baseline`
  the shield can also track slow drift, see
  [drift/detectors.md](drift/detectors.md).

### Measured cost

`OnlineRobustMahalanobis` runs in `< 1 ms` per sample (`budget_s = 1e-3 s`),
memory `O(n^2)` with `n` the number of features. The Kalman filter in the
comparison is a standard Kalman filter from a general-purpose filtering library, no JIT.

### Development diary

The comparison was added to the README after a user asked whether
Dense-Armor is "a Kalman replacement". It is not: it is a shield, it works
without a process model, it does not give the minimum-variance estimate.
The table is kept honest by showing the case where a well-calibrated
Kalman *wins* (0.12 vs 0.23), not only the case where Dense-Armor wins.
