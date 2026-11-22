# BatchedLandmarkShield: batched Kalman + Hampel for landmark streams

This primitive cleans a stream of landmark points (e.g. 68 face landmarks) using a
batched constant-velocity Kalman filter with a causal Hampel outlier filter per
coordinate.

It is the vectorized version of the shield used in D-Avatar: every point gets its
own 4D Kalman filter (x, y, vx, vy) and a causal Hampel window. Points with
confidence 0 run predict-only; after `max_gap` frames they decay toward a neutral
pose. A point that jumps far from its recent median is replaced by that median.

## 1. Quick example

```python
import numpy as np
from dense_armor.core.signals.vision.landmarks import BatchedLandmarkShield, LandmarkStream

# Synthetic 68-point face
base = np.random.randn(68, 2).astype(np.float32)
conf = np.ones(68, dtype=np.float32)
times = np.arange(100) / 30.0

stream = LandmarkStream(np.tile(base, (100, 1, 1)), np.tile(conf, (100, 1)), times, fps=30.0)

shield = BatchedLandmarkShield(window=7, n_sigmas=3.0, max_gap=5)
for frame in stream:
    out = shield.transform_one(frame)
    # out['points'] has cleaned landmarks, out['cleaned'] has per-point counts
```

## 2. How it works

Each point gets a 4D constant-velocity Kalman filter with state `(x, y, vx, vy)`:

- **Prediction**: `x_{k+1} = F x_k`, with `F = [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]]`
- **Measurement update** (when confidence > 0): standard Kalman update after Hampel outlier check
- **Predict-only** (when confidence == 0): advances state with `F`, increments gap counter
- **Neutral decay**: after `max_gap` frames of occlusion, prediction blends toward the point's neutral pose
- **Hampel outlier filter**: per-coordinate causal window, replaces values beyond `n_sigmas` MADs with the median

The filter is "batched" because all points are processed with vectorized NumPy operations
instead of Python loops, making it ~10x faster than a per-point loop.

### Motion model

Constant velocity: `x_{k+1} = x_k + v_k dt + w_k` with `w_k ~ N(0, Q)`. The default `Q`
penalizes velocity changes more than position: `Q = diag(1e-6, 1e-6, 1e-4, 1e-4)`.

### Hampel filter

Causal window per coordinate per point. A value is an outlier if it exceeds `n_sigmas`
median absolute deviations from the running median. Outliers are replaced by the median.

Reference: Pearson et al. (2016), *Robust outlier detection using the Hampel filter*.

## 3. API

### BatchedLandmarkShield

```python
BatchedLandmarkShield(
    window: int = 7,
    n_sigmas: float = 3.0,
    max_gap: int = 5,
    neutral: np.ndarray | None = None,
)
```

**Parameters**

- `window`: causal Hampel window size (default 7)
- `n_sigmas`: Hampel threshold in scaled MAD units (default 3.0)
- `max_gap`: max frames of predict-only before neutral decay (default 5)
- `neutral`: neutral pose (N, 2) to decay to after max_gap. If None, uses first valid measurement.

**Input to `transform_one`**

```python
{
    "points": np.ndarray (N, 2),      # landmark coordinates
    "confidence": np.ndarray (N,),    # per-point confidence in [0, 1]
    "t": float | None                 # optional timestamp
}
```

**Output of `transform_one`**

```python
{
    "points": np.ndarray (N, 2),      # cleaned landmarks
    "cleaned": np.ndarray (N,)        # 0, 1, or 2 per point
}
```

**Attributes**

- `budget_s`: 3.5e-2 (p99 latency bound)
- `memory_class`: "O(N * window)"

### LandmarkStream

See `LandmarkStream` in the landmarks module for the input stream format.

## 4. Comparison with LandmarkCleaner

| Feature | LandmarkCleaner | BatchedLandmarkShield |
|---------|----------------|----------------------|
| Kalman prediction during gap | No (holds last value) | Yes (constant velocity) |
| Neutral decay after max_gap | No | Yes |
| Batched vectorized ops | No (per-point loops) | Yes |
| p99 latency (68 pts) | ~25 ms | ~3 ms |
| budget_s | Not declared | 35 ms |
| Hampel outlier filter | Yes (per-coordinate) | Yes (batched) |

Use `BatchedLandmarkShield` when you need Kalman prediction during occlusion and
high throughput. Use `LandmarkCleaner` for simpler gap-filling without velocity
prediction.

## 5. Measured cost

Measured on a Kaggle CPU (Intel Xeon), 2000 frames of 68 landmarks:

| Metric | Value |
|--------|-------|
| p99 latency | 2.6 ms |
| median latency | 1.4 ms |
| `budget_s` | 35 ms |
| `memory_class` | O(N * window) |

The p99 is well within the declared 35 ms budget.

## 6. References

- Thieu, N. & Melnik, R. (2025). *Nonlinear Dynamical Systems for Automatic
  Face Annotation in Head Tracking and Pose Estimation*. arXiv:2502.15179.
  EKF equations §2.1 pp. 4-5; constant-velocity motion model §4.2 p. 13.
- Zhu, X. et al. (2021). *Reasoning Structural Relation for
  Occlusion-Robust Facial Landmark Localization*. arXiv:2112.10087.
  Test protocol: dynamic occlusion moving across face.
- Pearson, R. et al. (2016). *Robust outlier detection using the Hampel filter*.
  In *Data Mining and Knowledge Discovery*.

## 7. Development diary

- Promoted from D-Avatar's `FaceShield` (batched version) to Dense-Armor
  primitive to eliminate duplication between D-Avatar and Dense-Armor.
- The original per-point loop version had p99 ~17 ms; batched NumPy brought
  it to ~3 ms.
- An occlusion first appeared as a Kalman update with measurement (0, 0),
  because `KalmanFilter.learn_one` treats empty dict as zero measurement.
  The shield now calls it with `None` for predict-only.
- The Hampel batch rule was validated against `LandmarkCleaner`'s per-point
  rule on random streams.