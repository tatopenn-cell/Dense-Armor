# cusum_detector: the same alarms on a whole recording

[← CUSUM](index.md)

`cusum_detector` runs on a recorded series in one call and fires at exactly the same samples as
the streaming detector.

```python
import numpy as np
from dense_armor.core.online.drift.cusum import cusum_detector
from dense_armor.core.online.drift.detector import CUSUMDriftDetector

rng = np.random.default_rng(0)
x = np.concatenate([rng.normal(0, 1, 500), rng.normal(1.5, 1, 500)])
flags, s = cusum_detector(x, reference="fixed")
det = CUSUMDriftDetector(reference="fixed")
print(np.flatnonzero(flags)[:4], [i for i, v in enumerate(x) if det.update(v).drift_detected][:4])
```

```
[511 524 533 547] [511, 524, 533, 547]
```

Use the batch function to study a recording, and the streaming detector inside the control loop:
the alarms are the same.
