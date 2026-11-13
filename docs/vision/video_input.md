# Video into any primitive

`VideoInput` gives frames to a primitive that already exists in the library, in the form that primitive takes. Nothing is copied: a frame becomes numbers by the existing [`FrameFeatures`](features.md), and which method of the primitive is called is decided by the same adapter as the [audio one](../audio/index.md), which `VideoInput` extends.

```python
from dense_armor.core.signals.vision.video_input import VideoInput
```

## What the primitive receives

A frame (a `Frame` of a stream, a dict with `"frame"`, or an array) is turned by `FrameFeatures` into the flat dict `hog_*`, `mean`, `contrast`, `centroid_*` and `flow_*`. The optical flow is computed against the previous frame, so use one adapter for one stream and give the frames in order. For colour images pass `extractor=FrameFeatures(gray=False)`.

The adapter looks at the methods of the primitive:

| the primitive has | `feed(frame, y)` does |
| --- | --- |
| `update` and `drift_detected` (a drift detector) | gives it one number (`key`, `"mean"` by default), and reports whether it fired |
| `score_one` | scores the frame, then learns it |
| `predict_one` and `learn_one` | predicts, then learns with the label `y` (`NeedsLabels` if there is none) |
| `transform_one` | learns, then transforms |
| only `learn_one` | learns |
| only `update_many` (a filter of samples) | raises `ValueError`: a frame has no samples |

## A drift detector on a stream

A square moves for 60 frames, then the whole image turns bright.

```python
import numpy as np
from dense_armor.core.online.drift.adwin import ADWIN
from dense_armor.core.signals.vision.streams import ArrayStream
from dense_armor.core.signals.vision.video_input import VideoInput

frames = []
for i in range(60):
    f = np.zeros((16, 16), dtype=np.float32)
    f[4:10, 2 + i // 8 : 8 + i // 8] = 1.0
    frames.append(f)
for i in range(20):
    frames.append(np.full((16, 16), 0.9, dtype=np.float32))

v = VideoInput(ADWIN())
out = v.run(ArrayStream(np.stack(frames), fps=10.0))
print(v.mode, v.n, v.alarms, out.index(True))   # drift 80 4 64
```

The detector fires 4 frames after the change, on the mean luminance.

## Checked

All the 40 classifiers, regressors, anomaly scorers and drift detectors of `core/online` that are built without arguments were fed 24 frames of random noise through `VideoInput` without an error. The 28 that need arguments were not tried, and neither were the classes of other sections. The test is `test/vision/test_video_input.py`.
