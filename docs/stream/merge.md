# Merge by time and shuffle

`merge_by_time` puts several streams in one timeline and `shuffle` mixes a stream in a bounded buffer. The overview of the section is on [Streams](index.md).

## 2. Two sensors, one timeline

A camera at 30 frames per second and joint encoders at 100 Hz give two streams; an estimator
needs them as one stream, in time order.

```python
import numpy as np
from dense_armor.core.signals.stream import iter_array, merge_by_time

cam = iter_array(np.array([[1.0], [2.0]]), t=np.array([0.0, 1 / 30]), names=["frame"])
joints = iter_array(np.array([[0.1], [0.2], [0.3], [0.4]]), t=np.arange(4) / 100, names=["q0"])
for sig, _ in merge_by_time(cam, joints):
    print(round(sig.t, 3), sig.names[0])
```

```
0.0 frame
0.0 q0
0.01 q0
0.02 q0
0.03 q0
0.033 frame
```

The samples come out sorted by time across the two sensors; at equal times (0.0 here)
the order of the arguments decides, camera first. The merge keeps only one sample per stream in
memory, so it works on live sensors as well as on files.

## API reference

::: dense_armor.core.signals.stream.merge

::: dense_armor.core.signals.stream.shuffle

---

## Details

- `merge_by_time` is a k-way merge with a heap; it assumes each input stream is already in time
  order. `shuffle(stream, buffer_size, seed)` shuffles streams without timestamps in a bounded
  buffer.
