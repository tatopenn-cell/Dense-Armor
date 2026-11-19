# Data in, one sample at a time

Robot data comes in many forms: files, recordings, arrays, live sensors. This page turns all of them into the same stream of readings, which every tool in the library then reads one at a time.

Every estimator in the library learns from a stream of samples: a `Signal` (named channels with
units and a timestamp) and, when there is one, a target `y`. Robot data arrive in many shapes:
log files, ROS bags, arrays, live sensors at different rates. This page turns all of them into
the same stream, merges sensors in time order and chooses and reshapes channels. The test streams, whose
change or fault happens at a known moment, are on the [Datasets](datasets.md) page.

## 1. A robot log file becomes a stream

A CSV file with the columns of the CASPER recording as it is distributed on Kaggle
(Kayan et al. 2023), one row per sample, with a timestamp column and a column holding the six
joint velocities as a list; here three joints and three rows.

```python
from pathlib import Path
from dense_armor.core.signals.stream import iter_csv

p = Path("arm.csv")
p.write_text(
    "Timestamp,Actual Joint Velocities\n"
    '0.00,"[0.1, -0.2, 0.0]"\n'
    '0.05,"[0.2, -0.1, 0.0]"\n'
    '0.10,"[0.3, 0.0, 0.1]"\n'
)
for sig, y in iter_csv(p, time_col="Timestamp", array_cols=["Actual Joint Velocities"]):
    print(sig.t, {k: round(float(v), 2) for k, v in sig.items()})
```

```
0.0 {'Actual_Joint_Velocities_0': 0.1, 'Actual_Joint_Velocities_1': -0.2, 'Actual_Joint_Velocities_2': 0.0}
0.05 {'Actual_Joint_Velocities_0': 0.2, 'Actual_Joint_Velocities_1': -0.1, 'Actual_Joint_Velocities_2': 0.0}
0.1 {'Actual_Joint_Velocities_0': 0.3, 'Actual_Joint_Velocities_1': 0.0, 'Actual_Joint_Velocities_2': 0.1}
```

Each row becomes one `Signal` with its timestamp; the list column is split into one
channel per joint. The file is read one row at a time, so a 24-hour log uses the same memory as a
three-row one; `start` and `stop` read only a time window and stop at the end of it. An empty or
unreadable cell becomes a missing channel, not an error.

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

## 3. Choosing and reshaping channels

A sample with a joint position, a torque and a temperature; the model needs only the
first two, with the torque scaled.

```python
from dense_armor.roles import Signal
from dense_armor.core.infrastructure.compose import FuncTransformer, Select

sig = Signal(values=[0.5, 2.0, 9.0], names=["q0", "tau0", "temp"], units=["rad", "N m", "C"], t=1.0)
pipe = Select(keys=("q0", "tau0")) | FuncTransformer(lambda v: v * 2.0, keys=("tau0",))
out = pipe.transform_one(sig)
print({k: float(v) for k, v in out.items()}, out.t, out.units)
```

```
{'q0': 0.5, 'tau0': 4.0} 1.0 ['rad', 'N m']
```

`Select` keeps the named channels (`Discard` drops them), `FuncTransformer` applies a
function to some channels; joined with `|` they form a pipeline like any other in the library.
The timestamp and the units of the kept channels travel with the sample.

## API reference

::: dense_armor.core.signals.stream

::: dense_armor.core.infrastructure.compose

---

## Details

- `iter_csv` uses Python's `csv` module, one row at a time; a list cell `"[a, b, c]"` becomes
  channels `<name>_0, <name>_1, ...` (spaces in the name become underscores); `start` / `stop`
  select a window on `time_col` and stop reading after `stop`.
- `iter_rosbag(path, topics=None)` reads ROS 1 and ROS 2 bags with the pure Python package
  `rosbags` (extra `[ros]`); `sensor_msgs/msg/JointState` becomes channels `q_<joint>`,
  `qd_<joint>`, `tau_<joint>` with the time of the message header.
- `merge_by_time` is a k-way merge with a heap; it assumes each input stream is already in time
  order. `shuffle(stream, buffer_size, seed)` shuffles streams without timestamps in a bounded
  buffer.
