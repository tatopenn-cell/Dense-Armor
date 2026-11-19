# CSV files, ROS bags and arrays: the readers

`iter_csv`, `iter_rosbag` and `iter_array` read a file, a ROS bag or an array and give a stream of `Signal` and, when there is one, of a target `y`. The overview of the section is on [Streams](index.md).

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

## API reference

::: dense_armor.core.signals.stream.csv

::: dense_armor.core.signals.stream.rosbag

::: dense_armor.core.signals.stream.array

---

## Details

- `iter_csv` uses Python's `csv` module, one row at a time; a list cell `"[a, b, c]"` becomes
  channels `<name>_0, <name>_1, ...` (spaces in the name become underscores); `start` / `stop`
  select a window on `time_col` and stop reading after `stop`.
- `iter_rosbag(path, topics=None)` reads ROS 1 and ROS 2 bags with the pure Python package
  `rosbags` (extra `[ros]`); `sensor_msgs/msg/JointState` becomes channels `q_<joint>`,
  `qd_<joint>`, `tau_<joint>` with the time of the message header.
