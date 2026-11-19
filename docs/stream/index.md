# Streams: files, bags and arrays as one stream of readings

Robot data comes in many forms: files, recordings, arrays, live sensors. This page turns all of them into the same stream of readings, which every tool in the library then reads one at a time.

## The map of streams

Click a page to open it, or the job on the left (or the button at the top) to go back to the [map of the library](../index.md#the-library-folder-by-folder).

<object type="image/svg+xml" data="../assets/maps/stream.svg" style="width:100%"></object>

## A first example: the payments of a file

A file of payments, one row per payment, with the time and the amount. `iter_csv` reads it one row at a time and gives each row as a `Signal` with its timestamp, the same kind of sample that every estimator of the library learns from.

```python
from pathlib import Path
from dense_armor.core.signals.stream import iter_csv

path = Path("payments.csv")
path.write_text("time,amount\n0,120.0\n1,80.0\n2,100.0\n")
for sig, _ in iter_csv(path, time_col="time"):
    print(sig.t, float(sig["amount"]))
```

```
0.0 120.0
1.0 80.0
2.0 100.0
```

The file is read one row at a time, so a file of a million payments uses the memory of three. The same stream, with the same kind of `Signal`, comes out of a ROS bag, of an array or of several sensors merged in time order.

## The pages of the section

| Page | What it is for |
|---|---|
| [CSV files, ROS bags and arrays](readers.md) | `iter_csv`, `iter_rosbag` and `iter_array`: the readers |
| [Merge by time and shuffle](merge.md) | `merge_by_time` puts several streams in one timeline; `shuffle` mixes a stream in a bounded buffer |
| [Choosing and reshaping channels](channels.md) | `Select`, `Discard` and `FuncTransformer` keep and change the channels of a sample |

The streams made on purpose to test a model, whose change or fault happens at a known moment, are in [Datasets](../datasets/index.md).
