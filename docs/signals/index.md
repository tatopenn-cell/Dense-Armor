# Signals: data in, and what to do with it, one sample at a time

Everything that comes before a model lives in the folder `core/signals`: the files, bags and arrays that become a stream of readings, the streams made on purpose to test a model, the statistics that keep a few numbers for everything seen so far, the preprocessing that puts raw readings on a common scale, and the vision tools that read a video one frame at a time.

## The map of signals

Click a page to open it, or the job on the left (or the button at the top) to go back to the [map of the library](../index.md#the-library-folder-by-folder).

<object type="image/svg+xml" data="../assets/maps/signals.svg" style="width:100%"></object>

## A first example: the payments of a day

A bank receives payments one at a time and wants their average and their spread without keeping the list. `RunningMoments` keeps a few numbers and updates them with each payment. Five payments of 120, 80, 100, 95 and 105:

```python
from dense_armor.core.signals.stats.moments import RunningMoments

m = RunningMoments()
for amount in [120.0, 80.0, 100.0, 95.0, 105.0]:
    m.learn_one({"x": amount})
print(m.mean, m.var)
```

```
100.0 212.5
```

The mean is 100.0 and the variance 212.5 (the variance of a sample, with the divisor `n - 1`). The same two numbers would come from the whole list, but `RunningMoments` never stored it: it is the idea of every tool of the section.

## The pages of the section

| Page | What it is for |
|---|---|
| [Streams](../stream/index.md) | CSV files, ROS bags and arrays as one stream of readings, merged by time |
| [Datasets](../stream/datasets.md) | streams whose change or fault happens at a known moment, to test a model |
| [Statistics](../stats/index.md) | running moments, quantiles, sketches and the other numbers that summarise a stream |
| [Preprocessing](../preprocessing/index.md) | raw readings on a common scale, and speed, acceleration and power from them |
| [Vision](../vision/index.md) | a video read one frame at a time, in fixed memory, with no pretrained model |
