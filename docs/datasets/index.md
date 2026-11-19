# Datasets: streams with a known change

The datasets of the library are streams whose rule is known: a binary stream whose concept changes once (`DriftStream`), a Panda arm with a fault at a known time (`SyntheticArm`), and the reader of the CASPER recording of an industrial arm (`Casper`). They are what a drift detector or a protected model is tested on. The streams from files, bags and arrays are on the [Streams](../stream/index.md) page.

## The map of datasets

Click a page to open it, or the job on the left (or the button at the top) to go back to the [map of the library](../index.md#the-library-folder-by-folder).

<object type="image/svg+xml" data="../assets/maps/datasets.svg" style="width:100%"></object>

## A first example: payments whose rule is replaced

A stream of 2000 payments with two numbers each, whose rule is replaced once, at half of the stream. `DriftStream` makes it, and it also says where the change is and which concept each payment comes from.

```python
from dense_armor.core.signals.datasets import DriftStream

ds = DriftStream(n_samples=2000, n_features=2, n_drifts=1, kind="sudden", seed=0)
X, y = ds.data()
print(X.shape, ds.drift_positions)
print(ds.concepts_[:1000].mean(), ds.concepts_[1000:].mean())
```

```
(2000, 2) [0.5]
0.0 1.0
```

The stream has 2000 samples of two numbers, the change sits at the position 0.5 (sample 1000), and the concept of the first 1000 samples is 0 and of the last 1000 is 1. A drift detector or a protected model is tested on a stream like this, where the answer is known.

## The pages of the section

| Page | What it is for |
|---|---|
| [DriftStream](drift_stream.md) | a binary stream whose concept changes, suddenly, gradually or incrementally |
| [SyntheticArm](synthetic_arm.md) | a Panda arm with a fault, a friction or a payload, at a known time |
| [Casper](casper.md) | the reader of the CASPER recording of an industrial arm |
