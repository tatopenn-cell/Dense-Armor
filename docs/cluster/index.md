# Clustering: incremental partitioning and density-based data stream models

This page shows how a system can automatically group similar data points together in real time, one by one, without needing any pre-assigned categories. It covers standard tracking, groups that shift positions over time, filtering out noise, and translating image patches into visual signatures.

## The map of clustering

Click a page to open it, or the job on the left (or the button at the top) to go back to the [map of the library](../index.md#the-library-folder-by-folder).

<object type="image/svg+xml" data="../assets/maps/cluster.svg" style="width:100%"></object>

## A first example: kinds of customers

A bank looks at its customers through two numbers: how many payments they make in a month and the average amount of a payment. Nobody tells it the kinds of customers. It sees 900 customers, one at a time, and asks `OnlineKMeans` for 3 groups; the customers are drawn around three kinds, with `(2, 30)`, `(12, 60)` and `(30, 400)` payments and euros.

```python
import numpy as np
from dense_armor.core.online.cluster.kmeans import OnlineKMeans

rng = np.random.default_rng(0)
kinds = np.array([[2.0, 30.0], [12.0, 60.0], [30.0, 400.0]])
km = OnlineKMeans(k=3)
for _ in range(900):
    customer = kinds[rng.integers(0, 3)] + rng.normal(0, [1.0, 10.0])
    km.learn_one({"x": customer.tolist()})
print(sorted([round(float(v), 1) for v in c] for c in km.centers_))
```

```
[[2.3, 28.4], [11.1, 59.1], [30.0, 399.4]]
```

Each centre is the mean of the customers that joined its group, so the three printed centres are the three kinds that the bank did not know, close to the ones the customers were drawn from. The groups are in the order they were found.

## The pages of the section

| Page | What it is for |
|---|---|
| [OnlineKMeans](kmeans.md) | k groups, one point at a time; a half-life lets the groups move |
| [DenStream](denstream.md) | groups of any shape, with outliers left out |
| [Visual vocabulary](visual_vocabulary.md) | image patches grouped by `OnlineKMeans` into words, a frame as a histogram of words |
