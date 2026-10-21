# Neighbors: machine learning against online learning

The machine learning classes read the whole table and the online classes read one row at a time. They are two different parts of the library; this page runs both on the same synthetic data and reports how far apart their outputs are.

A nearest-neighbour model answers a question about a new row by looking at the rows of the table that are closest to it: the most common class among them, or the average of their targets.

## The difference, on synthetic data

10 seeds. `abs` is the largest absolute difference between the two outputs, `rel` the largest relative difference; median, 90th percentile and maximum over the seeds.

| class | kind | median | p90 | max | why |
|---|---|---|---|---|---|
| `KNNClassifier` | abs | 0 | 0 | 0 | the online model searches a window as large as the table, so the two see the same rows |
| `KNNRegressor` | abs | 3.3e-10 | 4.81e-10 | 5.6e-10 | the online model searches a window as large as the table, so the two see the same rows |

Wilcoxon and A12: not computed: the machine learning and the online value are two outputs of the same formula on the same data, so the table gives their difference and not a ranking of the two.

## Run it

~~~bash
python benchmarks/compare/neighbors.py
~~~

The command writes `benchmarks/compare/neighbors.json`, the table above.
