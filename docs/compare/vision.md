# Vision: machine learning against online learning

The machine learning classes read the whole table and the online classes read one row at a time. They are two different parts of the library; this page runs both on the same synthetic data and reports how far apart their outputs are.

Vision here works on a pair of frames and on tables of features. DenseFlow gives the motion of every pixel between two frames; RandomProjection and PCA reduce a table of features to fewer columns.

## The difference, on synthetic data

10 seeds. `abs` is the largest absolute difference between the two outputs, `rel` the largest relative difference; median, 90th percentile and maximum over the seeds.

| class | kind | median | p90 | max | why |
|---|---|---|---|---|---|
| `DenseFlow` | abs | 2.28e-07 | 3.13e-07 | 3.4e-07 | the same iteration on the same pair of frames |
| `RandomProjection` | abs | 1.78e-15 | 2.26e-15 | 2.66e-15 | the same matrix from the same seed |
| `PCA` | abs | 0.000272 | 0.000427 | 0.000798 | the eigenvectors of the covariance of the whole table, against the rule that approaches them one row at a time |

Wilcoxon and A12: not computed: the machine learning and the online value are two outputs of the same formula on the same data, so the table gives their difference and not a ranking of the two.

## Run it

~~~bash
python benchmarks/compare/vision.py
~~~

The command writes `benchmarks/compare/vision.json`, the table above.
