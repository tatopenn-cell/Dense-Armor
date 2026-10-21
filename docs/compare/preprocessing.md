# Preprocessing: machine learning against online learning

The machine learning classes read the whole table and the online classes read one row at a time. They are two different parts of the library; this page runs both on the same synthetic data and reports how far apart their outputs are.

Before a model sees the data, the table is rescaled, some columns are dropped, the joint records are turned into velocities and the text is turned into numbers. Each class reads the whole table with `fit` and applies the result with `transform`.

## The difference, on synthetic data

10 seeds. `abs` is the largest absolute difference between the two outputs, `rel` the largest relative difference; median, 90th percentile and maximum over the seeds.

| class | kind | median | p90 | max | why |
|---|---|---|---|---|---|
| `StandardScaler` | abs | 3.77e-15 | 7.22e-15 | 8.22e-15 |  |
| `EWScaler` | abs | 8.88e-16 | 1.71e-15 | 3.11e-15 |  |
| `RobustScaler` | abs | 0 | 0 | 0 |  |
| `MinMaxScaler` | abs | 0 | 0 | 0 |  |
| `VarianceThreshold` | abs | 0 | 0 | 0 |  |
| `SelectKBest` | abs | 0 | 0 | 0 |  |
| `JointDerivatives` | abs | 1.8e-09 | 5.68e-09 | 6.52e-09 | rounding of a polynomial through four samples 0.01 s apart |
| `JointPower` | abs | 0 | 0 | 0 |  |
| `Tokenizer` | abs | 0 | 0 | 0 |  |
| `BagOfWords` | abs | 0 | 0 | 0 |  |
| `TFIDF` | abs | 0 | 0 | 0 |  |
| `FeatureHasher` | abs | 0 | 0 | 0 |  |
| `TextVectorizer` | abs | 7.84e-08 | 7.84e-08 | 7.84e-08 | the online matrix is float32, the batch one float64 |

Wilcoxon and A12: not computed: the classic and the online value are two outputs of the same formula on the same data, so the table gives their difference and not a ranking of the two.

## Run it

~~~bash
python benchmarks/compare/preprocessing.py
~~~

The command writes `benchmarks/compare/preprocessing.json`, the table above.
