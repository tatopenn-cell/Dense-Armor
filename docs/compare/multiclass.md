# Multiclass: machine learning against online learning

The machine learning classes read the whole table and the online classes read one row at a time. They are two different parts of the library; this page runs both on the same synthetic data and reports how far apart their outputs are.

A binary classifier answers yes or no. To choose among many classes, these strategies train several binary classifiers on the whole table and combine their answers: one against all the others, one for every pair of classes, or one for every bit of a codeword.

## The difference, on synthetic data

10 seeds. `abs` is the largest absolute difference between the two outputs, `rel` the largest relative difference; median, 90th percentile and maximum over the seeds.

| class | kind | median | p90 | max | why |
|---|---|---|---|---|---|
| `OneVsRestClassifier` | abs | 0.0525 | 0.082 | 0.1 | difference between the test accuracies; the online binary models see each row once |
| `OneVsOneClassifier` | abs | 0.0475 | 0.0865 | 0.1 | difference between the test accuracies; the online binary models see each row once |
| `OutputCodeClassifier` | abs | 0.0325 | 0.0805 | 0.085 | difference between the test accuracies; the online binary models see each row once |

Wilcoxon and A12: not computed: the machine learning and the online value are two outputs of the same formula on the same data, so the table gives their difference and not a ranking of the two.

## Run it

~~~bash
python benchmarks/compare/multiclass.py
~~~

The command writes `benchmarks/compare/multiclass.json`, the table above.
