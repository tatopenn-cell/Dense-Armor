# Tree: machine learning against online learning

The machine learning classes read the whole table and the online classes read one row at a time. They are two different parts of the library; this page runs both on the same synthetic data and reports how far apart their outputs are.

A tree asks a series of yes/no questions about the features and ends in a leaf that gives the answer. DecisionTreeClassifier chooses each question with the gini impurity; the Mondrian forests draw the questions at random and mix the answers along the path of each query.

## The difference, on synthetic data

10 seeds. `abs` is the largest absolute difference between the two outputs, `rel` the largest relative difference; median, 90th percentile and maximum over the seeds.

| class | kind | median | p90 | max | why |
|---|---|---|---|---|---|
| `BatchDecisionTreeClassifier` | abs | 0 | 0 | 0 | the same gini splits on the same table |
| `MondrianForestClassifier` | abs | 0.01 | 0.021 | 0.03 | the trees are sampled from the whole table at once, not grown row by row, so the random splits differ |
| `MondrianForestRegressor` | abs | 0.0167 | 0.0303 | 0.0449 | the trees are sampled from the whole table at once, not grown row by row, so the random splits differ |

Wilcoxon and A12: not computed: the machine learning and the online value are two outputs of the same formula on the same data, so the table gives their difference and not a ranking of the two.

## Run it

~~~bash
python benchmarks/compare/tree.py
~~~

The command writes `benchmarks/compare/tree.json`, the table above.
