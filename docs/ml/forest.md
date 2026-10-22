# Forest: the algorithms

A forest asks many trees and averages their answers. RandomForestClassifier grows each tree on a bootstrap sample with a random subset of features; ExtraTreesRegressor uses the whole sample and a random threshold at each split.

Every class reads the whole dataset with `fit` and leaves its results in attributes with a trailing underscore; `transform` or a query method applies them. The data are `synthetic` where a block makes them up.

## 1. `RandomForestClassifier`

Random forest: trees on bootstrap samples with `sqrt(d)` random features per split.

~~~python
import numpy as np
from dense_armor.ml.forest import RandomForestClassifier
X = [[i / 10, i % 2] for i in range(6)] + [[1.0 + i / 10, i % 2] for i in range(6)]
m = RandomForestClassifier(n_trees=15, seed=2).fit(X, [0] * 6 + [1] * 6)
print(m.predict([[0.2, 0.5], [1.3, 0.5]]).tolist())
~~~

~~~
[0, 1]
~~~

## 2. `ExtraTreesRegressor`

Extremely randomised trees: the whole sample and a random threshold at each split.

~~~python
import numpy as np
from dense_armor.ml.forest import ExtraTreesRegressor
X = [[0.0], [0.2], [1.0], [1.2]]
m = ExtraTreesRegressor(n_trees=5, seed=3).fit(X, [0.0, 0.0, 4.0, 4.0])
print(m.predict([[0.1], [1.1]]).round(1).tolist())
~~~

~~~
[0.0, 4.0]
~~~

## Details

### References

Breiman, L. (2001). Random forests, Machine Learning 45(1): a tree per
        bootstrap sample, a random subset of features at each split (not in
        the folder, bibliographic only).
    Geurts, P., Ernst, D., Wehenkel, L. (2006). Extremely randomized trees,
        Machine Learning 63(1): the whole sample and a random threshold at
        each split (not in the folder, bibliographic only).

### Choices stated explicitly

The class of a sketch is the exact value the sketch approximates, and the class of a statistic is the textbook formula on the whole array: both are `our choice`, because the references name no other batch algorithm.

### Development diary

The classes are new and independent of the online classes of dense_armor.utility.

Classes without a form:
- `AdaptiveRandomForestClassifier`: the batch form is RandomForestClassifier: the drift detectors and the background trees have no batch counterpart
- `AggregatedMondrianForestClassifier`: the aggregation over the prunings is defined online (Mourtada et al.); the batch Mondrian forest is in the tree module
- `AggregatedMondrianForestRegressor`: the aggregation over the prunings is defined online (Mourtada et al.); the batch Mondrian forest is in the tree module
