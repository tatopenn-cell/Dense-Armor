# Ensemble: the algorithms

An ensemble combines several models: bagging and boosting train many copies, voting and stacking combine different models, and the exponentially weighted average follows the member with the smallest loss.

Every class reads the whole dataset with `fit` and leaves its results in attributes with a trailing underscore; `transform` or a query method applies them. The data are `synthetic` where a block makes them up.

## 1. `BaggingClassifier`

Bagging: one model per bootstrap sample, probabilities averaged.

~~~python
import numpy as np
from dense_armor.ml.ensemble import BaggingClassifier
from dense_armor.ml.tree import DecisionTreeClassifier
X = [[0.0], [0.2], [1.0], [1.2]]
m = BaggingClassifier(DecisionTreeClassifier(), n_models=5, seed=1).fit(X, [0, 0, 1, 1])
print(m.predict([[0.1], [1.1]]).tolist())
~~~

~~~
[0, 1]
~~~

## 2. `BoostingClassifier`

AdaBoost with the multiclass weights SAMME (Zhu et al. 2009).

~~~python
import numpy as np
from dense_armor.ml.ensemble import BoostingClassifier
from dense_armor.ml.tree import DecisionTreeClassifier
X = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
m = BoostingClassifier(DecisionTreeClassifier(max_depth=1), n_models=5).fit(X, [0, 0, 1, 1, 0, 0])
print(m.predict([[0.5], [2.5], [4.5]]).tolist())
~~~

~~~
[0, 1, 0]
~~~

## 3. `ExponentiallyWeightedAverage`

Weights `exp(-eta L_m)` from the total loss of each model on the data (Hedge).

~~~python
import numpy as np
from dense_armor.ml.ensemble import ExponentiallyWeightedAverage
P = [[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]]
m = ExponentiallyWeightedAverage(eta=1.0).fit(P, [1.0, 2.0, 3.0])
print(m.weights_.round(3).tolist())
~~~

~~~
[1.0, 0.0]
~~~

## 4. `VotingClassifier`

A fixed list of models; the hard vote or the mean of the probabilities.

~~~python
import numpy as np
from dense_armor.ml.ensemble import VotingClassifier
from dense_armor.ml.tree import DecisionTreeClassifier
from dense_armor.ml.neighbors import KNNClassifier
X = [[0.0], [0.2], [1.0], [1.2]]
m = VotingClassifier([DecisionTreeClassifier(), KNNClassifier(n_neighbors=1)], mode="soft").fit(X, [0, 0, 1, 1])
print(m.predict([[0.1], [1.1]]).tolist())
~~~

~~~
[0, 1]
~~~

## 5. `StackingClassifier`

A meta-model fitted on the out-of-fold probabilities of the base models (Wolpert 1992).

~~~python
import numpy as np
from dense_armor.ml.ensemble import StackingClassifier
from dense_armor.ml.tree import DecisionTreeClassifier
from dense_armor.ml.neighbors import KNNClassifier
X = [[0.0], [0.1], [0.2], [0.3], [1.0], [1.1], [1.2], [1.3]]
m = StackingClassifier([KNNClassifier(n_neighbors=1)], DecisionTreeClassifier(), cv=2).fit(X, [0, 0, 0, 0, 1, 1, 1, 1])
print(m.predict([[0.15], [1.15]]).tolist())
~~~

~~~
[0, 1]
~~~

## Details

### References

Breiman, L. (1996). Bagging predictors, Machine Learning 24(2): bootstrap
        samples and the mean of the models (not in the folder, bibliographic
        only).
    Freund, Y., Schapire, R. E. (1997). A decision-theoretic generalization of
        on-line learning and an application to boosting, JCSS 55(1): the
        Hedge weights `exp(-eta L)` and AdaBoost (not in the folder,
        bibliographic only). Zhu, J. et al. (2009). Multi-class AdaBoost
        (SAMME) for the multiclass weights (not in the folder, bibliographic
        only).
    Wolpert, D. H. (1992). Stacked generalization, Neural Networks 5(2): a
        meta-model on out-of-fold predictions (not in the folder,
        bibliographic only).

### Choices stated explicitly

The class of a sketch is the exact value the sketch approximates, and the class of a statistic is the textbook formula on the whole array: both are `our choice`, because the references name no other batch algorithm.

### Development diary

The classes are new and independent of the online classes of dense_armor.utility.
