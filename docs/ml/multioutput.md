# Multioutput: the algorithms

Some problems have several targets at once, for example the position and the speed of a joint. These classes build one model per target, or a chain in which each model also sees the targets of the earlier links, or a chain that finds the most probable set of labels as a whole.

Every class reads the whole dataset with `fit` and leaves its results in attributes with a trailing underscore; `transform` or a query method applies them. The data are `synthetic` where a block makes them up.

## 1. `PerOutputClassifier`

One binary classifier per target column, trained on `X` alone.

~~~python
import numpy as np
from dense_armor.ml.multioutput import PerOutputClassifier
from dense_armor.ml.linear import LogisticRegression
X = [[0.0], [1.0], [2.0], [3.0]]
Y = [[0, 1], [0, 1], [1, 0], [1, 0]]
m = PerOutputClassifier(lambda: LogisticRegression(l2=0.1)).fit(X, Y)
print(m.predict([[0.0], [3.0]]).tolist())
~~~

~~~
[[0, 1], [1, 0]]
~~~

## 2. `PerOutputRegressor`

One regressor per target column, trained on `X` alone.

~~~python
import numpy as np
from dense_armor.ml.multioutput import PerOutputRegressor
from dense_armor.ml.linear import LinearRegression
X = [[0.0], [1.0], [2.0]]
m = PerOutputRegressor(LinearRegression).fit(X, [[1.0, 0.0], [3.0, -1.0], [5.0, -2.0]])
print(m.predict([[3.0]]).round(4).tolist())
~~~

~~~
[[7.0, -3.0]]
~~~

## 3. `ClassifierChain`

Chained binary classifiers: link `j` sees `X` and the targets of the earlier links.

~~~python
import numpy as np
from dense_armor.ml.multioutput import ClassifierChain
from dense_armor.ml.linear import LogisticRegression
X = [[0.0], [1.0], [2.0], [3.0]]
Y = [[0, 1], [0, 1], [1, 0], [1, 0]]
m = ClassifierChain(lambda: LogisticRegression(l2=0.1), order=[0, 1]).fit(X, Y)
print(m.predict([[0.0], [3.0]]).tolist())
~~~

~~~
[[0, 1], [1, 0]]
~~~

## 4. `RegressorChain`

Chained regressors: link `j` sees `X` and the targets of the earlier links.

~~~python
import numpy as np
from dense_armor.ml.multioutput import RegressorChain
from dense_armor.ml.linear import LinearRegression
X = [[0.0], [1.0], [2.0]]
m = RegressorChain(lambda: LinearRegression(l2=1e-9), order=[0, 1]).fit(X, [[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
print(m.predict([[3.0]]).round(2).tolist())
~~~

~~~
[[7.0, 14.0]]
~~~

## 5. `ProbabilisticClassifierChain`

Probabilistic classifier chain: the joint mode of `P(y | x)` (Dembczynski et al. 2010).

~~~python
import numpy as np
from dense_armor.ml.multioutput import ProbabilisticClassifierChain
from dense_armor.ml.linear import LogisticRegression
X = [[0.0], [1.0], [2.0], [3.0]]
Y = [[0, 1], [0, 1], [1, 0], [1, 0]]
m = ProbabilisticClassifierChain(lambda: LogisticRegression(l2=0.1), order=[0, 1]).fit(X, Y)
print(m.predict([[0.0], [3.0]]).tolist())
~~~

~~~
[[0, 1], [1, 0]]
~~~

## Details

### References

Spyromitros-Xioufis, E. et al. (2016). Multi-target regression via input
        space expansion, Section 2 (not in the folder, bibliographic only).
    Read, J. et al. (2009). Classifier chains for multi-label classification
        (not in the folder, bibliographic only).
    Dembczynski, K., Cheng, W., Hullermeier, E. (2010). Bayes optimal
        multilabel classification via probabilistic classifier chains, Eq. (10)
        to (12) (not in the folder, bibliographic only).

### Choices stated explicitly

The class of a sketch is the exact value the sketch approximates, and the class of a statistic is the textbook formula on the whole array: both are `our choice`, because the references name no other batch algorithm.

### Development diary

The classes are new and independent of the online classes of dense_armor.utility.
