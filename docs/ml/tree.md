# Tree: the algorithms

A tree asks a series of yes/no questions about the features and ends in a leaf that gives the answer. DecisionTreeClassifier chooses each question with the gini impurity; the Mondrian forests draw the questions at random and mix the answers along the path of each query.

Every class reads the whole dataset with `fit` and leaves its results in attributes with a trailing underscore; `transform` or a query method applies them. The data are `synthetic` where a block makes them up.

## 1. `DecisionTreeClassifier`

Binary decision tree with the gini impurity, grown recursively on the whole table.

~~~python
import numpy as np
from dense_armor.ml.tree import DecisionTreeClassifier
m = DecisionTreeClassifier().fit([[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1])
print(m.predict([[0.5], [2.5]]).tolist(), m.depth_)
~~~

~~~
[0, 1] 1
~~~

## 2. `MondrianForestClassifier`

Forest of Mondrian trees; the probability is the mixture of the class histograms along the path.

~~~python
import numpy as np
from dense_armor.ml.tree import MondrianForestClassifier
X = [[0.0], [0.1], [1.0], [1.1]]
m = MondrianForestClassifier(n_trees=8, seed=1).fit(X, [0, 0, 1, 1])
print(m.predict([[0.05], [1.05]]).tolist())
~~~

~~~
[0, 1]
~~~

## 3. `MondrianForestRegressor`

Forest of Mondrian trees; the prediction is the mean of the mixture over the path.

~~~python
import numpy as np
from dense_armor.ml.tree import MondrianForestRegressor
m = MondrianForestRegressor(n_trees=8, seed=1).fit([[0.0], [0.1], [1.0], [1.1]], [0.0, 0.0, 4.0, 4.0])
print(m.predict([[0.05], [1.05]]).round().tolist())
~~~

~~~
[0.0, 4.0]
~~~

## Details

### References

Breiman, L., Friedman, J., Olshen, R., Stone, C. (1984). Classification and
        Regression Trees, Wadsworth, chapter 2: the gini impurity and the
        recursive binary splitting (not in the folder, bibliographic only).
    Lakshminarayanan, B., Roy, D. M., Teh, Y. W. (2014). Mondrian forests:
        efficient online random forests, NIPS, section 2, Algorithms 1 and 2
        (`SampleMondrianTree` and `SampleMondrianBlock`, page 4).
    Lakshminarayanan, B., Roy, D. M., Teh, Y. W. (2016). Mondrian forests for
        large-scale regression when uncertainty matters, AISTATS, Algorithm 5 and
        eq. 3: the prediction as a mixture over the nodes of the path, as the
        docstring of the online class writes it (not in the folder,
        bibliographic only).

### Choices stated explicitly

The class of a sketch is the exact value the sketch approximates, and the class of a statistic is the textbook formula on the whole array: both are `our choice`, because the references name no other batch algorithm.

### Development diary

The classes are new and independent of the online classes of dense_armor.utility.

Classes without a form:
- `HoeffdingTreeClassifier`: the batch tree that the Hoeffding tree approaches is DecisionTreeClassifier
- `HoeffdingAnytimeTreeClassifier`: the batch tree that the Hoeffding tree approaches is DecisionTreeClassifier
- `HoeffdingAdaptiveTreeClassifier`: the batch tree that the Hoeffding tree approaches is DecisionTreeClassifier; the replacement of subtrees on drift is defined on a stream
- `FIMTDDRegressor`: the reference chapter (Ikonomovska 2012, chapter 6) lies outside the folder and the formulas of the chapter are outside the folder; the model tree with drift detection is defined on a stream
- `SGTRegressor`: the reference defines the algorithm on a stream (a bound or a test on statistics that arrive one example at a time): the algorithm of the reference is written for one example or one window at a time
- `SGTClassifier`: the reference defines the algorithm on a stream (a bound or a test on statistics that arrive one example at a time): the algorithm of the reference is written for one example or one window at a time
