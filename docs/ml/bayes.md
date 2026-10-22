# Bayes: the algorithms

A naive Bayes classifier counts how often each feature value goes with each class and turns the counts into probabilities. Here the counts are taken over the whole table, with a small pseudo-count so that no probability is zero.

Every class reads the whole dataset with `fit` and leaves its results in attributes with a trailing underscore; `transform` or a query method applies them. The data are `synthetic` where a block makes them up.

## 1. `MultinomialNB`

Multinomial naive Bayes on count features.

~~~python
import numpy as np
from dense_armor.ml.bayes import MultinomialNB
X = [[3, 0], [2, 1], [0, 3], [1, 2]]
print(MultinomialNB().fit(X, [0, 0, 1, 1]).predict([[4, 0], [0, 4]]).tolist())
~~~

~~~
[0, 1]
~~~

## 2. `BernoulliNB`

Bernoulli naive Bayes on presence features (a value above 0 is a presence).

~~~python
import numpy as np
from dense_armor.ml.bayes import BernoulliNB
X = [[1, 0], [1, 1], [0, 1], [0, 1]]
print(BernoulliNB().fit(X, [0, 0, 1, 1]).predict([[1, 0], [0, 1]]).tolist())
~~~

~~~
[0, 1]
~~~

## 3. `CategoricalNB`

Categorical naive Bayes: every feature takes one of a few category codes.

~~~python
import numpy as np
from dense_armor.ml.bayes import CategoricalNB
X = [[0, 1], [0, 0], [1, 1], [1, 0]]
print(CategoricalNB().fit(X, [0, 0, 1, 1]).predict([[0, 1], [1, 0]]).tolist())
~~~

~~~
[0, 1]
~~~

## 4. `ComplementNB`

Complement naive Bayes (Rennie et al. 2003): weights from the other classes.

~~~python
import numpy as np
from dense_armor.ml.bayes import ComplementNB
X = [[3, 0], [2, 1], [0, 3], [1, 2]]
print(ComplementNB().fit(X, [0, 0, 1, 1]).predict([[4, 0], [0, 4]]).tolist())
~~~

~~~
[0, 1]
~~~

## Details

### References

McCallum, A., Nigam, K. (1998). A comparison of event models for naive
        Bayes text classification, page 4 (multinomial and Bernoulli models)
        (not in the folder, bibliographic only).
    Rennie, J. D. M. et al. (2003). Tackling the poor assumptions of naive
        Bayes text classifiers, pages 6-7 (complement naive Bayes) (not in
        the folder, bibliographic only).
    Laplace smoothing with a pseudo-count `alpha` is the textbook form.

### Choices stated explicitly

The class of a sketch is the exact value the sketch approximates, and the class of a statistic is the textbook formula on the whole array: both are `our choice`, because the references name no other batch algorithm.

### Development diary

The classes are new and independent of the online classes of dense_armor.utility.
