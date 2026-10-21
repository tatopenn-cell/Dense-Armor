# Multiclass: the algorithms

A binary classifier answers yes or no. To choose among many classes, these strategies train several binary classifiers on the whole table and combine their answers: one against all the others, one for every pair of classes, or one for every bit of a codeword.

Every class reads the whole dataset with `fit` and leaves its results in attributes with a trailing underscore; `transform` or a query method applies them. The data are `synthetic` where a block makes them up.

## 1. `OneVsRestClassifier`

One binary problem per class: that class against all the others.

~~~python
import numpy as np
from dense_armor.ml.multiclass import OneVsRestClassifier
from dense_armor.ml.linear import LogisticRegression
X = [[0.0, 0.0], [0.0, 1.0], [5.0, 0.0], [5.0, 1.0], [0.0, 5.0], [1.0, 5.0]]
m = OneVsRestClassifier(lambda: LogisticRegression(l2=0.1)).fit(X, [0, 0, 1, 1, 2, 2])
print(m.predict([[0.5, 0.5], [5.0, 0.5], [0.5, 5.0]]).tolist())
~~~

~~~
[0, 1, 2]
~~~

## 2. `OneVsOneClassifier`

One binary problem per pair of classes; each votes for one class.

~~~python
import numpy as np
from dense_armor.ml.multiclass import OneVsOneClassifier
from dense_armor.ml.linear import LogisticRegression
X = [[0.0, 0.0], [0.0, 1.0], [5.0, 0.0], [5.0, 1.0], [0.0, 5.0], [1.0, 5.0]]
m = OneVsOneClassifier(lambda: LogisticRegression(l2=0.1)).fit(X, [0, 0, 1, 1, 2, 2])
print(m.predict([[0.5, 0.5], [0.5, 5.0]]).tolist())
~~~

~~~
[0, 2]
~~~

## 3. `OutputCodeClassifier`

Error-correcting output code: one binary classifier per bit of the class codewords.

~~~python
import numpy as np
from dense_armor.ml.multiclass import OutputCodeClassifier
from dense_armor.ml.linear import LogisticRegression
X = [[0.0, 0.0], [0.0, 1.0], [5.0, 0.0], [5.0, 1.0], [0.0, 5.0], [1.0, 5.0]]
m = OutputCodeClassifier(lambda: LogisticRegression(l2=0.1), code_size=8).fit(X, [0, 0, 1, 1, 2, 2])
print(m.codes_.shape)
~~~

~~~
(3, 8)
~~~

## Details

### References

Rifkin, R., Klautau, A. (2004). In defense of one-vs-all classification:
        one binary problem per class (not in the folder, bibliographic only).
    Furnkranz, J. (2002). Round robin classification, JMLR 2: one binary problem
        per pair of classes, decided by votes (not in the folder,
        bibliographic only).
    Dietterich, T. G., Bakiri, G. (1995). Solving multiclass learning problems
        via error-correcting output codes, JAIR 2: a codeword per class and the
        closest codeword in Hamming distance (not in the folder, bibliographic
        only).

### Choices stated explicitly

The class of a sketch is the exact value the sketch approximates, and the class of a statistic is the textbook formula on the whole array: both are `our choice`, because the references name no other batch algorithm.

### Development diary

The classes are new and independent of the online classes of dense_armor.utility.
