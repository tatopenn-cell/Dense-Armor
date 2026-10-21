# Neighbors: the algorithms

A nearest-neighbour model answers a question about a new row by looking at the rows of the table that are closest to it: the most common class among them, or the average of their targets.

Every class reads the whole dataset with `fit` and leaves its results in attributes with a trailing underscore; `transform` or a query method applies them. The data are `synthetic` where a block makes them up.

## 1. `KNNClassifier`

k-nearest-neighbour classifier with a uniform or inverse-distance vote.

~~~python
import numpy as np
from dense_armor.ml.neighbors import KNNClassifier
m = KNNClassifier(n_neighbors=1).fit([[0.0], [1.0], [5.0]], [0, 0, 1])
print(m.predict([[0.4], [4.0]]).tolist())
~~~

~~~
[0, 1]
~~~

## 2. `KNNRegressor`

k-nearest-neighbour regressor: the weighted mean of the neighbours' targets.

~~~python
import numpy as np
from dense_armor.ml.neighbors import KNNRegressor
print(KNNRegressor(n_neighbors=2).fit([[0.0], [1.0], [9.0]], [0.0, 2.0, 10.0]).predict([[0.4]]).tolist())
~~~

~~~
[1.0]
~~~

## Details

### References

Cover, T., Hart, P. (1967). Nearest neighbor pattern classification,
        IEEE Trans. Information Theory 13(1) (not in the folder,
        bibliographic only). The Minkowski distance of order `p` and the
        uniform or inverse-distance vote are textbook definitions.

### Choices stated explicitly

The class of a sketch is the exact value the sketch approximates, and the class of a statistic is the textbook formula on the whole array: both are `our choice`, because the references name no other batch algorithm.

### Development diary

The classes are new and independent of the online classes of dense_armor.utility.
