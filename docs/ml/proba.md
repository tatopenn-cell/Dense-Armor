# Proba: the algorithms

A distribution summarises a sample in a few numbers. These classes keep the posterior of the Beta, Gaussian and Multinomial families: the prior counts plus the counts observed in the whole sample.

Every class reads the whole dataset with `fit` and leaves its results in attributes with a trailing underscore; `transform` or a query method applies them. The data are `synthetic` where a block makes them up.

## 1. `Beta`

Beta posterior of a Bernoulli sample: `a + successes`, `b + failures`.

~~~python
import numpy as np
from dense_armor.ml.proba import Beta
d = Beta().fit([1, 1, 0, 1])
print(d.a_, d.b_, round(d.mean(), 4))
~~~

~~~
4.0 2.0 0.6667
~~~

## 2. `Gaussian`

Gaussian of a sample: the mean and the variance (`n - 1`), pooled with a prior of weight `n`.

~~~python
import numpy as np
from dense_armor.ml.proba import Gaussian
d = Gaussian().fit([1.0, 2.0, 3.0, 4.0])
print(d.mean_, round(d.var_, 4), d.n_)
~~~

~~~
2.5 1.6667 4.0
~~~

## 3. `Multinomial`

Dirichlet posterior of a categorical sample: the prior counts plus the observed counts.

~~~python
import numpy as np
from dense_armor.ml.proba import Multinomial
d = Multinomial(alpha=1.0, n_categories=3).fit([0, 0, 2])
print(d.counts_.tolist(), d.pmf(0))
~~~

~~~
[3.0, 1.0, 2.0] 0.5
~~~

## Details

### References

Murphy, K. P. (2007). Conjugate Bayesian analysis of the Gaussian
        distribution (not in the folder, bibliographic only).
    The Beta posterior of a Bernoulli sample and the Dirichlet posterior of a
        categorical sample are textbook conjugate pairs: the prior counts plus
        the observed counts.

### Choices stated explicitly

The class of a sketch is the exact value the sketch approximates, and the class of a statistic is the textbook formula on the whole array: both are `our choice`, because the references name no other batch algorithm.

### Development diary

The classes are new and independent of the online classes of dense_armor.utility.
