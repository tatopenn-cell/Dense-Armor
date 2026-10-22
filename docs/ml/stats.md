# Stats: the algorithms

A statistic such as a mean, a variance or a correlation is computed on a whole table at once. This page lists the statistics of the machine learning part of the library: covariance and correlation, moments, the exponentially weighted mean and variance, rolling windows, the rolling median and the shrunk covariance.

Every class reads the whole dataset with `fit` and leaves its results in attributes with a trailing underscore; `transform` or a query method applies them. The data are `synthetic` where a block makes them up.

## 1. `Covariance`

Sample covariance (`n - 1`) and Pearson correlation of two channels.

~~~python
import numpy as np
from dense_armor.ml.stats import Covariance
m = Covariance().fit([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
print(round(m.cov_, 6), round(m.corr_, 6))
~~~

~~~
2.0 1.0
~~~

## 2. `Correlation`

Pearson correlation of two channels over the whole dataset.

~~~python
import numpy as np
from dense_armor.ml.stats import Correlation
print(Correlation().fit([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]).corr_)
~~~

~~~
-1.0
~~~

## 3. `RollingCovariance`

Sample covariance over the last `window` pairs and over every window.

~~~python
import numpy as np
from dense_armor.ml.stats import RollingCovariance
m = RollingCovariance(window=3).fit([1., 2., 3., 4.], [1., 2., 3., 5.])
print(m.cov_series_.shape)
~~~

~~~
(2,)
~~~

## 4. `RollingCorrelation`

Pearson correlation over the last `window` pairs and over every window.

~~~python
import numpy as np
from dense_armor.ml.stats import RollingCorrelation
m = RollingCorrelation(window=3).fit([1., 2., 3.], [1., 2., 3.])
print(round(m.corr_, 6))
~~~

~~~
1.0
~~~

## 5. `Autocorrelation`

Sample autocorrelation at a fixed lag: the standard estimator.

~~~python
import numpy as np
from dense_armor.ml.stats import Autocorrelation
print(round(Autocorrelation(lag=1).fit([1., 2., 1., 2., 1.]).value_, 4))
~~~

~~~
-0.8
~~~

## 6. `Moments`

Mean, variance (`n - 1`), skewness, excess kurtosis, min, max, range.

~~~python
import numpy as np
from dense_armor.ml.stats import Moments
m = Moments().fit([1.0, 2.0, 3.0, 4.0])
print(m.mean_, round(m.var_, 4), m.ptp_)
~~~

~~~
2.5 1.6667 3.0
~~~

## 7. `ColumnMoments`

The moments of every column of a matrix.

~~~python
import numpy as np
from dense_armor.ml.stats import ColumnMoments
m = ColumnMoments().fit([[1., 10.], [3., 20.]])
print(m.mean_.tolist())
~~~

~~~
[2.0, 15.0]
~~~

## 8. `EWStats`

Exponentially weighted mean and variance of the whole series.

~~~python
import numpy as np
from dense_armor.ml.stats import EWStats
print(round(EWStats(alpha=0.5).fit([1.0, 3.0]).mean_, 4))
~~~

~~~
2.0
~~~

## 9. `RollingMedian`

Median over every window and over the last window, per column.

~~~python
import numpy as np
from dense_armor.ml.stats import RollingMedian
m = RollingMedian(window=3).fit([1., 5., 2., 8.])
print(m.median_.tolist())
~~~

~~~
[5.0]
~~~

## 10. `ShrinkageCovariance`

Linear shrinkage toward `(1/p) I` on the sample covariance of all rows.

~~~python
import numpy as np
from dense_armor.ml.stats import ShrinkageCovariance
import numpy as np
X = np.random.default_rng(0).normal(size=(30, 3))
print(ShrinkageCovariance(3).fit(X).shrunk_.shape)
~~~

~~~
(3, 3)
~~~

## Details

### References

Chan, T. F., Golub, G. H., LeVeque, R. J. (1979). Updating formulae and a pairwise
        algorithm for computing sample variances, Compstat: the two-pass formulas for the
        mean, the variance and the covariance (not in the folder, bibliographic only).
    West, D. H. D. (1979). Updating mean and variance estimates, CACM 22(9):
        the exponentially weighted recursion (not in the folder,
        bibliographic only).
    Bodnar, T., Gupta, A. K., Parolya, N. (2014). Optimal linear shrinkage
        estimator for large dimensional precision matrix, Eq. (4.3) and (4.4),
        page 12 (not in the folder, bibliographic only).

### Choices stated explicitly

The class of a sketch is the exact value the sketch approximates, and the class of a statistic is the textbook formula on the whole array: both are `our choice`, because the references name no other batch algorithm.

### Development diary

The classes are new and independent of the online classes of dense_armor.utility.

Classes without a form:
- `DDSketch`: a sketch is the same for every order of the rows: its form on the whole table is the online class itself, and the exact quantile differs from the algorithm of Masson et al.
- `TDigest`: a sketch is the same for every order of the rows: its form on the whole table is the online class itself, and the exact quantile differs from the algorithm of Dunning and Ertl
- `CountMinSketch`: a sketch is the same for every order of the rows: its form on the whole table is the online class itself, and the exact count differs from the algorithm of Cormode and Muthukrishnan
- `HyperLogLog`: a sketch is the same for every order of the rows: its form on the whole table is the online class itself, and the exact distinct count differs from the algorithm of Flajolet et al.
- `BloomFilter`: a sketch is the same for every order of the rows: its form on the whole table is the online class itself, and exact membership differs from the filter of Bloom
- `SpaceSaving`: a sketch is the same for every order of the rows: its form on the whole table is the online class itself, and the exact counts are not the algorithm of Metwally et al.
