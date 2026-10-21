# Anomaly: the algorithms

An anomaly detector marks the values that do not look like their neighbours. Four rules (Hampel, Tukey, Chauvenet, sigma clipping) judge each value of a series against its centred window; the Local Outlier Factor compares the density around a row with the density around its neighbours.

Every class reads the whole dataset with `fit` and leaves its results in attributes with a trailing underscore; `transform` or a query method applies them. The data are `synthetic` where a block makes them up.

## 1. `ChauvenetScorer`

Chauvenet criterion on the centred window.

~~~python
import numpy as np
from dense_armor.ml.anomaly import ChauvenetScorer
print(ChauvenetScorer(radius=4).fit([0.0, 0.0, 0.0, 0.0, 8.0]).anomalies_)
~~~

~~~
[4]
~~~

## 2. `HampelFilter`

Hampel filter on the centred window: median and scaled MAD.

~~~python
import numpy as np
from dense_armor.ml.anomaly import HampelFilter
m = HampelFilter(radius=2).fit([1.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0])
print(m.anomalies_, m.cleaned_[4])
~~~

~~~
[4] 1.0
~~~

## 3. `LocalOutlierFactor`

Local Outlier Factor of Breunig et al. (2000), Definitions 3 to 7, on the whole table.

~~~python
import numpy as np
from dense_armor.ml.anomaly import LocalOutlierFactor
X = [[0.0], [1.0], [2.0], [3.0], [10.0]]
m = LocalOutlierFactor(n_neighbors=2).fit(X)
print(m.lof_.round(2).tolist())
~~~

~~~
[1.0, 1.0, 1.0, 1.0, 5.0]
~~~

## 4. `SigmaClipScorer`

Iterative sigma clipping on the centred window without the value itself.

~~~python
import numpy as np
from dense_armor.ml.anomaly import SigmaClipScorer
m = SigmaClipScorer(radius=4).fit([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 50.0])
print(m.anomalies_)
~~~

~~~
[8]
~~~

## 5. `TukeyScorer`

Tukey fences on the centred window: `[Q1 - k IQR, Q3 + k IQR]`.

~~~python
import numpy as np
from dense_armor.ml.anomaly import TukeyScorer
m = TukeyScorer(radius=2).fit([1.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0])
print(m.anomalies_)
~~~

~~~
[4]
~~~

## Details

### References

Breunig, M. M., Kriegel, H.-P., Ng, R. T., Sander, J. (2000). LOF:
        Identifying Density-Based Local Outliers. In Proceedings of the 2000
        ACM SIGMOD International Conference on Management of Data, pages
        93-104. Definitions 3 to 7 in Section 4 (not in the folder,
        bibliographic only).
    Chauvenet (1863), Tukey (1977), Hampel (1974), Davies and Gather (1993):
        the four rules on a centred local window, as the docstrings of the
        batch functions of the library write them (not in the folder,
        bibliographic only).

### Choices stated explicitly

The class of a sketch is the exact value the sketch approximates, and the class of a statistic is the textbook formula on the whole array: both are `our choice`, because the references name no other batch algorithm.

### Development diary

The classes are new and independent of the online classes of dense_armor.utility.

Classes without a form:
- `StreamingDeviationScorer`: the reference defines the score on the window of the values learned before the one scored, or on statistics that move at each new value: the algorithm of the reference is written for one example or one window at a time
- `StreamingDeviationDetector`: the reference defines the score on the window of the values learned before the one scored, or on statistics that move at each new value: the algorithm of the reference is written for one example or one window at a time
- `MultiChannelStreamingDeviationDetector`: the reference defines the score on the window of the values learned before the one scored, or on statistics that move at each new value: the algorithm of the reference is written for one example or one window at a time
- `HampelScorer`: its batch form is HampelFilter, which scores every value on the centred window
- `HalfSpaceTrees`: the reference defines the score on the window of the values learned before the one scored, or on statistics that move at each new value: the algorithm of the reference is written for one example or one window at a time
- `LODA`: the reference defines the score on the window of the values learned before the one scored, or on statistics that move at each new value: the algorithm of the reference is written for one example or one window at a time
- `OneClassSGD`: the reference defines the score on the window of the values learned before the one scored, or on statistics that move at each new value: the algorithm of the reference is written for one example or one window at a time
- `OnlineIsolationForest`: the reference defines the score on the window of the values learned before the one scored, or on statistics that move at each new value: the algorithm of the reference is written for one example or one window at a time
- `OnlineRobustMahalanobis`: the whole-sample estimator is Section A.2 of Guillot et al. (2025), which lies outside the folder; the library initialises the dispersion with the average of the centred outer products as a declared choice, and the rest is stochastic gradient on a stream
