# Optim: the algorithms

An optimiser moves the weights downhill on a loss. These classes take the gradient of the loss over the whole dataset and apply the update rule of the paper, step after step, so they can minimise any loss whose gradient you can write.

Every class reads the whole dataset with `fit` and leaves its results in attributes with a trailing underscore; `transform` or a query method applies them. The data are `synthetic` where a block makes them up.

## 1. `GradientDescent`

`w <- w - lr g` on the whole gradient.

~~~python
import numpy as np
from dense_armor.ml.optim import GradientDescent
print(GradientDescent(lr=0.5).minimize(lambda w: 2 * w, [1.0], 3).round(4).tolist())
~~~

~~~
[0.0]
~~~

## 2. `AdaGrad`

`w <- w - lr g / (sqrt(sum g^2) + eps)` per coordinate.

~~~python
import numpy as np
from dense_armor.ml.optim import AdaGrad
print(AdaGrad(lr=1.0).minimize(lambda w: np.array([1.0]), [0.0], 1).round(4).tolist())
~~~

~~~
[-1.0]
~~~

## 3. `Adam`

Adam with bias-corrected moments (Algorithm 1 of Kingma and Ba).

~~~python
import numpy as np
from dense_armor.ml.optim import Adam
print(Adam(lr=0.1).minimize(lambda w: np.array([1.0]), [0.0], 1).round(4).tolist())
~~~

~~~
[-0.1]
~~~

## 4. `FTRLProximal`

Follow-the-Regularized-Leader with linearised losses and an L1 and L2 term.

~~~python
import numpy as np
from dense_armor.ml.optim import FTRLProximal
print(FTRLProximal(alpha=1.0, beta=0.5).minimize(lambda w: np.array([1.0]), [0.0], 1).round(4).tolist())
~~~

~~~
[-1.0]
~~~

## 5. `NewtonStep`

`w <- w - (1 / gamma) A^-1 g` with `A = eps I + sum g g^T`.

~~~python
import numpy as np
from dense_armor.ml.optim import NewtonStep
print(NewtonStep(gamma=1.0, eps=1.0).minimize(lambda w: np.array([1.0]), [0.0], 1).round(4).tolist())
~~~

~~~
[-0.5]
~~~

## Details

### References

Duchi, J., Hazan, E., Singer, Y. (2011). Adaptive subgradient methods for
        online learning and stochastic optimization, Eq. (4) (not in the
        folder, bibliographic only).
    Kingma, D. P., Ba, J. (2015). Adam: a method for stochastic optimization,
        Algorithm 1, arXiv:1412.6980 (not in the folder, bibliographic only).
    Orabona, F. (2019). A modern introduction to online learning, Chapter 7,
        Algorithm 7.2 (FTRL with linearised losses) (not in the folder,
        bibliographic only).
    Hazan, E., Agarwal, A., Kale, S. (2007). Logarithmic regret algorithms for
        online convex optimization, Algorithm 2 (Online Newton Step) (not in
        the folder, bibliographic only).

### Choices stated explicitly

The class of a sketch is the exact value the sketch approximates, and the class of a statistic is the textbook formula on the whole array: both are `our choice`, because the references name no other batch algorithm.

### Development diary

The classes are new and independent of the online classes of dense_armor.utility.
