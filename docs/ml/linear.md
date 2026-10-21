# Linear: the algorithms

A linear model predicts a number, or the probability of a class, as a weighted sum of the features. Here the weights are found on the whole table: by one linear solve for the regression, by Newton steps for the logistic model, and as a posterior distribution for the Bayesian regression.

Every class reads the whole dataset with `fit` and leaves its results in attributes with a trailing underscore; `transform` or a query method applies them. The data are `synthetic` where a block makes them up.

## 1. `LinearRegression`

Ridge regression in closed form.

~~~python
import numpy as np
from dense_armor.ml.linear import LinearRegression
m = LinearRegression().fit([[0.0], [1.0], [2.0]], [1.0, 3.0, 5.0])
print(m.coef_.round(6).tolist(), round(m.intercept_, 6))
~~~

~~~
[2.0] 1.0
~~~

## 2. `LogisticRegression`

Binary logistic regression by Newton steps on the mean log loss.

~~~python
import numpy as np
from dense_armor.ml.linear import LogisticRegression
m = LogisticRegression(l2=0.01).fit([[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1])
print(m.predict([[0.0], [3.0]]).tolist())
~~~

~~~
[0, 1]
~~~

## 3. `BayesianLinearRegression`

Bayesian linear regression with a Gaussian prior, in closed form.

~~~python
import numpy as np
from dense_armor.ml.linear import BayesianLinearRegression
m = BayesianLinearRegression(alpha=1e-6, beta=100.0).fit([[0.0], [1.0], [2.0]], [1.0, 3.0, 5.0])
print(round(float(m.predict([[3.0]])[0]), 3))
~~~

~~~
7.0
~~~

## Details

### References

Bishop, C. M. (2006). Pattern Recognition and Machine Learning, Eq. (3.50)
        to (3.53), pages 153-154: the posterior of the weights of a Bayesian
        linear regression with a Gaussian prior (not in the folder,
        bibliographic only).
    Ridge regression and logistic regression by Newton steps (iteratively
        reweighted least squares) are textbook definitions.

### Choices stated explicitly

The class of a sketch is the exact value the sketch approximates, and the class of a statistic is the textbook formula on the whole array: both are `our choice`, because the references name no other batch algorithm.

### Development diary

The classes are new and independent of the online classes of dense_armor.utility.

Classes without a form:
- `PayloadEstimator`: the reference estimates the payload with the dynamics model of the robot; the interface on a table of regressors and residuals is not in it
