# Exponentially weighted average: a regressor

[← Ensembles](index.md)

`ExponentiallyWeightedAverageRegressor` is the regression version of the same idea. Each member
has a cumulative loss, and its weight is `exp(-eta * L_i)`, so a member that has been wrong
loses weight and its vote counts less. The prediction is the weighted average of the members'
predictions. This is the Hedge rule of Freund and Schapire (1997), Section 2.1.

```python
from dense_armor.roles import Regressor
from dense_armor.core.online.ensemble import ExponentiallyWeightedAverageRegressor

class Const(Regressor):
    def __init__(self, v):
        self.v = float(v)
    def learn_one(self, x, y, t=None):
        return self
    def predict_one(self, x, t=None, return_std=False):
        return self.v

ewa = ExponentiallyWeightedAverageRegressor(
    [Const(0.0), Const(50.0), Const(90.0)], eta=0.5)
for _ in range(30):
    ewa.learn_one({"x": 0.0}, 100.0)
print(round(ewa.predict_one({"x": 0.0}), 1))
```

```
90.0
```

Three constant members predict 0, 50 and 90 and the truth is 100. The simple average would give
46.7, the weighted average gives 90.0: the best member's weight dominates. The `bound()` method
returns the bound of equation 9 of the paper, which limits how much worse the ensemble can be
than its best member. On the regression stream of the cost section of the [index](index.md#what-the-ensembles-cost) the ensemble's cumulative loss is
above the best member's and below the bound, which is what the bound promises: it caps the loss,
it does not promise the ensemble wins.

---

## Details

### Source

- Freund, Y., Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. Journal of Computer and System Sciences 55(1), 119-139. Section 2.1 for the Hedge rule, Theorem 2 and equation 9 for the bound of the exponentially weighted average.
- Kivinen, J., Warmuth, M. K. (1997). Exponentiated gradient versus gradient descent for linear predictors. Information and Computation 132(1), 1-63. The exponentiated gradient gives the same multiplicative update of the weights, `w_i = exp(-eta * L_i)`, from the gradient of the relative entropy.

The bound of the exponentially weighted average is equation 9 of Freund and Schapire,
`L_ensemble <= (eta / (1 - exp(-eta))) * min_i L_i + ln(K) / (1 - exp(-eta))`, with `K` the
number of members. The factor `eta / (1 - exp(-eta))` tends to one as `eta` goes to zero, so a
small learning rate loses almost nothing over the best member, and the price of a large learning
rate is the second term. The bound assumes the loss is in `[0, 1]`; the script clips the squared
error to that interval before summing it, and reports the clipping in the JSON.

### Choices

- **The initial weights of the weighted average.** The paper's bound is for uniform initial
  weights, which is what the class uses.

### Limits

- The bound of the weighted average needs a loss in `[0, 1]`. A regression stream with squared
  errors outside that interval has a bound that is not meaningful, and the class does not clip
  the user's loss for them.

- On the regression stream the weighted average does not beat its own best member. The rule of
  the stream is not hard enough for the weaker members to be useful, and blending them in costs.
  A stream where every member is wrong in a different way is where the weighted average pays.
