# RecursiveLeastSquares: a line that keeps up with the data

`RecursiveLeastSquares` fits a linear model `y ≈ w · x` one sample at a time, without keeping the samples: it keeps the weights and a small matrix (the covariance) and updates both at every sample. With a forgetting factor `lam` below 1 it weighs the recent samples more, so it follows a model that changes. It is the engine of [ResidualDynamicsLearner](residual_dynamics.md), which uses one of them per joint of a robot, and of [DriftAwareResidualDynamicsLearner](drift_aware.md). The overview is on [Online dynamics](online_dynamics.md).

## Gamestory

London Theater, deep night. The curtain is about to rise and the young puppeteer already has his hands tangled in the wooden strings.

Until the night before, the puppeteer had a big problem: during the show his hands were busy working and he certainly couldn't stop to do the math on how hard to pull the strings. So he had to wait until the end of the show, when the audience had already left, to sit at the table, grab a giant ledger, and calculate all the mistakes of the evening. By then it was too late!

Tonight, however, backstage there is an assistant with a pen and a tiny sheet of paper, ready to apply the RecursiveLeastSquares trick:

- As soon as the show begins, the assistant's sheet is completely blank. He doesn't know yet how stiff the strings are or how heavy the puppet is. But thanks to a rule called delta, as soon as the puppeteer gives the very first tug, the assistant immediately trusts what he sees and writes down the first two little numbers with his pen.
- From that moment on, while the puppeteer works nonstop on stage, the assistant writes one piece at a time, live. Every time the puppet's arm goes up too high or too low, the assistant crosses out the old numbers on the little sheet and writes the updated ones. No giant ledgers, no waiting until the end of the evening!
- And what if halfway through the show the strings loosen up or the wood gets heavier? The puppeteer gives a nod to the assistant, who uses the lambda rule: he lets the calculations from the start of the show slowly fade away and keeps only the ones written in the last few minutes.

In the end, what is RecursiveLeastSquares? It is the exact trick of the assistant with the little sheet: a super-fast method that doesn't wait until the end of the day to study all the data together, but updates the math one piece at a time live, erasing and rewriting on the very same small space!

![Left: the fee that a recursive least squares learner predicts for a payment of 100 over 600 payments, with and without forgetting: both find the real fee of 2.50, which is 0.50 above what the manual says; at payment 300 the real fee jumps to 4.00, lam 0.99 follows within a few hundred payments and lam 1 is still at 3.30 at payment 600. Right: the absolute error, on a log scale.](../assets/learn/rls_tracking.png)

The picture follows the fee of a payment of 100. The dotted line is what the manual says (2.00). Both learners find the real fee, 2.50, within the first few dozen payments: they discover the fixed charge. At payment 300 the commission changes and the real fee jumps to 4.00. The learner with `lam = 0.99` is back within 0.1 of it 256 payments later and ends 0.07 away; the one with `lam = 1` weighs the first 300 payments as much as the last, and ends 0.70 away.

## A first example: a fee formula that changes

The manual says that the fee is 2 % of the amount. In reality there is a fixed 0.50 on top, and at payment 300 the commission becomes 3.5 %. The model has two features: the amount and a constant `1` (that is how a fixed charge is learned).

```python
import numpy as np
from dense_armor.core.online.learn.online_dynamics import RecursiveLeastSquares

rng = np.random.default_rng(0)
models = {
    "lam=1": RecursiveLeastSquares(lam=1.0),
    "lam=0.99": RecursiveLeastSquares(lam=0.99),
}
for i in range(600):
    amount = rng.uniform(10, 500)
    fee = (0.02 if i < 300 else 0.035) * amount + 0.5 + rng.normal(0, 0.05)
    x = {"amount": amount, "one": 1.0}
    for m in models.values():
        m.learn_one(x, fee)
    if i + 1 in (300, 600):
        for name, m in models.items():
            print(
                i + 1,
                name,
                round(m.predict_one({"amount": 100.0, "one": 1.0}), 2),
            )
```

```
300 lam=1 2.5
300 lam=0.99 2.49
600 lam=1 3.3
600 lam=0.99 3.93
```

At payment 300 they predict 2.50 and 2.49 for a payment of 100, close to the real fee (the manual would say 2.00). At payment 600 the learner with `lam=0.99` predicts 3.93 and the one with `lam=1` 3.30, against a real fee of 4.00.

## How it learns

### The formula, symbol by symbol

Let `w` be the weight vector (5 numbers per joint), `P` the inverse-covariance matrix
(5×5), `x` the feature vector, `y` the residual torque.

```
K = P x / (λ + xᵀ P x)          (Kalman gain, 5-vector)
w ← w + K (y − xᵀ w)             (weight update)
P ← (P − K xᵀ P) / λ             (covariance update)
```

`P₀ = δ · I` and `w₀ = 0`. `λ` is the **forgetting factor**: `λ = 1` weights all past
samples equally (batch least squares), smaller values weight recent samples more (the
model tracks a parameter that changes).

`P` is symmetrised after each update (`P ← 0.5 (P + Pᵀ)`) to keep numerical errors from
breaking the symmetry that the derivation assumes.

### Hand case

Start `δ = 1e6` (so `P₀ = 1e6 · I`, essentially uninformative), `λ = 1`, one sample
`x = (1, 2)`, `y = 5`:

```
P₀ x = 1e6 · (1, 2) = (1e6, 2e6)
xᵀ P₀ x = 1·1e6 + 2·2e6 = 5e6
K = (1e6, 2e6) / (1 + 5e6) ≈ (0.2, 0.4)
w₀ = (0, 0), so xᵀ w₀ = 0, and the error is y − 0 = 5
w ← (0, 0) + 5 · (0.2, 0.4) = (1.0, 2.0)
```

```python
from dense_armor.core.online.learn.online_dynamics import RecursiveLeastSquares

rls = RecursiveLeastSquares(lam=1.0, delta=1e6)
rls.learn_one({"a": 1.0, "b": 2.0}, 5.0)
print(
    round(rls.predict_one({"a": 1.0, "b": 2.0}), 4),
    round(rls.predict_one({"a": 1.0, "b": 0.0}), 4),
)
```

```
5.0 1.0
```

After one sample the weights are exactly `(1, 2)`: `w · x = 1·1 + 2·2 = 5`. That is the
point of the large `δ`: with no prior information, the first sample determines the fit
completely.

With `λ < 1` the same update happens but the effective count of past samples is capped
at `1 / (1 − λ)`. `λ = 0.99` means roughly 100 samples of memory; `λ = 0.9` means
roughly 10.

## The parameters

| Parameter | Meaning | Default |
|---|---|---|
| `lam` | forgetting factor in `(0, 1]`: `1` weighs all the past samples equally and matches the batch least squares; a smaller value weighs the recent samples more | 1.0 |
| `delta` | the initial covariance scale, `P₀ = delta · I`: a larger value means less confidence in the starting weights, which are zero | 1e6 |
| `feature_keys` | the features and their order; `None` sorts the keys of the first dict, so the result does not depend on the order of the dict | `None` |

The choice of `lam` is a trade-off: too small and the learner is noisy (it fits the last few samples), too large and it is slow to follow a change. 0.99 forgets over about a hundred samples.

## API reference

::: dense_armor.core.online.learn.online_dynamics.RecursiveLeastSquares
    options:
      docstring_style: numpy

---

## Details

- `learn_one(x, y)` takes a dict of features and the target; `predict_one(x)` returns the prediction.
- RLS with exponential forgetting: Ljung, L., Soderstrom, T. (1983). *Theory and Practice of Recursive Identification*. MIT Press.
