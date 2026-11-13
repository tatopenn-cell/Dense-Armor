# Following a graph online

`learn_graph` looks at a table of readings all at once. When readings arrive one at a time and the links may change, `AdaCGP` updates its estimate of the weights at every sample. It follows Jenkins, Variddhisai, El-Medany, Ng and Mandic (2024, arXiv:2411.01567).

## 1. The model

The signal at time `t` is a mix of the signals of the last `P` time-slots, spread along the graph: `x_t = H_1 x_(t-1) + ... + H_P x_(t-P) + noise`. Each `H_p` is a polynomial of the weight matrix `W` (the "graph shift operator"). `AdaCGP` estimates the matrices `H_p` with a least squares that forgets old samples at rate `forgetting`, and `weights` is the estimate of `W`.

```python
import numpy as np
from dense_armor.core.graph import AdaCGP

w = np.zeros((4, 4))
for i in range(4):
    w[i, (i + 1) % 4] = w[(i + 1) % 4, i] = 1.0
h = 0.2 * np.eye(4) + 0.3 * w
rng = np.random.default_rng(0)
model = AdaCGP(4, order=1, forgetting=0.995, mu=0.0, alpha=0.001)
x = np.zeros(4)
for _ in range(3000):
    x = h @ x + rng.standard_normal(4)
    model.update(x)
print(round(float(np.linalg.norm(model.psi - h) / np.linalg.norm(h)), 2) < 0.3)
```

```text
True
```

The signal lives on a ring of four nodes and follows `H = 0.2 I + 0.3 W`. After 3000 samples the estimate `model.psi` of `H` is within 30% of the true one, measured in relative norm.

## 2. Sparse weights, and the two paths

With `mu` above 0 the estimate is pushed towards zeros, so that a missing link is exactly 0 and not just small. `mu` can be a number or one number for each of the `P` filters, and each step uses `mu_p` times the largest entry of the correlation of the block, as in the paper.

For an order above 1, the filters must commute with each other, because they are polynomials of the same `W`. `gamma` weighs that requirement. `path=2` takes `W` from the first filter and keeps the requirement inside the filter step. `path=1` leaves it out of the filter step and fits `W` in a second step (`beta` is its step size).

## 3. After the sparsity: refit and the filter coefficients

`debias(x)` refits only the entries that are not zero, to remove the shrinkage that `mu` causes. `update_filter(x)` then estimates the coefficients `h` of the polynomials with an instantaneous step; with order `P` there are `P (P + 3) / 2` of them, in `filter_coefficients`. Call `update_filter(x)` before `update(x)` or `debias(x)` with the same sample.

## Details

The step sizes `alpha` and `beta` are given; the paper also allows a line search. The update of `h` in Eq. (19) is written with `+ eta b`. The objective (10) has the penalty `eta ||h||_1`, whose gradient step is `- eta b`, so the code uses `h <- h + rho (Y^T e - eta b)`: the sign of the penalty in the equation is the one that would grow the coefficients and not shrink them. The tests check the first update from zero against the value worked out by hand, that the zeros of the filters stay zero after `debias`, and that the filter and the coefficients of a known process are recovered.
