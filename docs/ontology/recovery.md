# Recovering a signal

A sensor network gives numbers that are noisy, partly missing and sometimes wrong, and the links between the sensors change as they move. `recover_graph_signal` estimates the true signal from such readings, using the fact that sensors linked in the graph read similar values and that a signal changes slowly in time. It follows Yamagata, Naganuma and Ono (2022, arXiv:2202.06432).

## 1. What goes in, what comes out

The input is a table `Y` with one row per sensor and one column per time-slot, and one Laplacian for each time-slot (a single one if the graph does not change). The output is a table of the same shape with the estimated signal.

```python
import numpy as np
from dense_armor.core.graph import Graph, recover_graph_signal

n, p = 15, 20
g = Graph(n, [(i, i + 1, 1.0) for i in range(n - 1)])
i, k = np.arange(n)[:, None], np.arange(p)[None, :]
x0 = np.sin(0.3 * i) * np.cos(0.1 * k)
y = x0 + 0.1 * np.random.default_rng(1).standard_normal((n, p))
xh = recover_graph_signal(y, g, lam=0.05, eps=0.9 * 0.1 * np.sqrt(n * p), temporal="l2")
print(np.sqrt(np.mean((y - x0) ** 2)) > np.sqrt(np.mean((xh - x0) ** 2)))
```

The true signal `x0` varies smoothly along the path and in time. `y` is `x0` plus noise of size 0.1. `eps` is the radius of the ball around `y` that the estimate must stay in: the paper takes it as 0.9 times the noise size times the square root of the number of readings. The last line compares the error of the noisy table with the error of the estimate, and the estimate is closer to the truth.

## 2. Missing values and wrong readings

A `mask` with 1 where a reading exists and 0 where it is missing lets the method fill the gaps from the neighbours in the graph and in time. The radius `eta` bounds the total size of the outliers, which `return_outliers=True` gives back as a second table.

```python
import numpy as np
from dense_armor.core.graph import Graph, recover_graph_signal

n, p = 10, 12
g = Graph(n, [(i, i + 1, 1.0) for i in range(n - 1)])
i, k = np.arange(n)[:, None], np.arange(p)[None, :]
x0 = np.sin(0.3 * i) * np.cos(0.1 * k)
y = x0.copy()
y[3, 4] += 5.0
mask = np.ones((n, p))
mask[7, 2] = 0.0
x, s = recover_graph_signal(y, g, mask=mask, lam=0.05, eps=0.1, eta=4.0, return_outliers=True)
print(x.shape, float(abs(s).sum()) <= 4.0 * 1.001)
```

The reading at `(3, 4)` is wrong by 5 and the one at `(7, 2)` is missing. The outlier table `s` is the part of the readings that the method explains as errors: its total size never exceeds `eta`. The estimate `x` has the shape of `y`.

## 3. The choices

| Argument | Meaning |
| --- | --- |
| `temporal="l1"` or `"l2"` | how a change in time is penalised: the sum of absolute changes (allows jumps) or the sum of squared changes (smooth) |
| `smoothness="signal"` or `"difference"` | whether the Laplacian measures the signal at each time-slot, or the change between two time-slots |
| `lam` | weight of the temporal term |
| `tol`, `max_iter` | stop when the estimate and the outliers change by less than `tol` in relative norm, or after `max_iter` steps |

A list of `p` Laplacians, or `Graph` objects, follows a graph that moves; one matrix gives the static case. The steps of the algorithm are chosen from the stability condition of the primal-dual method, and `gamma1` and `gamma2` replace them if you know better.

## Details

The projection on the `l1` ball, the soft-threshold and the projection on the `l2` ball are the proximity operators of the paper (Eq. (23)-(28)) and are in `dense_armor.core.graph.recovery`. The tests check them on small cases worked out by hand, and check that the projection on the `l1` ball is the nearest point of the ball.
