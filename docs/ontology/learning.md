# Learning a graph

Often you have the readings of many sensors and no map of how they are related. `learn_graph` builds the graph from the readings: sensors whose readings are close get a strong link. If you already know part of the structure, you can give it and the method keeps the weights close to it where you know, and learns the rest. It follows Niresi, Kuhn, Frusque and Fink (2024, arXiv:2406.03898).

## 1. A graph from readings

The input is a table with one row per sensor and one column per observation. The output is a `Graph`.

```python
import numpy as np
from dense_armor.core.graph import learn_graph

rng = np.random.default_rng(3)
base = rng.standard_normal(12)
x = np.stack([base, base + 0.01 * rng.standard_normal(12), 3 * rng.standard_normal(12)])
g, w = learn_graph(x, return_weights=True)
print(g.n_nodes, bool(w[0, 1] > w[0, 2]))
```

```text
3 True
```

The first two sensors read almost the same values and the third reads something different. The weight between the first two is larger than the one between the first and the third. `w` is the symmetric matrix of weights and `g` has an edge wherever a weight is above `threshold`.

Three terms are balanced in the problem (Eq. (9)): the smoothness of the readings on the graph, a barrier on the degrees that gives every sensor at least one link (weight `alpha`), and a small penalty on the size of the weights (weight `beta`).

## 2. Adding what you know

`known` is a matrix of prior weights, with a non-zero entry wherever you know there is a link. `upsilon` says how much to trust it.

```python
import numpy as np
from dense_armor.core.graph import learn_graph

x = np.random.default_rng(2).standard_normal((6, 10))
known = np.zeros((6, 6))
for a in range(5):
    known[a, a + 1] = known[a + 1, a] = 1.0
_, free = learn_graph(x, return_weights=True)
_, informed = learn_graph(x, known=known, upsilon=5.0, return_weights=True)
on = known > 0
print(bool(abs(informed[on] - 1.0).mean() < abs(free[on] - 1.0).mean()))
```

```text
True
```

The chain `known` links neighbouring sensors with weight 1. With `upsilon` large, the learned weights on those links are closer to 1 than without the prior.

## 3. Weights from the physics of the links

When you know how a quantity drops along each link, a small drop means a strong link. `weights_from_drops` turns the drops into weights between 0 and 1 and removes the ones below `cut`, as in Section III.A of the paper; the result can be given as `known`.

```python
from dense_armor.core.graph import weights_from_drops

print(weights_from_drops([0.0, 1.0, 2.0, 20.0]).tolist())
```

```text
[0.0, 1.0, 0.5, 0.0]
```

A drop of 0 stands for no link and gets weight 0. The drop 1 is the smallest, so its weight is 1; the drop 2 gives 0.5; the drop 20 gives 0.05, which is below the cut of 0.1 and is removed.

## Details

The problem is solved with a primal-dual splitting, in the form of Eq. (10) of the paper of the previous page. The paper names the Monotone+Lipschitz Forward-Backward-Forward algorithm instead; both solve the same problem (11). The tests compare the value of the objective with the value found by a general solver (`scipy.optimize`, L-BFGS-B) on a small graph. The weights are non-negative and symmetric, and have a zero diagonal.
