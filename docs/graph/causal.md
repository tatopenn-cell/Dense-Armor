# What causes what?

Two things that move together are not always one the cause of the other. This page builds the causal graph of three variables, one of which pushes both of the others, and estimates the effect of one on the other once that third variable is taken out.

The page needs `pip install "dense-armor[causal]"` (DoWhy and pandas).

## 1. A causal graph

`CausalGraph(nodes, edges)` takes the names of the variables and the edges `(cause, effect)`. It refuses an edge to an unknown name and a cycle with `ValueError`. `parents(name)` gives the direct causes of a variable.

```python
from dense_armor.core.graph import CausalGraph

cg = CausalGraph(["C", "Z", "Y"], [("C", "Z"), ("C", "Y"), ("Z", "Y")])
print(cg.parents("Z"), cg.parents("Y"), cg.is_valid())
```

```text
['C'] ['C', 'Z'] True
```

`C` is a cause of `Z` and of `Y`, and `Z` is a cause of `Y`. `C` is a confounder: it makes `Z` and `Y` move together even if `Z` did nothing to `Y`.

## 2. The effect, with and without the confounder

`estimate_effect(treatment, value, data, outcome)` gives the change of `outcome` when `treatment` is set to `value` from 0. It adjusts for the parents of the treatment with the linear regression of DoWhy (`backdoor.linear_regression`). The data are a pandas table with one column for each variable. Here the true effect of `Z` on `Y` is 2.

```python
import numpy as np
import pandas as pd
from dense_armor.core.graph import CausalGraph

rng = np.random.default_rng(0)
m = 3000
c = rng.standard_normal(m)
z = c + 0.5 * rng.standard_normal(m)
y = 2.0 * z + 3.0 * c + 0.5 * rng.standard_normal(m)
cg = CausalGraph(["C", "Z", "Y"], [("C", "Z"), ("C", "Y"), ("Z", "Y")])
print(round(cg.estimate_effect("Z", 1.0, pd.DataFrame({"C": c, "Z": z, "Y": y}), "Y"), 4))
print(round(float(np.polyfit(z, y, 1)[0]), 4))
```

```text
2.0077
4.3964
```

The adjusted effect is 2.0077, close to the true 2. A line fitted to `Y` against `Z` alone gives 4.3964: it counts the part of `Y` that comes from `C` as if it came from `Z`. A parent of the treatment that is not a column of the data raises `ValueError`.

## Details

The estimator is a choice of this module: the simplest one of DoWhy, on the parents of the treatment, so that the effect is identifiable and can be tested on a graph whose answer is known. DoWhy also has instrumental variables, front-door and propensity scores, and the cited paper (Zan and others 2026) is about a different problem, the search for the causal graph with a language model. The graph itself is not passed to DoWhy: only the parents of the treatment are. The estimate is the slope times `value`, so it is a linear effect.
