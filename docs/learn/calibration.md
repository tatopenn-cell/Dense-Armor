# Calibration (online Platt scaling)

A classifier can rank its predictions well and still attach the wrong numbers to them: a
decision tree says 0.99 when it is right 80% of the time. `OnlinePlattScaling` wraps any
[river](https://riverml.xyz) classifier and corrects its probabilities one sample at a time,
with the map `sigmoid(a · logit(p) + b)` learned by Online Newton Step, exactly as in
Algorithm 1 of Gupta and Ramdas, *Online Platt Scaling with Calibeating* (ICML 2023,
[arXiv:2305.00070](https://arxiv.org/abs/2305.00070)).

```bash
pip install dense-armor[river]
```

```python
from river import datasets, evaluate, metrics, tree
from dense_armor.learn.calibration import OnlinePlattScaling

dataset = datasets.Phishing()
evaluate.progressive_val_score(dataset, tree.HoeffdingTreeClassifier(), metrics.LogLoss())
evaluate.progressive_val_score(dataset, OnlinePlattScaling(tree.HoeffdingTreeClassifier()), metrics.LogLoss())
```

```
LogLoss: 0.4535476064322544
LogLoss: 0.35021471255578124
```

The calibrated tree's log-loss on Phishing drops from 0.4535 to 0.3502. The wrapper passes
river's own `check_estimator` and runs at about 32,000 samples per second on top of the base
model.

| Element | Paper (Algorithm 1) | Parameter |
|---|---|---|
| Start | `(a, b) = (1, 0)`: base probabilities pass through unchanged | — |
| Newton step | scaled by `1/gamma`, `gamma = 0.1` | `gamma` |
| Initial Hessian | `rho · I`, `rho = 100` | `rho` |
| Projection | `‖(a, b)‖₂ ≤ 100` | `radius` |
| Clipping | `f(x)` in `[0.01, 0.99]` | `clip` |

::: dense_armor.learn.calibration

---

**Details**: the comparison against the paper's tracking variant (TOPS) and a JSD/ensemble
variant on seven streams is in Dense-Evolution-Discovery,
[Online Platt Scaling for river](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/river_platt_scaling/).
