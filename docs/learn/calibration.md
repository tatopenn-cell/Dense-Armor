# Calibration (online Platt scaling)

A classifier can rank its predictions well and still attach the wrong numbers to them.
A decision tree that has 80 % accuracy on some region will happily output 0.99 on every
sample of that region — the ranking is fine, the numbers are lies. On a robot, the
control policy reads those numbers: a 0.99 that means "80 %" makes the policy act far
too confidently.

**Calibration** fixes the numbers without touching the ranking. `OnlinePlattScaling`
wraps any [river](https://riverml.xyz) classifier and corrects its probabilities one
sample at a time, using the map `sigmoid(a · logit(p) + b)` learned by Online Newton
Step.

## The idea, in one line

The base classifier's probability `p` is transformed into
`σ(a · logit(p) + b)`, with `a` and `b` learned online. If the base model is
over-confident (probabilities pushed toward 0 and 1), `a < 1` shrinks them toward 0.5.
If it is under-confident, `a > 1` stretches them out. If it is systematically biased
toward one side, `b ≠ 0` shifts them. Both numbers adapt as new samples arrive.

## The formula, symbol by symbol

Let `p` be the base model's probability for a sample, and `y ∈ {0, 1}` its true label.

- `logit(p) = ln(p / (1 − p))` — the inverse of the sigmoid, from `(0, 1)` to `ℝ`.
- `σ(z) = 1 / (1 + e^{−z})` — the sigmoid, from `ℝ` back to `(0, 1)`.
- The corrected probability is `σ(a · logit(p) + b)`.
- Start: `(a, b) = (1, 0)` — the base model's probabilities pass through unchanged.

After each label `y`, the parameters update by **Online Newton Step**:

```
u = (logit(p), 1)
g = (σ(θ · u) − y) · u
A ← A + g gᵀ                 (start A = ρ I, ρ = 100)
θ ← Proj(θ − A⁻¹ g / γ)      (γ = 0.1, projection onto ‖θ‖ ≤ 100)
```

The clip on `p` to `[0.01, 0.99]` is a numerical guard: `logit(0)` and `logit(1)` are
infinite, and the paper's regret bound (Theorem 2.1 of Gupta–Ramdas 2023,
[arXiv:2305.00070](https://arxiv.org/abs/2305.00070)) assumes the base model never
outputs exactly 0 or 1.

## Hand case

Base model says `p = 0.8` on a sample whose true label is `y = 0`. Start
`(a, b) = (1, 0)`, so `θ = (1, 0)`.

```
u = (logit(0.8), 1) = (1.3863, 1)
σ(θ · u) = σ(1.3863) = 0.8
g = (0.8 − 0) · (1.3863, 1) = (1.109, 0.8)
```

Assume `A = 100 · I` (start of the run, `A` has not accumulated much yet). Then:

```
A⁻¹ g = (0.01109, 0.008)
θ − A⁻¹ g / γ = (1, 0) − (0.111, 0.08) = (0.889, −0.08)
```

New corrected probability on the same input:

```
σ(0.889 · 1.3863 + (−0.08)) = σ(1.152) = 0.760
```

The base model was wrong and confident; after one sample of feedback, the calibrated
model says 0.760. It **lowered its confidence** after being wrong — exactly what
calibration is supposed to do.

## The result

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

The calibrated tree's log-loss on Phishing drops from 0.4535 to 0.3502. The wrapper
passes river's own `check_estimator` and runs at about 32,000 samples per second on top
of the base model.

## The parameters

| Element | Paper (Algorithm 1) | Parameter |
|---|---|---|
| Start | `(a, b) = (1, 0)`: base probabilities pass through unchanged | — |
| Newton step | scaled by `1/gamma`, `gamma = 0.1` | `gamma` |
| Initial Hessian | `rho · I`, `rho = 100` | `rho` |
| Projection | `‖(a, b)‖₂ ≤ 100` | `radius` |
| Clipping | `f(x)` in `[0.01, 0.99]` | `clip` |

## API reference

::: dense_armor.learn.calibration

---

## Details

The comparison against the paper's tracking variant (TOPS) and a JSD/ensemble variant
on seven streams is in Dense-Evolution-Discovery,
[Online Platt Scaling for river](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/river_platt_scaling/).

This is the same Online Newton Step as the paper's Algorithm 1 with the same
hyperparameters; no tuning was done on Phishing or on any of the other streams. The
0.3502 is what the paper's algorithm gives on this dataset, not a number that was
fished for.
