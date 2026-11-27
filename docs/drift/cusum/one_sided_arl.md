# one_sided_arl: how long will a one-direction CUSUM wait?

[← CUSUM](index.md)

`one_sided_arl(delta, h, corrected=True)` gives the average run length (ARL): the expected number of samples before the alarm of a CUSUM that watches one direction only. It lets you know the answer before you run anything.

## First example

```python
from dense_armor.core.online.drift.cusum import one_sided_arl

for d in (-0.5, 0.0, 0.5, 1.0):
    print(d, round(float(one_sided_arl(d, 5.0)), 1))
print(round(float(one_sided_arl(0.5, 5.0, corrected=False)), 1), round(float(one_sided_arl(0.0, 5.0, corrected=False)), 1))
```

```
-0.5 938.2
0.0 38.0
0.5 10.3
1.0 5.7
8.0 25.0
```

`delta` is the drift in units of the noise: the true mean minus the slack `k`. With the threshold `h` = 5: when `delta` is negative the total keeps falling to zero and the alarm is rare (938.2 samples); with `delta` = 0 it is 38.0; with `delta` = 0.5 it is 10.3; with `delta` = 1 it is 5.7. The last line is the same two cases with `corrected=False`, the plain textbook formula: 8.0 instead of 10.3, and 25.0, which is $h^2$, instead of 38.0.

## How it works

The formula, for one side, with drift $\delta = \mu - k$ in noise units, is

$$\mathrm{ARL}(\delta) = \frac{e^{-2\delta h'} - 1 + 2\delta h'}{2\delta^2}, \qquad h' = h + 1.166,$$

and for $\delta = 0$ it becomes $h'^2$. This is equation (6) of Reynolds (1975), obtained by treating the running total as a Brownian motion; the $+1.166$ is Siegmund's correction for the last jump that overshoots the threshold. With `corrected=True` the function uses $h' = h + 1.166$; with `corrected=False`, $h' = h$.

## The parameters

| Parameter | What it is |
|---|---|
| `delta` | standardized drift, the true mean minus `k`; 0 is the case with no change for that direction |
| `h` | decision boundary, in robust-sigma units, the `h` of the detector |
| `corrected` | `True` (default) applies the correction of Siegmund; leave it `True` unless you want to reproduce the plain textbook formula |

The function returns a float: the approximate run length, in samples. Both directions together are on the page [two_sided_arl](two_sided_arl.md).

## Details

- **Sources.** Page, E. S. (1954), "Continuous inspection schemes", *Biometrika* 41(1-2), introduced the CUSUM. The approximation follows Reynolds, M. R. Jr. (1975), "Approximations to the Average Run Length in Cumulative Sum Control Charts", *Technometrics* 17(1), with the boundary correction refined by Siegmund, D. (1985), *Sequential Analysis*, Theorem 10.16, to about 1.166 (Reynolds' own empirical value was about 1.2).
- **How good it is.** Against a Monte Carlo simulation of the exact idealized process, the corrected formula matches to about 1% and the uncorrected one underestimates by about 20% at realistic `h`, as Reynolds found in 1975 (check made in Dense-Evolution-Discovery, `scripts/cusum_detectability_theory/validate_arl_theory.py`).
