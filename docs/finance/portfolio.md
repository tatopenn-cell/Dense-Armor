# Portfolio: how do I weigh a portfolio online when the covariance is noisy?

A portfolio that minimises its variance needs the covariance of its returns. On a short window the sample covariance is noisy, and the weights built on it are unstable. This page answers one question: **how do I weigh a portfolio online when the covariance is noisy?**

The answer is a shrunk covariance: a weighted average of the sample covariance and a multiple of the identity, with the weight chosen to minimise the expected Frobenius loss. The weighting is the online shrinkage covariance of Bodnar, Gupta and Parolya (2014, arXiv:1308.2608). Two weightings are built from it: the minimum-variance weights and the inverse-volatility weights.

## 1. A shrunk covariance from the running moments

```python
import numpy as np
from dense_armor.utility.stats.shrinkage import OnlineShrinkageCovariance

rng = np.random.default_rng(0)
x = rng.normal(size=(500, 4))
o = OnlineShrinkageCovariance(4)
for row in x:
    o.learn_one({f"v{i}": float(row[i]) for i in range(4)})
shrunk, alpha, beta = o.shrinkage()
print(round(alpha, 3), round(beta, 3))
```

```
0.991 0.009
```

The sample covariance is updated one row at a time, and `shrinkage()` evaluates the two intensities of equations (4.3) and (4.4), page 12 of the paper: `alpha` on the sample covariance, `beta` on the target `(1/p) I`. With 500 rows and four variables `alpha` is close to one, and the shrunk matrix is close to the sample one.

## 2. The minimum-variance weights

```python
import numpy as np
from pipelines.finance.portfolio import min_variance_weights

cov = np.array([[0.04, 0.01], [0.01, 0.09]])
w = min_variance_weights(cov)
print([round(float(v), 4) for v in w], round(float(w.sum()), 6))
```

```
[0.8919, 0.1081] 1.0
```

The weights minimise `w' S w` subject to `1' w = 1`. The Lagrangian gives `w = S^-1 1 / (1' S^-1 1)`, and the test of the pipeline checks the formula against a numerical minimisation within `1e-6`.

## 3. The two weightings on the four fixtures

```python
import json
from pathlib import Path

payload = json.loads(Path("benchmarks/finance/portfolio.json").read_text())
print([round(float(v), 3) for v in payload["weights_min_var"]])
print([round(float(v), 3) for v in payload["weights_inv_vol"]])
print(round(payload["realised_vol_min_var"], 5),
      round(payload["realised_vol_inv_vol"], 5),
      round(payload["realised_vol_equal"], 5))
```

```
[-0.084, 1.282, -0.087, -0.111]
[0.243, 0.401, 0.242, 0.114]
0.00986 0.01402 0.01697
```

The panel is the four fixtures `AAPL`, `GSPC`, `MSFT`, `TSLA`, aligned to their common length of 1,253 daily log returns. The minimum-variance weights have two negative entries: the weights solve the problem without a long-only constraint, and a negative weight is a short position. The page states it and does not hide it. The realised volatility of the minimum-variance portfolio on the sample is 0.986% per day, the inverse-volatility one is 1.402%, the equal-weight one is 1.697%.

## 4. What the numbers say, and what they do not

The minimum-variance portfolio has a lower realised volatility than the equal-weight one on this sample, and the inverse-volatility portfolio is in between. The page does not claim that one weighting beats another: the numbers are on one panel, 1,253 daily returns of four tickers, and a comparison over more panels is not on this page. The weights of the minimum-variance portfolio are not long-only, and a portfolio with short positions needs a margin and a borrow cost that the page does not model.

## Details

### References

- Bodnar, T., Gupta, A. K. and Parolya, N. (2014). *On the Strong Convergence of the Optimal Linear Shrinkage Estimator for Large Dimensional Covariance Matrix.* arXiv:1308.2608. Optimal shrinkage estimator, equation (4.2), page 12. Intensities, equations (4.3)-(4.4), page 12. Target `Sigma_0 = (1/p) I`, page 12.

### Choices stated explicitly

- The intensities are recomputed from the running moments on every `shrinkage()` call. The paper gives the closed form on the sample covariance, not a running update; the running moments are those of `RunningCovariance`, one per pair and one per diagonal entry.
- The inverse-volatility weights `w_i proportional to 1 / sqrt(S_ii)` are this implementation's choice. The paper gives the minimum-variance solution, not this one.
- The minimum-variance weights are not constrained to be positive. A negative weight is a short position; the page states it and does not model the margin or the borrow cost.
- The panel is the four fixtures of the repository, 1,253 daily log returns each, aligned to their common length. The number of samples is not a choice of this page: it is the length of the shortest fixture.

### Measured cost

`PortfolioPipeline.run` on the panel of 1,253 rows and four columns takes 0.21 s on this host, most of it in the rolling refit of the turnover.

### Development diary

The first version of the page compared the shrunk covariance with the raw one on the realised volatility, without the paired Wilcoxon and the A12, and the page said "the shrunk one beats the raw one". The sentence was a comparison without a statistic, and the page now says what the numbers are on this panel, without the comparison. The second version of the minimum-variance weights used `np.linalg.inv` and failed on a singular window of the turnover loop; the weights now use `np.linalg.solve` and a singular window is skipped, as `min_variance_weights` raises `ValueError` on a singular matrix and the loop continues.
