# How it works

`LogLoss` accumulates $-\ln p$ when the true label is 1 and $-\ln(1-p)$ when it is 0, then returns the mean. `BrierScore` accumulates $(1-p)^2$ when the true label is 1 and $p^2$ when it is 0, then returns the mean. Both metrics are proper scoring rules: they are minimized when the predicted probabilities equal the true class probabilities.

## The parameters

| Parameter | Meaning | Default |
|---|---|---|
| `eps` | small constant to avoid log(0) in LogLoss | 1e-15 |
| `normalize` | divide by count to get mean | True |

## API reference

::: dense_armor.core.evaluation.metrics.classification.LogLoss  
::: dense_armor.core.evaluation.metrics.classification.BrierScore  

## Details

- **Source:** proper scoring rules theory — both metrics are minimized when predicted probabilities match the true class probabilities.
