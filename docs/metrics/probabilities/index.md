# Probabilities: log loss and Brier score

A probabilistic model does not output just 0 or 1 — it outputs a confidence score between 0 and 1. Log loss and Brier score measure how well those predicted probabilities match what actually happened.

For a trading model that quotes the default probability of a bond, these two metrics tell you whether the probabilities are honest before you lose money.

## The map of probabilities

Click a page to open it, or the button at the top to go back to the [map of the library](../index.md#the-library-folder-by-folder).

<object type="image/svg+xml" data="../assets/maps/probabilities.svg" style="width:100%"></object>

## What this is

`LogLoss` accumulates $-\ln p$ when the true label is 1 and $-\ln(1-p)$ when it is 0, then returns the mean.

`BrierScore` accumulates $(1-p)^2$ when the true label is 1 and $p^2$ when it is 0, then returns the mean.

Both are proper scoring rules: they are minimized when the predicted probabilities equal the true class probabilities.
