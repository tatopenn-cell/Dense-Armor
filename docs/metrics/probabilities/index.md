# Probabilities: log loss and Brier score

A probabilistic model does not output just 0 or 1 — it outputs a confidence score between 0 and 1. Log loss and Brier score measure how well those predicted probabilities match what actually happened.

For a trading model that quotes the default probability of a bond, these two metrics tell you whether the probabilities are honest before you lose money.

## The map of probabilities

Click a page to open it, or the button at the top to go back to the [map of the library](../index.md#the-library-folder-by-folder).

<object type="image/svg+xml" data="../assets/maps/probabilities.svg" style="width:100%"></object>

## Pages

1. [Log loss](log_loss.md): $-\ln p$ scoring for hard predictions with confidence
2. [Brier score](brier_score.md): mean squared distance between probability and outcome
3. [How it works](how_it_works.md): proper scoring rules, parameters, API reference
