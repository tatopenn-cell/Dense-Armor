# Metrics and evaluation

These tools keep a running score that is updated every time a new reading arrives and its answer can be compared with what really happened.

## The map of metrics

Click a page to open it, or the button at the top to go back to the [map of the library](../index.md#the-library-folder-by-folder).

<object type="image/svg+xml" data="../assets/maps/metrics.svg" style="width:100%"></object>

## What this is

A metric reads one pair `(y_true, y_pred)` with `update` and answers the current score with `get()`. Each pair represents one observation: the true outcome and the model's prediction.

The power is in the update: every time a new pair arrives, the metric adjusts its score. No need to keep a list of all past pairs. The history is compressed into just the current state.

## The seven pages

1. [Metrics: pairs](basics.md): the foundation — one pair at a time, the Metric base class, the update/get pattern
2. [Classification](classification/index.md): accuracy, precision, recall, F1, confusion matrix, balanced accuracy
3. [Probabilities](probabilities/index.md): log-loss, Brier score for probabilistic predictions
4. [Regression](regression/index.md): MAE, MSE, RMSE, R², interval metrics for uncertainty
5. [Uncertainty](uncertainty/index.md): interval coverage, mean interval width, Gaussian NLL
6. [Stream](stream/index.md): rolling windows, ROC AUC for streaming evaluation
7. [Events](events/index.md): detection delay, false alarms, NAB score for anomaly detection

In the end, what are the metrics (Metric, Accuracy, F1, LogLoss, MeanAbsoluteError, Rolling, EventMetrics)? They are super-light tools that calculate and update in real time the score of a predictive model one prediction at a time, in fixed memory, without ever having to keep the whole pile of past predictions.