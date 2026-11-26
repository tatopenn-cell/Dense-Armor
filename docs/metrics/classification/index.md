# Classification: scores for a model that answers with a class

[← Metrics](../index.md)

## The map of classification

Click a page to open it, or a job on the left (or the button at the top) to go back to the [map of the library](../../index.md#the-library-folder-by-folder).

<object type="image/svg+xml" data="../../assets/maps/classification.svg" style="width:100%"></object>

A classifier answers with a class: crash or no crash, fault or no fault. After each answer the truth arrives, and a classification metric updates its score with that one pair and forgets it. The eight metrics of this section differ in what they ask: how often the answer is right, how many of the rare cases it found, how every kind of mistake is counted, and how much better it is than a guess.

## A first example: an alarm for the crash days of a stock index

Ten trading days: 1 is a crash day and 0 is a normal day. The alarm rings (1) or stays quiet (0). Four metrics read the same ten pairs.

```python
from dense_armor.core.evaluation.metrics import Accuracy, BalancedAccuracy, CohenKappa, F1

crash = [0, 0, 1, 0, 0, 1, 0, 0, 0, 1]
alarm = [0, 1, 1, 0, 0, 1, 0, 1, 0, 0]
ms = [Accuracy(), F1(positive=1), BalancedAccuracy(), CohenKappa()]
for truth, flag in zip(crash, alarm):
    for m in ms:
        m.update(truth, flag)
print([round(m.get(), 3) for m in ms])
```

```
[0.7, 0.571, 0.69, 0.348]
```

The alarm is right on 7 days out of 10, so the accuracy is 0.7. It caught 2 of the 3 crashes and 2 of its 4 rings were false, so the F1 is 0.571. The balanced accuracy averages the recall of the normal days (5/7) and of the crash days (2/3): 0.69. The kappa subtracts the agreement two guessers with the same habits would reach, 0.54, from the observed 0.7: (0.7 − 0.54) / (1 − 0.54) = 0.348. Each metric answers its own question, and each page below explains one of them with the same alarm.

## The classification metrics, one page each

| page | classes | what it is for |
|---|---|---|
| [Precision, recall and F1](precision_recall_f1.md) | `Accuracy`, `Precision`, `Recall`, `F1`, `FBeta` | how often an alarm is right when it rings and how many real cases it finds |
| [Confusion matrix and balanced accuracy](confusion_matrix.md) | `ConfusionMatrix`, `BalancedAccuracy` | every kind of answer counted, and a score that does not favour the common class |
| [Cohen's kappa](kappa.md) | `CohenKappa` | agreement with the truth beyond what chance gives |

The metrics that read a probability instead of a class are in [Probabilities](../probabilities/index.md).

---

## Details

### Choices

- **The example** uses the same ten days as the three pages: `crash` is what happened and `alarm` is what the alarm said.
- **The map** is drawn with the library by `docs/assets/maps/make_classification_map.py`; the classes are the names of the catalog of `usable`.
