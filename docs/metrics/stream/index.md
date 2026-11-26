# On a stream: windows, predict then learn, model choice

[← Metrics](../index.md)

## The map of the stream pages

Click a page to open it, or a job on the left (or the button at the top) to go back to the [map of the library](../../index.md#the-library-folder-by-folder).

<object type="image/svg+xml" data="../../assets/maps/stream.svg" style="width:100%"></object>

A model that keeps learning lives on a stream: the data never stop, the world changes, and the label of a sample may arrive late. The metrics of the earlier pages score one pair at a time; this section says how to use them on a stream. A window keeps only the recent past, the predict-then-learn protocol scores each sample before the model has seen its label, and a model choice runs several models on the same stream and tells which one is ahead.

## A first example: the forecast of a stock index on a window

A forecast of the direction of a stock index is right on some days and wrong on others. Ten days make ten pairs: the accuracy of all the ten days, and the accuracy of the last five days only.

```python
from dense_armor.core.evaluation.metrics import Accuracy, Rolling

crash = [0, 0, 1, 0, 0, 1, 0, 0, 0, 1]
alarm = [0, 1, 1, 0, 0, 1, 0, 1, 0, 0]
whole, recent = Accuracy(), Rolling(Accuracy(), window=5)
for truth, flag in zip(crash, alarm):
    whole.update(truth, flag)
    recent.update(truth, flag)
print(whole.get(), recent.get(), recent.count)
```

```
0.7 0.6 5
```

The forecast is right on 7 days out of 10, so the accuracy of all the days is 0.7. In the last five days (days 6 to 10) it is right on days 6, 7 and 9 and wrong on days 8 and 10: 3 of 5, so the window reads 0.6, and `recent.count` is 5 because the window holds five pairs. When the forecast gets worse, the window shows it first; the next pages show how much first.

## The stream pages, one page each

| page | what it is | what it is for |
|---|---|---|
| [Rolling window](rolling.md) | `Rolling` | any metric over the last samples, or the last seconds |
| [Rolling AUC](rolling_auc.md) | `RollingAUC` | the ROC-AUC, exact, over the last samples |
| [Predict, then learn](predict_then_learn.md) | `progressive_val_score`, `progressive_val_proba_score` | score each sample before the model learns it, with late labels |
| [Choosing a model](model_selection.md) | `best_of`, `OnlineModelSelection` | several models on the same stream, and which one is ahead |

---

## Details

### Choices

- **The example** uses the same ten days as the Classification pages: `crash` is what happened and `alarm` is what the forecast said.
- **The map** is drawn with the library by `docs/assets/maps/make_stream_map.py`; the functions of the evaluation module are not in the catalog of `usable`, so their cards carry their names as a summary line.
