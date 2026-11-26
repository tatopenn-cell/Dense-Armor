# Events: scores for a detector that raises an alarm

[← Metrics](../index.md)

## The map of events

Click a page to open it, or a job on the left (or the button at the top) to go back to the [map of the library](../../index.md#the-library-folder-by-folder).

<object type="image/svg+xml" data="../../assets/maps/events.svg" style="width:100%"></object>

The metrics of the earlier pages score one value at a time. A fault, a crash or a change is not one sample: it lasts, and a detector answers with alarms. Two alarms close together are one detection, an alarm outside every labelled event is a false alarm, and an event with no alarm is missed. The metrics of this section answer the questions an operator asks: how late was the alarm, how many times did it ring for nothing, how much of the event was caught, and is the detector better than a coin.

## A first example: flash crashes on a stock tick

A trading desk watches a stock tick. Two flash crashes were labelled afterwards: one from second 10 to second 20 and one from second 40 to second 45. The detector rang at seconds 5, 12, 12.5 and 30, over one hour of normal operation.

```python
from dense_armor.core.evaluation.metrics import EventMetrics

m = EventMetrics([(10.0, 20.0), (40.0, 45.0)], dt=0.5, normal_time_s=3600.0)
for t in (5.0, 12.0, 12.5, 30.0):
    m.add_alarm(t)
r = m.report()
print(r["detection_delays_s"], r["missed"], r["false_alarms_per_hour"], round(r["range_f1"], 3), round(r["nab_score"], 2))
```

```
[2.0, None] 1 2.0 0.067 43.93
```

The first crash starts at 10 s and the first alarm inside it is at 12 s: delay 2 s. The second crash has no alarm: missed, and its delay is `None`. The alarms at 5 s and 30 s are outside every event: two false alarms over one hour of normal operation, 2 per hour. The other two numbers, the range F1 of 0.067 and the NAB score of 43.93, are explained on their own pages below; `report()` returns all of them in one dict.

## The events pages, one page each

| page | classes | what it is for |
|---|---|---|
| [Delay and false alarms](delays.md) | `EventMetrics`, `EventWindow` | how late the alarm was, which events were missed, how often it rang for nothing |
| [Range precision and recall](range.md) | `EventMetrics` | how much of the event the alarms covered, and where in the event |
| [The NAB score](nab.md) | `EventMetrics` | one score that rewards early detection and punishes false alarms |
| [A detector on a stream](evaluate_events.md) | `evaluate_events` | run a drift or anomaly detector over a stream and get all the numbers |
| [Point-adjusted F1](point_adjusted_f1.md) | `PointAdjustedF1` | a score from the papers, and why a random flag fools it |

---

## Details

### Choices

- **The example** uses the same two events and the same four alarms on the first three pages.
- **The map** is drawn with the library by `docs/assets/maps/make_events_map.py`; `evaluate_events` is a function, not a class of the catalog of `usable`, so its card carries its name as a summary line.
