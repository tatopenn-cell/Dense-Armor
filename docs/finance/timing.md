# Timing: how steady is the time per sample?

A pipeline that runs in a control loop must fit its time budget not only on average, but at the tail. This page measures, for each finance pipeline and each new module of items A-D, the time per sample over 500 samples after a warm-up of 20: the median, the interquartile range, the 99th percentile, the maximum, and the ratio of the 99th percentile to the median. Never the bare mean.

The numbers are the numbers of this host, and the host is read by the script:

```text
CPU:    x86_64
Python: 3.13.16
arch:   x86_64
```

## 1. The table

| Pipeline | median ms | IQR ms | p99 ms | max ms | p99/median |
|---|---|---|---|---|---|
| `surveillance` | 5.7385 | 1.3854 | 11.7382 | 12.9520 | 2.05 |
| `execution` | 0.0041 | 0.0005 | 0.0056 | 0.0324 | 1.36 |
| `regime` | 0.2503 | 0.0184 | 0.5057 | 0.5332 | 2.02 |
| `drift_vs_noise` | 0.0525 | 0.0017 | 0.1096 | 0.1373 | 2.09 |
| `portfolio` | 0.0132 | 0.0003 | 0.0349 | 0.0997 | 2.65 |
| `order_book` | 0.0321 | 0.0004 | 0.0399 | 0.0472 | 1.24 |
| `stress` | 0.0017 | 0.0001 | 0.0031 | 0.0187 | 1.89 |
| `connectors` | 0.0005 | 0.0000 | 0.0007 | 0.0089 | 1.35 |

```python
import json
from pathlib import Path

rows = json.loads(Path("benchmarks/finance/jitter.json").read_text())["rows"]
print(round(rows["surveillance"]["median_ms"], 4),
      round(rows["surveillance"]["p99_over_median"], 2))
print(round(rows["connectors"]["median_ms"], 6),
      round(rows["connectors"]["p99_over_median"], 2))
```

```
5.7385 2.05
0.000514 1.35
```

## 2. What each row measures

Three rows are a whole pipeline on a stream, and four rows are one primitive, not the pipeline. The page says which is which:

- `surveillance`: `FinancePipeline.process_tick` on 500 ticks of a synthetic walk. It calls Orca on every tick, and it is the slowest row.
- `execution`: `ExecutionPipeline.process_order` on 500 log-normal sizes.
- `regime`: `RegimePipeline.run` on 500 returns of a synthetic two-regime stream, one return at a time.
- `drift_vs_noise`: `DriftVsNoisePipeline.run` on a sliding window of a synthetic tick stream.
- `portfolio`: `min_variance_weights` on the covariance of 500 rows, not the whole `PortfolioPipeline`.
- `order_book`: `order_flow_imbalance` on a sliding window of the synthetic book, not the regression.
- `stress`: `crash_metrics` on the generated crash, not the generator.
- `connectors`: `BoundedStream.push` and `pop`, one push and one pop per sample.

The four primitive rows are not called "pipeline" on the page, and a reader who compares `portfolio` to `surveillance` is comparing a weight formula to a five-stage pipeline: the page says it.

## 3. The tail

The ratio of the 99th percentile to the median is between 1.24 and 2.65. A ratio of 5 to 6 on the rows where the work is small (fractions of a millisecond) is the noise of the timer, not a control-loop risk: the p99 is under 0.506 ms on those rows. The row where the tail matters is `surveillance`, with a p99 of 11.74 ms on a median of 5.74 ms, a ratio of 2.05.

## Details

### References

- No paper in the folder is cited for this page. `ISTRUZIONI.md` cites none for item G, and the folder has no paper on timing or jitter. The metrics and the host statement are this implementation's choice.

### Choices stated explicitly

- The metrics are median, interquartile range, 99th percentile, maximum, and the ratio of the 99th percentile to the median. The page never reports the bare mean.
- The number of samples is 500 after a warm-up of 20. The smoke run of the script uses 50 samples and a warm-up of 5, and the two are declared in the JSON.
- Three rows are a whole pipeline and four rows are one primitive: the page says which is which, and does not call a primitive a pipeline.
- The host is read by the script (`platform.processor()`, `platform.python_version()`, `platform.machine()`), and the numbers are the numbers of that host. A number from another host is not comparable, and the page says so.

### Measured cost

The full run of the script takes 5.4 s for the two slowest rows, and under one second for each of the others.

### Development diary

The first version of the script measured `FinancePipeline.process_tick` with a constant relative volume of 1.0, and the median was 5.74 ms, while the stress benchmark of E2 measured 121 ms per tick on a stream with a multiplied volume. The difference is the input: Orca is called on every tick, and its cost depends on the input vector. The page reports the number of this stream (constant relative volume) and the stress page reports the number of the crash stream; the two are different inputs, and the page does not claim one number for the pipeline.
