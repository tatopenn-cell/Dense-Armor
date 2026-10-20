# Stress: what do the existing finance pipelines do in a flash crash?

A flash crash is a large price move within a very short time, followed by a partial recovery. This page runs the two existing finance pipelines on a synthetic flash crash and reports what they do. The pipelines are not modified: the page asks what they do, not what they should do.

The shape of the crash is from Braun et al. (2017): the price moves "strongly within milliseconds" (abstract, page 1), "less than 40 percent recover completely" (page 1), and 25 percent are almost recovered after one trade (page 1). The crash of this page is synthetic and its parameters are declared below.

## 1. The crash

```python
from pipelines.finance.stress import crash_metrics, synthetic_flash_crash

crash = synthetic_flash_crash(n=2000, seed=0, crash_at=1000, duration_ticks=5,
                              drop=0.05, recovery_fraction=0.6, vol=0.0)
m = crash_metrics(crash)
print(round(m["realised_drop"], 3), round(m["realised_recovery"], 3))
```

```
0.05 0.6
```

The price drops by 5% over five ticks and recovers 60% of the drop over the next five. The drop and the recovery are the parameters; with `vol=0` the two are exact. The page calls `crash_metrics` to show them, and the generator is the one the benchmark uses.

## 2. The surveillance pipeline on the crash

```python
import json
from pathlib import Path

payload = json.loads(Path("benchmarks/finance/stress.json").read_text())
s = payload["surveillance"]
print(s["seeds"], s["ticks"], s["alarms"]["median"],
      s["n_with_delay"])
```

```
5 100 0.0 0
```

`FinancePipeline` sees 100 ticks and 5 seeds, around a crash at tick 50. It raises **zero** alarms, on every seed: the pipeline does not see the crash. The reason is the calibration. `FinancePipeline` is built for daily or minute returns, with the ranges of `range_ret` and `range_vol` set for that scale, and Orca on every tick. A crash of five ticks, on a price path that walks with `vol=0.001` per tick, is not outside the ranges the pipeline expects. The page states the result and does not tune the pipeline: the task asks what the existing pipeline does, and the answer is "nothing on this crash".

## 3. The execution pipeline on the crash

```python
import json
from pathlib import Path

payload = json.loads(Path("benchmarks/finance/stress.json").read_text())
e = payload["execution"]
print(e["seeds"], e["ticks"], e["refused"]["median"], e["refused"]["iqr"])
```

```
20 2000 2.0 1.25
```

`ExecutionPipeline` sees 2,000 ticks and 20 seeds. During the crash the order sizes are multiplied by ten, and the pipeline refuses the largest ones: median 2.0 refusals per run, IQR 1.25. The refusal is the `size > max_size_multiple * mean` rule, with the default `max_size_multiple=10`: an order ten times the rolling mean is refused outright.

## 4. What the numbers say, and what they do not

The execution pipeline reacts to the crash, the surveillance pipeline does not. The page does not say that the surveillance pipeline is wrong: it is calibrated for a different time scale, and the page says so. The crash is synthetic, the order sizes and the relative volumes are synthetic (log-normal around one, multiplied by ten during the crash), and the two pipelines are not modified. A tuning of the surveillance pipeline for tick-level crashes is not on this page; it would be a change to the pipeline, and the task does not ask for it.

## Details

### References

- Braun, T., Fiegen, J. A., Wagner, D. C., Krause, S. and Guhr, T. (2017). *Impact and Recovery Process of Mini Flash Crashes: An Empirical Study.* arXiv:1707.05580. Duration in milliseconds, partial recovery, the fraction below 40 percent and the 25 percent after one trade: abstract, page 1.
- `pipelines/finance/surveillance.py`: `FinancePipeline`, not modified.
- `pipelines/finance/execution.py`: `ExecutionPipeline`, not modified.

### Choices stated explicitly

- The crash is synthetic: five ticks of drop, five ticks of recovery, and a Gaussian walk with `vol=0.001` per tick around them. The paper measures on real trade data; the page generates a synthetic event with the paper's shape and says so.
- The order sizes and the relative volumes are log-normal around one with standard deviation 0.2, multiplied by ten during the crash. The choice is this page's: the paper measures the price, not the sizes, and the multiplier is what makes the refusal rule fire.
- The surveillance pipeline runs 100 ticks and 5 seeds, not 2,000 and 20: it calls Orca on every tick (121 ms/tick on this host), and 2,000 ticks and 20 seeds would be 80 minutes. The execution pipeline runs 2,000 ticks and 20 seeds because it costs 0.2 ms/order. The two budgets are declared, and a median over 5 seeds is a small-sample median.
- The page does not tune the surveillance pipeline. A tuning for the tick scale is a change to an existing pipeline, and the task forbids it.

### Measured cost

`FinancePipeline.process_tick` costs 121 ms per tick on this host, and `ExecutionPipeline.process_order` costs 0.2 ms per order. The full benchmark takes 23.9 s: 6.1 s for the 5 surveillance runs and 13.1 s for the 20 execution runs, plus the generation.

### Development diary

The first version of the benchmark fed a constant order size of 1.0 and a constant relative volume of 1.0, and the execution pipeline refused nothing and the surveillance pipeline raised nothing: a constant stream has nothing to detect. The page now multiplies the sizes and the volumes by ten during the crash, and the execution pipeline refuses 2.0 orders per run median. The surveillance pipeline still raises nothing, and the page says it: the calibration is for a different time scale, and the result is the result.
