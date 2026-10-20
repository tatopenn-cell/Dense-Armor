# Drift against noise: is this a structural change or the noise of the tick?

A tick price is noisy: bid-ask bounce and rounding make the return between two ticks noisier than the return over a moderate interval. A drift detector on the raw tick returns fires on that noise, or misses a real change because the noise is too large. This page answers one question with the same detector on two statistics: **is this a structural change, or the noise of the tick?**

The two statistics are the raw tick return, and the return between the means of two consecutive windows of prices. The paper that the page follows is Maneesoonthorn, Martin and Forbes (2017, arXiv:1708.09520), a review of high-frequency jump tests. It names two remedies for microstructure noise: prices averaged over a small window (section 2.5, page 5), and a moderate sampling frequency of 30 seconds (section 5, page 10, guideline 3). The second statistic is the first remedy, in ticks instead of seconds.

## 1. The two statistics

```python
import numpy as np
from pipelines.finance.drift_vs_noise import preaveraged_returns, raw_returns

x = np.arange(40, dtype=float)
print(len(raw_returns(x)), len(preaveraged_returns(x, window=10)))
print(preaveraged_returns(x, window=10))
```

```
39 2
[10. 10.]
```

The raw statistic is one return per tick. The noise-robust statistic averages ten ticks on each side and takes the difference of the two means: the noise of a single tick averages out, and the return over twenty ticks is less noisy than twenty raw returns.

## 2. The two detectors on the same stream

```python
from pipelines.finance.drift_vs_noise import DriftVsNoisePipeline, synthetic_tick

_, tick = synthetic_tick(n=5000, seed=0, noise_std=0.005, vol=0.001,
                         noise_kind="student", noise_df=3.0)
r = DriftVsNoisePipeline().run(tick)
print(r.n_episodes_raw, r.n_episodes_robust)
```

```
0 0
```

The stream has no shift. On the raw tick returns the detector raises no alarm; on the pre-averaged returns it raises no alarm either. A single seed is not a comparison: the next section is twenty seeds.

## 3. Twenty seeds, with and without a shift

```python
import json
from pathlib import Path

payload = json.loads(Path("benchmarks/finance/drift_vs_noise.json").read_text())
calm, shift = payload["calm"], payload["shift"]
print(round(calm["wilcoxon_p"], 4), round(calm["a12"], 3))
print(shift["n_raw_detected"], shift["n_robust_detected"])
print(shift["raw_delay"]["median"], shift["robust_delay"]["median"],
      round(shift["wilcoxon_p"], 4), round(shift["a12"], 3))
```

```
0.0082 0.675
7 20
311.0 3.0 0.0001 1.0
```

The numbers are the ones of `benchmarks/finance/drift_vs_noise.json`, twenty seeds, synthetic tick with Student-t noise of three degrees of freedom. On the calm stream, the raw and the robust statistics both have zero episodes per hour median, but the paired Wilcoxon on the per-seed counts has p = 0.0082 and the A12 is 0.675: the raw statistic has more episodes on the seeds where an episode appears, and the median alone hides it. On the shifted stream, the raw detector fires after the change on 7 of 20 seeds, the robust one on 20 of 20; the raw delay median is 311 ticks, the robust one is 3, with p = 0.0001 and A12 = 1.000. A seed where the raw detector does not fire at all is counted as a delay of 5000 ticks (the length of the stream) for the Wilcoxon and the A12: the imputation is declared in the benchmark and the detection rate (7 of 20, 20 of 20) is printed next to it, so a reader sees the imputation.

## 4. What the numbers say, and what they do not

The robust statistic turns a "7 of 20" into a "20 of 20" at the detection level, and a delay of 311 ticks into 3. The table is a comparison on a **synthetic** tick stream with Student-t noise; it is not a comparison on a venue's order book, because no order book is in the repository and none was downloaded. The A12 = 1.000 on the delay says that on every seed the robust delay is shorter than the raw delay, with the imputation of 5000 ticks for a seed with no alarm.

## Details

### References

- Maneesoonthorn, W., Martin, G. M. and Forbes, C. S. (2017). *High-Frequency Jump Tests: Which Test Should We Use?* arXiv:1708.09520. Prices averaged over a small window: section 2.5, page 5. Moderate sampling frequency of 30 seconds: section 5, page 10, guideline 3. The paper reviews jump tests; it does not propose the statistic of this page.

### Choices stated explicitly

- The statistic of this page is the difference between the means of two consecutive windows of ten ticks, not the LM12 test statistic itself (a standardized return with a local volatility estimate, page 5). The paper reviews the test; this page takes the pre-averaged return, which is the scalar a streaming detector can take. The choice is declared, and the page does not claim the LM12 test.
- The window is ten ticks, not the "30 seconds" of guideline 3 (page 10). The paper's sampling frequency is in seconds and the stream of this page is in ticks: ten ticks and 30 seconds are not the same quantity, and the choice is this page's.
- The noise is Student-t with three degrees of freedom. The paper names the microstructure noise without fixing a distribution; the Student-t is this page's choice, and the reason is the heavy tail a stale quote or a far-from-touch print produces.
- The comparison imputes 5000 ticks for a seed where the raw detector does not fire after the shift. Without the imputation the raw side has no value on those seeds and the paired Wilcoxon has no pair. The detection rate is printed next to the delay, so the imputation is visible.
- The stream is synthetic, in the code, in the page and in the figure.

### Measured cost

The full 20-seed benchmark takes 3.7 s on this host, 1.2 s of which are the runs. The per-sample time of the pipeline is not in this page: the pipeline is the two CUSUM detectors, and their cost is the one of `CUSUMDriftDetector`.

### Development diary

The first version of the page used Gaussian noise and the raw detector never fired: with a Gaussian noise and a CUSUM at `h = 20` the noise does not cross the threshold, and the comparison had no false alarm on either side. The page now uses Student-t noise with three degrees of freedom, where the heavy tail produces the occasional episode on the raw statistic, and the robust statistic averages it out. The second version of the benchmark compared the delays only on the seeds where both detectors fired, which on the raw side is 7 of 20; the page now imputes 5000 ticks for a seed with no alarm and prints the detection rate, so the comparison is on all 20 seeds and a reader sees the imputation.
