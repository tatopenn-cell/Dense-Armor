# Events and streams

This page reads sound from a file or a live input, turns each block into a few numbers, and finds the moment something happens. Everything here uses estimators that Dense-Armor already had.

## 1. Blocks from a file

`AudioFileStream(path, block, hop)` yields blocks of `block` samples every `hop` samples from a WAV file (`hop` defaults to `block`). The `t0` of a block is the index of its first sample divided by the sample rate. A last block shorter than `block` is not yielded.

```python
from dense_armor.core.audio import AudioFileStream, plucked_string, write_wav

write_wav("pluck.wav", plucked_string(440.0, 8000, 0.5))
blocks = list(AudioFileStream("pluck.wav", 1024))
print(len(blocks), [round(b.t0, 3) for b in blocks])
```

```text
3 [0.0, 0.128, 0.256]
```

## 2. Blocks from a live input

`AudioLiveStream(sr, block, queue_size, device)` takes blocks from a sound card through the library `sounddevice` (the extra `audio`), imported only when you build the stream. A callback puts each block in a bounded queue; when the queue is full the block is dropped and `dropped` grows by one, so a slow reader never blocks the input. `read(timeout)` returns the next block, or `None`. Any object with `start()`, `stop()` and `close()` can replace the sound card through the `backend` argument, which is how the tests run it without one.

## 3. A block becomes a few numbers

`AudioFeatures(n_bands)` takes the magnitude of the spectrum of a block (windowed, as in the short-time Fourier transform) and averages it in `n_bands` groups of nearly equal size. The keys are `b0 .. b{n_bands-1}`, always the same, so the dict goes straight into `learn_one` of any estimator. This grouping is a choice made here: no source defines it. A block with fewer frequency bins than bands raises `ValueError`.

## 4. The moment something happens

`detect_events(blocks)` scores the first band of each block against the blocks before it with `HampelScorer` (the score of a value against the median and the spread of the values that precede it) and gives the score to a `CUSUMDriftDetector`. It returns the start time and score of each block where the detector fires. Here is noise for three seconds with a 440 Hz tone that starts at two seconds:

```python
import numpy as np
from dense_armor.core.audio import Audio, detect_events, white_noise

t = np.arange(24000) / 8000
x = 0.1 * white_noise(24000, 1) + np.where(t >= 2.0, np.sin(2 * np.pi * 440 * t), 0.0)
bl = [Audio(x[i:i + 512], 8000, i / 8000) for i in range(0, 24000 - 511, 512)]
print(detect_events(bl)[0][0])
```

```text
1.984
```

The first alarm is at 1.984 s: the block that starts at 1.984 s ends at 2.048 s and contains the start of the tone. The same function takes the blocks of `AudioFileStream` (a file) or `AudioLiveStream` (a live input), and gives the same alarms for the same samples.

## 5. Any estimator, in the form it takes

`detect_events` is one path built by hand. `AudioInput(primitive)` is the general one: it gives a block of sound to any primitive of Dense-Armor, in the form the primitive takes (the table is in the [overview](index.md)), and `run(blocks, label)` feeds every block of a stream. When a primitive raises on the dict, the adapter tries again with one number, and from then on gives it the number.

`run_estimators(blocks, {"name": primitive, ...}, label=...)` does the same for several primitives at once and computes the features of each block one time. It returns, for each name, the `mode`, the number of blocks `n`, the `last` answer, the `alarms`, the `error` if the primitive raised (it is then not called again) and `scalar`. The tests run it with 27 estimators of the online anomaly, drift, cluster, classifier, regressor, tree, neighbour, time-series, scaler and statistics groups, on a synthetic signal.

```python
import numpy as np
from dense_armor.core.audio import Audio, run_estimators
from dense_armor.core.online.anomaly.filters import HampelScorer
from dense_armor.core.online.drift.detector import CUSUMDriftDetector
from dense_armor.core.signals.preprocessing.scale import StandardScaler

rng = np.random.default_rng(1)
blocks = [Audio(rng.standard_normal(256), 8000, k * 256 / 8000) for k in range(12)]
rep = run_estimators(blocks, {"hampel": HampelScorer(), "cusum": CUSUMDriftDetector(), "scaler": StandardScaler()})
print({k: (v["mode"], v["n"], v["error"]) for k, v in rep.items()})
```

```text
{'hampel': ('score', 12, None), 'cusum': ('drift', 12, None), 'scaler': ('transform', 12, None)}
```
