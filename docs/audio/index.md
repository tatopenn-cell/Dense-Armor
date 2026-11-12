# How do I work with sound in Dense-Armor?

Sound is a long list of numbers, one for each tiny slice of time. The audio section gives you
a value for it (`Audio`), a way to read and write files, a way to look at it by frequency
(the short-time Fourier transform), a few sounds you can make from nothing, a filter, and a
way to hand each block of sound to the estimators Dense-Armor already has. Everything uses
NumPy only; a live input needs the optional extra `audio`.

## 1. Make a sound and save it

A plucked string is the simplest sound that needs no input file: a burst of random values
that dies away. `plucked_string(440.0, 8000, 0.5)` makes half a second of it at 440 Hz with
8000 samples per second.

```python
import numpy as np
from dense_armor.utility.signals.audio import plucked_string, read_wav, write_wav
a = plucked_string(440.0, 8000, 0.5)
write_wav("pluck.wav", a)
b = read_wav("pluck.wav")
print(b.sr, b.n, round(b.duration, 2))
```

```text
8000 4000 0.5
```

`write_wav` writes 16-bit PCM by default (`pcm24`, `pcm32` and `float32` are also available);
`read_wav` returns an `Audio` with the sample rate, the samples and the start time `t0`.

## 2. Look at it by frequency, and go back

The short-time Fourier transform cuts the sound into overlapping frames, multiplies each by a
window and takes its spectrum. With a Hann window and a hop of half the frame, adding the
inverse of the frames gives the sound back exactly in the middle of the signal.

```python
from dense_armor.utility.signals.audio import istft, stft
S = stft(b.x, 256, 128, "hann")
y = istft(S, 128, "hann", 256)
print(S.shape, bool(np.max(np.abs(y[128:-128] - b.x[128:len(y) - 128])) < 1e-9))
```

```text
(30, 129)
True
```

There are 30 frames of 129 frequencies each. `cola_ok(window, hop)` tells you whether a
window and a hop can be inverted this way, and `StreamingSTFT` gives the same spectra one
sample at a time.

## 3. Filter it

A biquad filter is the smallest filter: it mixes the current sample with the two before it
and the two previous outputs. `lowpass(500.0, 8000.0)` gives the coefficients that keep what
is below 500 Hz.

```python
from dense_armor.utility.signals.audio import Audio, Biquad, lowpass
n = Audio(np.random.default_rng(0).standard_normal(8000), 8000)
f = Biquad(*lowpass(500.0, 8000.0))
z = f.update_many(n.x)
print(round(float(z.std() / n.x.std()), 3))
```

```text
0.373
```

White noise has the same strength at every frequency, so the filtered noise keeps a fraction of
it. `highpass`, `bandpass`, `notch`, `peaking`, `lowshelf` and `highshelf` work the same way.

## 4. Find the moment something happens

`detect_events` reads blocks of sound, turns each into a few numbers (the mean magnitude of
eight bands of frequencies), scores the first band against the blocks before it, and gives the
score to a drift detector. Noise for three seconds with a 440 Hz tone that starts at two
seconds:

```python
from dense_armor.utility.signals.audio import white_noise
from dense_armor.utility.signals.audio.events import detect_events
t = np.arange(24000) / 8000
x = 0.1 * white_noise(24000, 1) + np.where(t >= 2.0, np.sin(2 * np.pi * 440 * t), 0.0)
bl = [Audio(x[i:i + 512], 8000, i / 8000) for i in range(0, 24000 - 511, 512)]
print(detect_events(bl)[0][0])
```

```text
1.984
```

The first alarm is at 1.984 s: the block that starts at 1.984 s ends at 2.048 s and contains
the start of the tone. The same function takes the blocks of `AudioFileStream` (a file) or
`AudioLiveStream` (a live input).

## 5. Give the sound to any estimator

`run_estimators(blocks, {"name": estimator, ...}, label=...)` turns each block into the dict
of band magnitudes and calls every estimator in the way its class asks for: a drift detector
gets one number, an anomaly scorer scores and then learns, a classifier or regressor predicts
and then learns with `label(block)`, a scaler learns and transforms. The tests run it with 27
estimators from the online anomaly, drift, cluster, classifier, regressor, tree, neighbour,
time-series, scaler and statistics groups, on a synthetic signal.

## Details

**Sources.** Windows: Harris (1978), Proc. IEEE 66(1), Eq. (28a) for the Hann window, the
Hamming section on PDF page 12 (`a = 0.54`, with `a = 25/46` the exact value), Eq. (31b) for
the Blackman window (0.42, 0.50, 0.08). STFT and overlap-add: Allen and Rabiner (1977), Proc.
IEEE 65(11). Biquads: the Audio EQ Cookbook of R. Bristow-Johnson, a technical note and not a
peer-reviewed paper. Plucked string: Karplus and Strong (1983), Computer Music Journal 7(2),
pages 3 and 4 of the PDF. Additive sawtooth and square: Hayes et al. (2023), Eq. (10) and (11).
PolyBLEP: the residual of Eq. (1) of Du et al. (2026), PDF page 2. Non-negative matrix
factorisation: Lee and Seung (2001), NIPS 13, Eq. (4) and (5). Resampling: Falkenberg (2026),
Algorithm 3, a preprint by one author. Frequency and cents: the same preprint, Algorithm 1.
Hard clipping: Esqueda, Valimaki and Bilbao (2016), Eq. (14).

**Choices stated.** The Karplus-Strong `y[-1]` is 0 and `p = round(sr / f - 1/2)`. The PolyBLEP
subtracts both residual terms: Eq. (2) of the source adds the second term, and with that sign
the energy outside the harmonics of a sawtooth at 3141.5926 Hz and 44100 Hz is -10.2 dB, no
better than the naive sawtooth at -10.9 dB, while subtracting both gives -27.9 dB (the test
`test_polyblep_has_less_aliasing_than_naive` measures the two). The NMF floors its denominators
at 1e-12. The separation keeps the phase of the mixture. The band grouping of `AudioFeatures`
is the mean magnitude over nearly equal groups of bins, with no source behind it. The
resampler interpolates linearly, as the source does, so its stopband is not that of a windowed
FIR filter. Ties in `freq_to_note` round half to even.

**Not in this section.** The triangle wave and polyBLAMP clipping (the formulas of
Esqueda et al. are not implemented), pink noise and the ADSR envelope (the sources are a scan
and an equation whose brackets are lost in the text), the chirp, FIR and IIR filters of any
order, the delay line, convolution, the envelope follower and gain curve, waveshapers,
dithering and quantisation, pan, mid/side and interaural delay, RMS and loudness meters, mel
and MFCC, pitch, onset, GCC-PHAT, scales and note events: no source in the project folder
defines them.

**Measured cost.** Each stateful primitive declares `budget_s` and `memory_class`; the values
are the ones in the classes and have not been re-measured on a reference machine.
