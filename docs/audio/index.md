# How do I work with sound in Dense-Armor?

Sound is a long list of numbers, one for each tiny slice of time. The audio section gives you
a value for it (`Audio`), a way to read and write files, a way to look at it by frequency
(the short-time Fourier transform), a few sounds you can make from nothing, a filter, and a
way to hand each block of sound to the estimators Dense-Armor already has. The sound itself
is handled with NumPy (the estimators it connects to need JAX); a live input needs the
optional extra `audio`.

## 1. Make a sound and save it

A plucked string is the simplest sound that needs no input file: a burst of random values
that dies away. `plucked_string(440.0, 8000, 0.5)` makes half a second of it at 440 Hz with
8000 samples per second.

```python
import numpy as np
from dense_armor.utility.audio import plucked_string, read_wav, write_wav
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
from dense_armor.utility.audio import istft, stft
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
from dense_armor.utility.audio import Audio, Biquad, lowpass
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

`FIR` is the other kind of filter: it adds the last few inputs, each multiplied by a weight
(the taps `h`). `convolve` does the same on a whole block, and `DelayLine` repeats a sample
`d` steps later:

```python
from dense_armor.utility.audio import FIR, DelayLine, convolve
print(convolve([1.0, 2.0, 3.0], [1.0, 1.0]).tolist())
print(FIR([0.5, 0.5]).update_many([1.0, 1.0, 3.0]).tolist())
print(DelayLine(2).update_many([1.0, 2.0, 3.0, 4.0]).tolist())
```

```text
[1.0, 3.0, 5.0, 3.0]
[0.5, 1.0, 2.0]
[0.0, 0.0, 1.0, 2.0]
```

The filter with taps 0.5 and 0.5 averages two neighbouring samples, which smooths the sound.

## 4. Round the corners, shape the loudness, add noise

A triangle wave and a clipped sine both have sharp corners, and a sharp corner puts energy at
frequencies the sound should not have (aliasing). `triangle_blamp` and `hard_clip_blamp` round
each corner with the four-point polyBLAMP of Esqueda, Valimaki and Bilbao. The number below is
the share of the energy that lies away from the harmonics, in dB, at 3141.5926 Hz and 44100 Hz:

```python
import numpy as np
from dense_armor.utility.audio import hard_clip, hard_clip_blamp, triangle_blamp
sr, f0 = 44100, 3141.5926
def db(x):
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    m = np.zeros(len(X), bool)
    for h in range(1, int(sr / 2 / f0) + 1):
        c = int(round(h * f0)); m[max(0, c - 3):c + 4] = True
    return round(float(10 * np.log10(X[~m].sum() / X.sum())), 1)
p = (np.arange(sr) * f0 / sr) % 1
print(db(1 - 4 * np.abs(p - 0.5)), db(triangle_blamp(f0, sr, sr)))
s = 2 * np.sin(2 * np.pi * f0 * np.arange(sr) / sr)
print(db(hard_clip(s, 1.0)), db(hard_clip_blamp(s, 1.0)))
```

```text
-34.3 -42.3
-31.3 -42.2
```

The first line is the triangle (plain, then corrected), the second the clipped sine. The
corrected corners are slightly rounded: the peaks of the triangle are 0.133 lower.

`ADSR` gives the loudness envelope of a note (attack, decay, sustain, release) and `pink_noise`
noise whose power falls as one over the frequency:

```python
from dense_armor.utility.audio import ADSR, pink_noise
e = ADSR(0.1, 0.2, 0.5, 0.3, 1.0, 1000)
print([round(e.value(t), 2) for t in (0.0, 0.05, 0.1, 0.2, 0.3, 1.0, 1.15, 1.3)])
x = pink_noise(1 << 16, 3)
X = np.abs(np.fft.rfft(x)) ** 2
print(round(float(np.polyfit(np.log(np.arange(1, len(X))), np.log(X[1:]), 1)[0]), 2))
```

```text
[0.0, 0.5, 1.0, 0.75, 0.5, 0.5, 0.25, 0.0]
-1.01
```

The envelope rises to 1 in 0.1 s, falls to the sustain level 0.5, holds it until the note ends at
1 s and returns to 0 in 0.3 s. The slope of the noise spectrum on a log-log plot is -1.01.

## 5. Choose an oscillator

An oscillator makes a repeating wave, one sample at a time, at the frequency you give it at
each step, so the pitch can glide without a click. `Oscillator(kind, sr)` has eight kinds:

| kind | wave | how it is made |
| --- | --- | --- |
| `sine` | sine | `sin(2 pi phi)` |
| `sawtooth_additive`, `square_additive`, `triangle_additive` | sawtooth, square, triangle | sum of the harmonics below half the sample rate |
| `sawtooth_polyblep`, `square_polyblep`, `pulse_polyblep` | sawtooth, square, pulse with `duty` | the naive wave with its jumps smoothed (polyBLEP) |
| `triangle_blamp` | triangle | the naive wave with its corners rounded (polyBLAMP), two samples late |

```python
import numpy as np
from dense_armor.utility.audio import Oscillator
f = np.full(8, 1000.0)
for k in ("sine", "triangle_additive", "square_polyblep", "triangle_blamp"):
    print(k, Oscillator(k, 8000).update_many(f).round(2).tolist())
print(Oscillator("pulse_polyblep", 8000, 0.25).update_many(f).round(2).tolist())
```

```text
sine [0.0, 0.71, 1.0, 0.71, 0.0, -0.71, -1.0, -0.71]
triangle_additive [0.0, 0.51, 0.9, 0.51, 0.0, -0.51, -0.9, -0.51]
square_polyblep [0.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0, -1.0]
triangle_blamp [0.0, 0.0, -0.77, -0.49, 0.0, 0.49, 0.77, 0.49]
[0.0, 1.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0]
```

The additive kinds have no aliasing but cost more at low frequencies (one sine per harmonic);
the polyBLEP and polyBLAMP kinds cost the same at any frequency and keep a little aliasing.
`triangle_blamp` needs two samples of look-ahead around each corner, so its output is the
sample two calls earlier (the first two are 0) and its frequency must stay below a quarter of
the sample rate. The pulse is `+1` for the first `duty` of each period and `-1` after it, so
the last line is a pulse of a quarter period at 1000 Hz and 8000 samples per second.

`chirp(f0, f1, duration, sr)` sweeps the frequency of any kind from `f0` to `f1`, in a line
(`method="linear"`) or in a geometric ratio (`method="log"`):

```python
from dense_armor.utility.audio import chirp
c = chirp(100.0, 1000.0, 1.0, 8000)
z = np.diff(np.sign(c.x)) != 0
print(c.n, int(z[:800].sum()), int(z[-800:].sum()))
```

```text
8000 29 191
```

The wave crosses zero 29 times in the first tenth of a second and 191 times in the last, so
the pitch rose. White noise, pink noise and the plucked string are in section 4 and section 1.

## 6. Find the moment something happens

`detect_events` reads blocks of sound, turns each into a few numbers (the mean magnitude of
eight bands of frequencies), scores the first band against the blocks before it, and gives the
score to a drift detector. Noise for three seconds with a 440 Hz tone that starts at two
seconds:

```python
import numpy as np
from dense_armor.utility.audio import Audio, detect_events, white_noise
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

## 7. Give the sound to any estimator

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
Algorithm 3, a preprint by one author. Frequency and cents: the same preprint, Algorithm 1. FIR filter and convolution: Hayes et al. (2023), Eq. (6). Delay line: Karplus and Strong (1983), page 3.
Hard clipping and the triangle: Esqueda, Valimaki and Bilbao (2016), Eq. (8) to (14) and Table 1. Pink noise: Timmer and Koenig (1995), Sec. 3. ADSR: Hayes et al. (2023), Eq. (15).

**Choices stated.** The Karplus-Strong `y[-1]` is 0 and `p = round(sr / f - 1/2)`. The PolyBLEP
subtracts both residual terms: Eq. (2) of the source adds the second term, and with that sign
the energy outside the harmonics of a sawtooth at 3141.5926 Hz and 44100 Hz is -10.2 dB, no
better than the naive sawtooth at -10.9 dB, while subtracting both gives -27.9 dB (the test
`test_polyblep_has_less_aliasing_than_naive` measures the two). The NMF floors its denominators
at 1e-12. The separation keeps the phase of the mixture. The band grouping of `AudioFeatures`
is the mean magnitude over nearly equal groups of bins, with no source behind it. The
resampler interpolates linearly, as the source does, so its stopband is not that of a windowed
FIR filter. Ties in `freq_to_note` round half to even. The polyBLAMP residual is derived by integrating the cubic B-spline twice and subtracting the ramp, as the paper describes: it reproduces three of the four bottom rows of Table 1 exactly, and the fourth row is garbled in the text layer of the PDF. The pink noise sets the component at zero frequency to 0 and scales to unit variance, which the paper does not fix. The clipper has a block form only, with no streaming form. The additive triangle is the textbook Fourier series, the square and the pulse are the naive wave with the sawtooth polyBLEP residual added at the rising edge and subtracted at the falling edge, and the chirp feeds the frequency at the time of each sample to an oscillator: no source in the project folder defines these four. The streaming `triangle_blamp` equals the block `triangle_blamp` two samples later, except for the last two samples of the block, whose corners lie beyond its end.

**Not in this section.** IIR filters of any order, convolution by the FFT,
the envelope follower and gain curve, waveshapers, dithering and quantisation,
pan, mid/side and interaural delay, RMS and loudness meters, mel and MFCC, pitch, onset,
GCC-PHAT, scales and note events: no source in the project folder defines them.

**Measured cost.** Each stateful primitive declares `budget_s` and `memory_class`; the values
are the ones in the classes and have not been re-measured on a reference machine.
