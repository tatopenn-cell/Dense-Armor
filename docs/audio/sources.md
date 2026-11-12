# Sources and checks

Every formula of the audio section comes from a source in the folder of the audio task. This page lists, for each primitive, the source and the equation, and says **what was checked and how**, so you can see which claims were read in the paper and which were only compared with a second implementation.

How to read the last column:

- **paper**: the equation was read in the PDF and compared with the code.
- **reference**: the code was compared with a second implementation written from the equation (a loop, `scipy`, `sympy`), but the page itself was not read again for this check.

## 1. Primitive by primitive

| primitive | source | check |
| --- | --- | --- |
| windows: Hann, Hamming, Blackman | Harris 1978, Eq. (28a), (30), (31b), (33) | paper: `a = 25/46` and `0.54`, the Blackman coefficients 7938, 9240, 1430 over 18608, and the sum of cosines for `n = 0..N-1` are Eq. (33) |
| `stft`, `istft`, `cola_ok` | Allen and Rabiner 1977, Eq. (4.4) to (4.6) | paper: the overlap-add sum of the shifted windows is a constant; the test rebuilds a signal to 1e-9 in the interior |
| biquads | Bristow-Johnson, Audio EQ Cookbook | paper: the seven sets of coefficients were compared one by one with the text of the note |
| `FIR`, `convolve` | Hayes and others 2023, Eq. (6) | paper; reference: `numpy.convolve` |
| `DelayLine` | Karplus and Strong 1983, page 3 ("a delay line of length p") | paper |
| `KarplusStrong`, `plucked_string` | Karplus and Strong 1983, pages 3 and 4 | paper: `y[t] = (y[t-p] + y[t-p-1]) / 2`, period `p + 1/2` |
| additive sawtooth and square | Hayes and others 2023, Eq. (10) and (11) | paper; the code starts at phase 0 and the paper sums up to and including `n` (one sample earlier) |
| `sawtooth_polyblep` | Du and others 2026, Eq. (1) and (2) | paper: Eq. (2) prints a `+` before the second residual term. The code subtracts both, and the energy outside the harmonics of a sawtooth at 3141.5926 Hz and 44100 Hz is -27.9 dB, against -10.2 dB with the sign as printed and -10.9 dB for the naive wave |
| `square_polyblep`, `pulse_polyblep` | Valimaki and Pekonen 2012, Eq. (12) and (13) | paper: the delay of a discontinuity is the counter after it over the step; for a pulse, the counter after it exceeds the duty cycle, with the duty cycle subtracted |
| `triangle_blamp`, `hard_clip_blamp`, `hard_clip` | Esqueda, Valimaki and Bilbao 2016, Eq. (8) to (14), Table 1 | paper; reference: the four bottom rows of Table 1 equal the second integral of the cubic B-spline minus the ramp (`sympy`) and the polynomials of the code |
| `pink_noise` | Timmer and Koenig 1995, Sec. 3 (page 3 of the scan) | paper: read as an image |
| `ADSR` | Hayes and others 2023, Eq. (15) | reference: the envelope of the formula in the docstring; the page of the PDF was **not read again** for this check |
| `nmf`, `separate` | Lee and Seung 2001, Eq. (4) and (5) | reference: one update equals a loop written from the equations; the text of the PDF page loses the position of the indices and was **not read again** |
| `Resampler`, `resample` | Falkenberg 2026, Algorithm 3 (page 4) | paper: linear interpolation, `Y[n] = G((1-alpha)s0 + alpha s1)`, `u <- u + q` |
| `pitch_ratio_transition`, `rational_ratio` | Falkenberg 2026, Algorithm 4 (page 4), page 3 | paper |
| `freq_to_cents`, `freq_to_note`, `note_to_freq` | Falkenberg 2026, Algorithm 1 (page 3) | paper |
| `YIN`, `yin_f0`, `cmndf` | Strahl and Mueller 2025, Sec. II-A, Eq. (1) and (2) | paper; reference: a loop written from Eq. (1) and (2), and the same steps |
| `AllPole` | Yu and others 2024, Eq. (1) | paper; reference: `scipy.signal.lfilter` |
| `reflection_to_lpc` | Hayes and others 2023, Eq. (5) | paper: with the sign of Hayes Eq. (4) the result is unstable in 1570 of 2000 random cases, with the sign of Yu Eq. (1) in none |

## 2. Choices that the sources leave to the code

- The Karplus-Strong `y[-1]` is 0 and `p = round(sr / f - 1/2)`.
- The pink noise sets the component at zero frequency to 0 and scales to unit variance.
- The NMF floors a denominator at `1e-12`, and `separate` keeps the phase of the mixture.
- The grouping of `AudioFeatures` (the mean magnitude over groups of bins) has no source.
- `yin_f0` returns `nan` when no lag is below the threshold, and takes the threshold as an argument with no default.
- Ties in `freq_to_note` round half to even.
- The `budget_s` of each primitive is a measurement on one machine, not a guarantee: for example the p99 of `YIN` on a block of 1024 samples was 3.2e-4 s and its budget is 4e-3 s.

## 3. What is not in the section

Because no source in the folder defines them: the chirp (Farina 2000 is not in the folder), IIR filters other than the biquads and the all-pole filter, convolution by the FFT, the envelope follower and the gain curve of a compressor, waveshapers, dithering and quantisation, pan, mid/side and interaural delay, RMS and loudness meters (the K-weighting filter is in ITU-R BS.1770, which is not in the folder), mel and MFCC, onset detection, GCC-PHAT, scales and note events. YIN stops at step 2.

## 4. Where the section lives

The audio section is `dense_armor/core/audio/`. In this release the names of the folders `core` and `utility` of `dense_armor` were swapped: the library is now `dense_armor.core` and the shielding engine `dense_armor.utility`. The old paths still work (see the changelog).
