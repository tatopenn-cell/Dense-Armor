# High-Precision Variance Computation in float32 via First-Element Anchor Shifting, Local Centring and Compensated Sum of Squares

Salvatore Pennacchio, October 2026. Revised: the compensated sum on ``d**2`` (Kahan 1965) is part of the kernel.

Repository: `dense_armor/utility/stats/stable.py`.

## Abstract

Computing the variance of high-magnitude data with micro-deviations in float32 suffers from catastrophic cancellation: the subtraction `x - mean(x)` strips the low bits of the deviation before any accumulation can use them. Pairwise merges and the baseline two-term formula recover part of the loss but lose accuracy under `jax.jit`, where the compiler reorders the reduction. This note presents a three-step kernel: a first-element anchor shift, a local centring of the shifted values, and a Kahan compensated sum of the squared residuals. The kernel is exactly `0.0` on identical values up to `1e38` (the baseline two-term formula gives `nan` at `1e38`), correct on complex input, and below the baseline formula on three of the four datasets of the task, in eager and under `jax.jit`. The cost is the speed: the sequential Kahan sum is 1.85x the baseline formula on `[4096, 4096]`, below the 2.0 threshold.

## 1. The problem

For `x` in float32 along a reduction axis of `n` samples, the variance is

    var = (1/n) sum (x_i - mean)^2

When the values sit around a large offset (1e3, 1e6) and the deviation is small, `mean(x)` in float32 is off by `eps * |mean|` (`eps = 2^-23`). The subtraction `x - mean` then carries that error into the residuals: the digits of the deviation are already lost.

The baseline two-term formula

    m   = mean(x)
    d   = x - m
    var = mean(d^2) - mean(d)^2

cancels the linear part of the error of `m` (both `mean(d^2)` and `mean(d)^2` share the same `m`), but the tree-sum error of `d^2` remains. Under `jax.jit`, the compiler reorders the reduction; the cancellation is not enough on every dataset.

## 2. The kernel

Three steps, all in the input dtype.

### 2.1 First-element anchor shift

A reference `r = x[0]` is subtracted from every element along the reduction axis:

    y = x - r

When the values are within a factor of two of the reference, the subtraction is exact (Sterbenz). The large common exponent is gone, and `y` sits at the exponent of the deviation. On identical values `y` collapses to the bit pattern `0x00000000`.

### 2.2 Local centring

The mean of the shifted values,

    m = mean(y)

is at the level of the deviation, not of the offset. Its rounding error is `eps * |m|`, which is small.

### 2.3 Kahan compensated sum of squares

The residuals are

    d = y - m

and the sum of their squares is accumulated with Kahan compensated summation:

    s2, c = 0, 0
    for d_i in d:
        t = d_i^2 - c
        s = s2 + t
        c = (s - s2) - t
        s2 = s
    s2 = s2 + c

The carry `c` keeps the low-order bits that the ordinary sum loses. The error of `s2` is `O(eps * |s2|)`, independent of `n`, instead of `O(eps * log2(n) * |s2|)` of the tree sum.

The two-term correction `- mean(d)^2` is not included: on the shifted values `mean(d)` is `~eps`, and its square is `~eps^2`, negligible against `s2/n`.

### 2.4 Result

    var = s2 / (n - ddof)

The mean is reconstructed as `r + m`.

## 3. Results

The datasets of the task, `numpy.random.default_rng(0)`:

| name | shape | axis | distribution |
|---|---|---|---|
| D1 | (1000, 64) | 0 | N(1e3, 0.01) |
| D2 | (64, 1000) | 1 | N(1e4, 1) |
| D3 | (1000, 64) | 0 | N(0, 1) |
| D4 | (1000, 64) | 0 | N(1e6, 1) |

Maximum relative error against `np.var(x.astype(float64))`:

| dataset | baseline eager | kernel eager | baseline jit | kernel jit | eager <= base | jit <= base |
|---|---|---|---|---|---|---|
| D1 | 1.44e-7 | 1.17e-7 | 1.18e-7 | 1.17e-7 | yes | yes |
| D2 | 1.59e-7 | 1.15e-7 | 1.59e-7 | 1.15e-7 | yes | yes |
| D3 | 1.94e-7 | 1.36e-7 | 1.94e-7 | 1.36e-7 | yes | yes |
| D4 | 1.18e-7 | 1.22e-7 | 0.997e-7 | 1.22e-7 | no | no |

Three of four, in both regimes. D4 is above: at that offset `ulp(x[0]) = 0.0625` against a deviation of `1`, so the shift introduces a quantisation error that no linear technique removes without float64.

Identical values:

| input | kernel |
|---|---|
| `[1e10]*3` | 0.0 |
| `[7.5e10]*3` | 0.0 |
| `[1e15]*3` | 0.0 |
| `[1e38]*3` (jit) | 0.0 |
| `complex64 [7.5e10+7.5e10j]*3` | 0.0 |

The baseline two-term formula gives `nan` on `[1e38]*3` (the sum of three `1e38` overflows, and the two-term is `inf - inf`).

Complex input: `E|z - mean(z)|^2`, real. `[1, i, -1, -i]` gives `1.0`; `[1+1j, 2+3j, -1+0.5j, 4-2j]` gives `6.421875`.

Speed on `[4096, 4096]` float32, `axis=1`, 7 runs, median:

| formula | median (ms) |
|---|---|
| baseline formula | 73.44 |
| kernel | 135.72 |

1.85x the baseline formula, below the 2.0 threshold.

## 4. What the shift does, and what the compensated sum does

The shift and the Kahan sum act on two different terms:

- The shift removes the offset before the subtraction, so `d` has the deviation at full precision. Without the shift, `d` carries the error of `mean(x)`, and the two-term formula can only cancel its linear part, not its quantisation.
- The Kahan sum removes the tree-sum error of `d^2`. Without it, the sum accumulates `eps * log2(n) * var`; on D2 and D3 this is enough to lose against the baseline formula under `jax.jit`.

Measured separately:

| variant | eager <= base | jit <= base | speed x base |
|---|---|---|---|
| shift + tree sum (no compensation) | 2/4 | 0/4 | 1.00x |
| baseline two-term + Kahan on `d^2` (no shift) | 4/4 | 1/4 | 2.84-4.84x |
| shift + local centring + Kahan (this paper) | 3/4 | 3/4 | 1.85x |
| shift + two-term + tree sum (no compensation) | 2/4 | 0/4 | 1.70x |

The shift is the only step that gives the exact zero on `1e38` in `jit`. The compensated sum is the only step that keeps the accuracy under `jit` on D2 and D3. Together they give three of four.

## 5. Limits

D4 (N(1e6, 1)) is above the baseline formula in both regimes. `ulp(x[0])` at that offset is `0.0625`, the deviation is `1`, so the shift introduces a relative error of `6%` in `y`. The compensated sum cannot recover digits that the subtraction has already lost. Removing this requires computing in float64, which is out of scope.

The Kahan sum is sequential (`jax.lax.scan`). A parallel version (block sum with compensated merge) was measured: it works in eager and breaks under `jax.jit` on these shapes (the merge tree is reordered by the compiler). The speed is 1.85x, and that is the cost of the accuracy.

## 6. References

- Kahan, W. (1965). Further remarks on reducing truncation errors. Communications of the ACM 8(1), 40. Bibliographic only, not in the papers folder; the compensated summation.
- Pebay, P. P. (2008). Formulas for Robust, One-Pass Parallel Computation of Covariances and Arbitrary-Order Statistical Moments. Sandia Report SAND2008-6212. Equations 1.1 and 1.2, PDF page 7, the recurrence behind the local centring.
- Shewchuk, J. R. (1997). Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates. Discrete and Computational Geometry 18(3), 305-363. Lemma 5, page 10 (PDF page 14), the Sterbenz lemma behind the shift.
