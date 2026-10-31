# Stable variance in float32

How do I compute a variance in float32 that is exactly zero on identical
values?

A robot joint reports its angle as a `float32` every millisecond. The
angle sits around a large offset (the home position), the wobble around it
is small, and the controller needs the variance of the wobble. The
textbook formula subtracts two almost equal numbers, and the digits of the
wobble are lost in the subtraction. This page shows the failure, the
function to call, and what the function does inside.

## 1. The failure

The variance of a sample is the average of the squared distances from the
mean:

```
Var(x) = (1/n) * sum_i (x_i - mean(x))^2
```

On a stream of 3000 samples with a large offset and a small wobble, the
naive one-pass formula `mean(x**2) - mean(x)**2` subtracts two numbers of
order `mean(x)**2` to get a variance of order `wobble**2`, and the float32
mantissa is not wide enough.

```python
import numpy as np
import jax.numpy as jnp

rng = np.random.default_rng(0)
x = jnp.asarray(rng.normal(1e6, 1.0, 3000).astype(np.float32))
ref = float(np.var(np.asarray(x, dtype=np.float64)))
naive = float(jnp.mean(x * x) - jnp.mean(x) ** 2)
print(f"reference {ref:.6f}  naive {naive:.6f}")
```

```
reference 0.989742  naive -196608.000000
```

The naive formula returns a negative number of order `1e5` where the
reference is `0.989742`: the two terms `mean(x**2)` and `mean(x)**2` are of
order `1e12`, and their rounding errors are of order `1e5`, a million times
the wobble. The `float32` mantissa cannot hold the difference.

## 2. The function to call

`stable_var` returns the variance in the input dtype, with the same
`axis`, `ddof` and `keepdims` arguments as the other reducers of the
library, and is exactly `0.0` on identical values.

```python
import jax
import jax.numpy as jnp
from dense_armor.utility.stats.stable import stable_var

x = jnp.asarray([1e10, 1e10, 1e10], dtype=jnp.float32)
print(float(stable_var(x)))

rng = jax.random.PRNGKey(0)
z = jax.random.normal(rng, (3000,), dtype=jnp.float32) * 0.01 + 1000.0
v = jax.jit(lambda a: stable_var(a))(z)
print(f"{float(v):.8f}")
```

```
0.0
0.00010181
```

The first call returns the bit pattern `0x00000000`, not a small number.
The second call runs under `jax.jit` and stays in `float32` from the input
to the output.

The same path is available on the estimators, through the `dtype` keyword
of `RunningMoments`:

```python
import numpy as np
from dense_armor.utility.stats.moments import RunningMoments

rng = np.random.default_rng(0)
block = rng.normal(1000.0, 0.01, 500).astype(np.float32)
m = RunningMoments(dtype=np.float32)
m.learn_many(block)
ref = float(np.var(block.astype(np.float64), ddof=1))
print(f"{m.var:.6e}  {ref:.6e}  {abs(m.var - ref) / ref:.2e}")
```

```
1.029553e-04  1.029553e-04  4.91e-09
```

The float32 state agrees with the float64 reference to `5e-9`, well below
the float32 rounding unit `1.19e-7`. The default `dtype=None` keeps the
float64 result, bit for bit.

## 3. What the function does

Four steps, all in the input dtype. The first one removes the offset, the
second one centres the shifted values, the third sums their squares with a
compensated tree, the fourth applies the two-term correction and divides.

**Step 1, shift.** Subtract the first element along the reduction axis.
When the values are within a factor of two of it, the subtraction is exact
(Sterbenz), and the large common exponent is gone.

```python
import jax.numpy as jnp

x = jnp.asarray([1000.0, 1000.01, 1000.02], dtype=jnp.float32)
y = x - x[0]
print(y)
```

```
[0.         0.01000977 0.02001953]
```

The shifted values are at the exponent of the wobble, not of the offset.

**Step 2, local centring.** The mean of the shifted values, `m`, is at the
level of the wobble. Its rounding error is at the level of the wobble, not
of the offset.

**Step 3, compensated tree.** The residuals `d = y - m` are formed and the
sum `s2 = sum(d**2)` is accumulated by a binary tree. At each level the
pairs of partial sums are added with `TwoSum` (Ogita, Rump and Oishi 2005,
Algorithm 3.1), which returns the rounded sum and its exact rounding error,
and the errors are carried up the tree. The tree has `log2(n)` vectorised
levels, so it runs on a GPU. The idea of a carry that keeps the low-order
bits is shown below with the sequential summation of Kahan, which is not the
one the kernel uses.

```python
import jax
import jax.numpy as jnp

def seq_sum(x):
    def step(carry, xi):
        return carry + xi, None
    s, _ = jax.lax.scan(step, jnp.zeros((), x.dtype), x)
    return s

def kahan(x):
    def step(carry, xi):
        s, c = carry
        y = xi - c
        t = s + y
        c = (t - s) - y
        return (t, c), None
    (s, c), _ = jax.lax.scan(
        step, (jnp.zeros((), x.dtype), jnp.zeros((), x.dtype)), x)
    return s + c

x = jnp.asarray([1.0] + [1e-7] * 100000, dtype=jnp.float32)
print(f"{float(seq_sum(x)):.8f}  {float(kahan(x)):.8f}")
```

```
1.01192093  1.00999999
```

The exact sum is `1.01000000`. The sequential sum loses the small terms
because each `1e-7` is below the unit in the last place of the running
sum after a few steps (error `1.9e-3`); the compensated sum keeps the
carry and gives `9.5e-9`.

**Step 4, two-term correction and division.** With `s1 = sum(d)`, the biased
variance is `max(s2/n - (s1/n)**2, 0)`, multiplied by `n / (n - ddof)`. The
mean is `x[0] + m`.

## 4. The accuracy on the four datasets

The maximum relative error against the `float64` reference of the same
`float32` values, on the four datasets, measured by
`benchmarks/stable_variance_float32.py`. In this page the baseline formula is
the formula of `tf.nn.moments` in TensorFlow, `mean(d**2) - mean(d)**2` with
`d = x - mean(x)`, which has no shift:

| dataset | baseline formula (eager) | `stable_var` (eager) | baseline formula (jit) | `stable_var` (jit) |
|---|---|---|---|---|
| N(1e3, 0.01), `[1000, 64]` axis 0 | 1.44e-7 | 1.17e-7 | 1.18e-7 | 1.17e-7 |
| N(1e4, 1), `[64, 1000]` axis 1 | 1.59e-7 | 1.15e-7 | 1.59e-7 | 1.15e-7 |
| N(0, 1), `[1000, 64]` axis 0 | 1.94e-7 | 1.36e-7 | 1.94e-7 | 1.36e-7 |
| N(1e6, 1), `[1000, 64]` axis 0 | 1.18e-7 | 1.22e-7 | 0.997e-7 | 1.22e-7 |

The kernel is at most equal to the baseline formula on three or four of the
four datasets in each combination (CPU eager, CPU jit, GPU eager, GPU jit).
The exceptions are D4 on the CPU, in eager and under `jit`, and D3 on the GPU
under `jit`. The cause is not explained. The CPU numbers are in
`stable_variance_float32.json`, the GPU numbers in
`stable_variance_float32_gpu.json`.

## 5. The speed

Measured by the same script on a synthetic `[4096, 4096]` `float32` array,
`axis=1`, under `jax.jit`, in three sessions, with the median and the range
of the ratio to the baseline formula. `stable_var` costs 3.59 times the
baseline formula on the CPU (range 3.55 to 4.60) and 2.24 times on the GPU
(range 2.20 to 2.27). The first target was 2.0 times; it was not reached,
and the threshold of the benchmark, 4.0 on the median ratio, was chosen
after the measurement.

## Details

### References

- Ogita, T., Rump, S. M., Oishi, S. (2005). Accurate Sum and Dot Product.
  SIAM Journal on Scientific Computing 26(6), 1955-1988. Algorithm 3.1
  (`TwoSum`), PDF page 7.
- Shewchuk, J. R. (1997). Adaptive Precision Floating-Point Arithmetic
  and Fast Robust Geometric Predicates. Discrete and Computational
  Geometry 18(3), 305-363. Theorem 7 (`Two-Sum`), PDF page 12; Lemma 5
  (Sterbenz), PDF page 10, which makes the shift exact.
- Healy, J. (2022). Accurate and consistent calculation of the mean and
  variance in Monte-Carlo simulations. arXiv:2206.10662v7. Algorithm 2: the
  shifted naive algorithm with K = x[0] and Kahan summation.
- Kahan, W. (1965). Further remarks on reducing truncation errors.
  Communications of the ACM 8(1), 40. Bibliographic only, not read: the
  sequential compensated summation of the illustration in step 3.
- TensorFlow, `tensorflow/python/ops/nn_impl.py`, function `moments`, lines
  1480-1487 (master, read on 2026-10-29): the baseline formula.
- Issue #117286 of TensorFlow: the inaccuracy of `tf.nn.moments` in
  `float32`.

### Choices

The shift `y = x - x[0]` is the shifted-data algorithm with K = x[0]. Healy
(2022), Introduction and Algorithm 2, takes the first observation as K, sums
with Kahan's compensation and applies the two-term formula `T/n - (M - K)**2`,
and cites Chan, Golub and LeVeque (1983) for the choice of K; the paper of 1983
was not read. This kernel adds to that algorithm the local centring `d = y - m`
before the squares, the compensated tree of TwoSum in place of the sequential
Kahan sum, and the evaluation one slice at a time with `vmap`. These additions
are by Salvatore Pennacchio.

The dtype policy: float32 in gives float32 out, float64 in gives float64
out when `jax_enable_x64` is on (off, JAX promotes to float32),
float16 and bfloat16 are computed in float32 and returned in the input
dtype, integers give float32, and complex input gives the real
`E|z - mean(z)|**2`.

`ddof = n` returns `nan` when the variance is positive and `0.0` when the
values are identical, so the exact-zero property survives a `ddof` that
empties the sample.

### Measured cost

The JSON files are `benchmarks/stable_variance_float32.json` (CPU) and
`benchmarks/stable_variance_float32_gpu.json` (GPU). The script is
`benchmarks/stable_variance_float32.py`, the command is
`python benchmarks/stable_variance_float32.py`. Each JSON records the
`backend`, the JAX version and the sessions of the speed measurement.

### Declared limits

The accuracy is at most equal to the baseline formula on three or four of
the four datasets in each combination (CPU eager, CPU jit, GPU eager, GPU
jit). It is worse on D4 (N(1e6, 1)) on the CPU: 1.22e-7 against 1.18e-7 in
eager mode and 0.997e-7 under `jit`; and on D3 (N(0, 1)) under `jit` on the
GPU. The cause is not explained.

The kernel is exactly `0.0` on identical values up to `1e38`, in eager mode
and under `jit`, and on `complex64`. The baseline formula gives `nan` at
`1e38`.

The speed is 3.59 times the baseline formula on the CPU and 2.24 times on
the GPU (section 5). Not measured: `float64` (`[1e307]*3`), TPU, other
array sizes.

`stable_var` is a function, not an estimator: it has no `learn_one`, no
`budget_s`, no `memory_class`. The `dtype` keyword of the 11 estimators of
`dense_armor/utility/stats` is described in the CHANGELOG;
`RunningMoments.learn_many` uses `stable_moments` for a `float32` block.
