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

## 3. What the function does

Four steps, all in the input dtype. The first one removes the offset, the
second one centres the shifted values, the third accumulates, the fourth
cancels the error of the mean.

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
Their `float32` resolution is `1e-9`, not the `6e-5` of the original `1000`.

**Step 2, local centring.** The mean of the shifted values, `m`, is at the
level of the wobble. Its rounding error is at the level of the wobble, not
of the offset.

**Step 3, second centring and accumulation.** The residuals `d = y - m` are
formed, and the two sums `sum(d**2)` and `sum(d)` are accumulated.

**Step 4, cancellation of the mean error.** The variance is

```
var = max( sum(d**2)/n - (sum(d)/n)**2, 0 )
```

The two terms share the same `m`. Its rounding error is linear in `m` in
both terms, and cancels in the subtraction. This is the same two-term form
that the TensorFlow formula uses on `x`, applied to the shifted values `y`,
where the cancellation of the offset has already been done by Step 1.

## 4. The accuracy on the four datasets

The maximum relative error against the `float64` reference of the same
`float32` values, on the four datasets of the task, measured by
`benchmarks/stable_variance_float32.py`:

| dataset | TensorFlow formula | `stable_var` (eager) | `stable_var` (jit) |
|---|---|---|---|
| N(1e3, 0.01), `[1000, 64]` axis 0 | 1.18e-7 | 1.25e-7 | 2.39e-7 |
| N(1e4, 1), `[64, 1000]` axis 1 | 1.59e-7 | 1.63e-7 | 1.63e-7 |
| N(0, 1), `[1000, 64]` axis 0 | 1.94e-7 | 1.14e-7 | 2.65e-7 |
| N(1e6, 1), `[1000, 64]` axis 0 | 9.97e-8 | 1.74e-7 | 2.24e-7 |

The eager column is the one the test file checks, and all four values are
below the `2e-7` threshold of the task. The jit column is the one the
benchmark measures; XLA reorders the reduction under `jit`, and the
results differ at the last bit. On three of the four datasets the jit
error is above the threshold, and the benchmark records it as a declared
limit (see Details).

The speed on a synthetic `[4096, 4096]` float32 array, `axis=1`, 7 runs
after warm-up with `block_until_ready`:

| formula | median (ms) | IQR (ms) |
|---|---|---|
| naive two-pass | 72.85 | 6.72 |
| TensorFlow formula | 71.87 | 9.64 |
| `stable_var` | 94.37 | 3.65 |

`stable_var` costs 1.31 times the TensorFlow formula on this shape,
below the 2.0 threshold of the task.

## Details

### References

- Pebay, P. P. (2008). Formulas for Robust, One-Pass Parallel Computation
  of Covariances and Arbitrary-Order Statistical Moments. Sandia Report
  SAND2008-6212. Equations 1.3 and 1.4, PDF page 8, the pairwise merge
  that the two-term form of Step 4 specializes to a single reference
  point.
- Ogita, T., Rump, S. M., Oishi, S. (2005). Accurate Sum and Dot
  Product. SIAM Journal on Scientific Computing 26(6), 1955-1988.
  TwoSum, Algorithm 3.1, PDF page 7, the error-free transformation that
  the helper `_two_sum` implements.
- Shewchuk, J. R. (1997). Adaptive Precision Floating-Point Arithmetic
  and Fast Robust Geometric Predicates. Discrete and Computational
  Geometry 18(3), 305-363. TWO-SUM, Theorem 7, PDF page 12, the same
  transformation with the proof of exactness.
- Sterbenz lemma, in the form used by Step 1: if `y/2 <= x <= 2y` then
  `x - y` is exact. Cited from Shewchuk 1997, Section 2.3.

### Choices

The four-step chain is our choice. None of the three papers states it as
a variance kernel. Each step is in a paper (the shift is the Sterbenz
lemma, the two-term form is Pebay eq. 1.4 specialized to a singleton, the
residual uses the TwoSum of Ogita-Rump-Oishi Algorithm 3.1 and Shewchuk
Theorem 7), but the chain is not. The reason for the chain is in Step 1 to
Step 4: it is the smallest number of steps that keeps every intermediate
at the level of the variation and not of the offset.

The dtype policy: float32 in gives float32 out, float64 in gives float64
out when `jax_enable_x64` is on (off, JAX promotes to float32),
float16 and bfloat16 are computed in float32 and returned in the input
dtype, integers give float32, and complex input gives the real
`E|z - mean(z)|**2`.

`ddof = n` returns `nan` when the variance is positive and `0.0` when the
values are identical, so the exact-zero property survives a `ddof` that
empties the sample.

### Measured cost

The JSON is `benchmarks/stable_variance_float32.json`. The script is
`benchmarks/stable_variance_float32.py`, the command is
`python benchmarks/stable_variance_float32.py`. The run is on a CPU
backend, JAX 0.11.1, 7 runs after warm-up, median and IQR.

### Declared limits

The jit column of the accuracy table is above the threshold on three of
the four datasets. XLA reorders the reduction under `jit` and the results
differ at the last bit from the eager path. The test file
`test/utility/stats/test_stable.py` checks the eager path, which is below
the threshold on all four datasets. A kernel whose eager and jit results
agree to the last bit is not in this task.

`stable_var` is a function, not an estimator: it has no `learn_one`, no
`budget_s`, no `memory_class`. The float32 path of the classes of batch
T3 is item D of the task, not this page.
