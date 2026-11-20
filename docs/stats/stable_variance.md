# Stable variance in float32

`stable_var` measures how much a list of numbers wiggles when every number is huge and the wiggle is tiny, and it does not get fooled by the huge part.

Why does it matter? A computer that works in `float32` keeps only about 7 digits of each number. A number like 4500.25 has no room left for a wiggle of a few cents: the usual formula loses the wiggle, as if it had never been there. `stable_var` first takes away the big base number and measures only the wiggle.

It is useful wherever there is a big number with small wiggles around it, in any field:

- **Finance:** a stock index quoted at about 4,500 euros that moves in steps of 25 cents (0.25). The usual formula in `float32` says the variation is `0.0000`, as if the market were standing still. `stable_var` says `0.2528`, which is the right value.
- **Robots and drones:** a joint resting around 1,000 that vibrates by 0.01. The usual formula is wrong by about 630 times. `stable_var` gives the right number.
- **Deep neural networks in `float32`:** a signal with a big base value that goes through many layers, one after the other. With the usual layer the gradient is wrong by up to about 20 out of 100 after 1024 layers; with `stable_var` it stays below 1 in 10,000 in the same test.

So it is not only for deep networks: it works in finance, in robots and in deep networks every time you have "big numbers plus small wiggles" in `float32`.

How do I compute a variance in float32 that is exactly zero on identical
values?

![How stable_var works in four three-dimensional panels: 1, quotes of an index near 4500 for which the usual float32 formula gives a variance of 0.0000 while stable_var gives 0.2528; 2, the shift: the error of the squares of the quotes is bigger than the variance, after the shift it is zero; 3, the two-sum: a sum that keeps the error it would lose; 4, the gradient of a stack of layers on a GPU: the usual layer is wrong by up to 21 percent, the layer with stable_var by less than 0.006 percent.](../assets/stable_variance/stable_var_explained.png)

The picture is the depth test on a GPU in float32, for 11 depths from 1 to 1024 layers and 6 values of the offset over the spread, from 800 to 250000 (66 cells). The pink surface is the gradient error of the usual layer, the blue one is the error of the layer with `stable_var`. The usual layer degrades with the depth and with the offset over the spread, and is at about 20 % at 1024 layers and 250000; the layer with `stable_var` is better in all 66 cells, by a factor between 35 and 17000 (median 960), and goes from 3e-07 to 6e-05. A step with `stable_var` takes between 0.76 and 3.1 times as long as one with the usual layer (median 1.4).

## A deep learning example: the gradient of a normalisation layer

A normalisation layer divides each row of activations by the square root of its variance. When the activations of a row sit around a large offset with a small spread, the variance and the subtraction of the mean in `float32` lose the digits of the spread, and the gradient that goes back to the earlier layers inherits the error. The test takes 64 rows of 1024 values around 1000 with a spread of 0.004 and applies three layers: the usual one, with `jnp.mean` and `jnp.var`; one that takes only the variance from `stable_var`; and one that takes the variance from `stable_var` and subtracts the mean on the data shifted by the first element, `(x - x[0]) - mean(x - x[0])`. The output and the gradient (by `jax.grad`) of each are compared with the same layer in float64:

```python
import jax
import jax.numpy as jnp
import numpy as np
from dense_armor.core.signals.stats.stable import stable_var

EPS = 1e-6


def usual(x):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + EPS)


def stable_variance_only(x):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = stable_var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + EPS)


def stable(x):
    y = x - x[..., :1]
    d = y - jnp.mean(y, axis=-1, keepdims=True)
    return d / jnp.sqrt(stable_var(x, axis=-1, keepdims=True) + EPS)


rng = np.random.default_rng(0)
x = rng.normal(1000.0, 0.004, (64, 1024)).astype(np.float32)
g = rng.normal(0.0, 1.0, (64, 1024)).astype(np.float32)
x64, g64 = x.astype(np.float64), g.astype(np.float64)
inv = 1.0 / np.sqrt(x64.var(-1, keepdims=True) + EPS)
xh = (x64 - x64.mean(-1, keepdims=True)) * inv
dx = inv * (g64 - g64.mean(-1, keepdims=True) - xh * (g64 * xh).mean(-1, keepdims=True))
for name, layer in (
    ("usual", usual),
    ("stable variance only", stable_variance_only),
    ("stable", stable),
):
    out = np.asarray(layer(jnp.asarray(x)), np.float64)
    grad = np.asarray(jax.grad(lambda a: jnp.sum(g * layer(a)))(jnp.asarray(x)), np.float64)
    print(f"{name:22s} output {np.abs(out - xh).max():.1e}  gradient {np.abs(grad - dx).max():.1e}")
print(f"largest true gradient {np.abs(dx).max():.0f}")
```

```
usual                  output 1.9e-02  gradient 8.8e-01
stable variance only   output 1.9e-02  gradient 7.8e-01
stable                 output 3.5e-07  gradient 5.1e-03
largest true gradient 1122
```

The usual layer is wrong in the output by `1.9e-02` and in the gradient by `0.88`, on a largest true gradient of 1122. Taking only the variance from `stable_var` changes nothing (`1.9e-02` and `0.78`): the digits are lost in the subtraction `x - mean`, not in the variance. When the layer also subtracts the mean on the shifted data, the output error falls to `3.5e-07` and the gradient error to `5.1e-03`, that is `4.5e-06` of the largest gradient against `7.8e-04` for the usual layer: about 170 times smaller. `stable_var` can be differentiated by `jax.grad`, and no gradient in this test is `NaN`.

This is one layer, in one setting, in `float32`; it is not a training run, and the page does not measure the accuracy of a trained network.

### How many layers

The same layer was stacked, `L` times: each layer is `x = 1000 + 0.004 * tanh(LN(x) @ W)`, with `W` orthogonal (times 0.7), 128 rows of 512 values, so that every layer receives an offset of 1000 and a spread of about 0.004. The test is `benchmarks/stable_variance_depth.py`, run on a Kaggle GPU with JAX 0.11.1 (the matrix products at the highest precision, one of the two visible devices in use). For each depth the gradient of the output with respect to the input is computed in `float32` by `jax.grad` and, as a reference, in float64 by replaying the backward pass on the same `float32` inputs. The table gives the relative error of the gradient and the time of one forward and backward step (median of 11 runs):

![The relative error of the gradient and the time of one forward and backward step against the number of layers, for the usual layer normalisation and for the one with stable_var: the usual layer is at 0.4 % with one layer, above 1 % from 4 layers and at 20 % with 1024; with stable_var the error goes from 3e-07 to 6e-05; the step takes between 1.2 and 3 times as long.](../assets/stable_variance/depth_gradient.png)

| layers | gradient error, usual layer | gradient error, `stable_var` | step time, usual | step time, `stable_var` | time ratio |
|---|---|---|---|---|---|
| 1 | 3.66e-03 | 3.45e-07 | 0.31 ms | 0.37 ms | 1.20 |
| 2 | 7.87e-03 | 4.88e-07 | 0.57 ms | 0.74 ms | 1.29 |
| 4 | 1.11e-02 | 6.95e-07 | 0.96 ms | 1.37 ms | 1.43 |
| 8 | 1.77e-02 | 1.04e-06 | 1.71 ms | 2.46 ms | 1.44 |
| 16 | 2.54e-02 | 1.62e-06 | 3.15 ms | 4.56 ms | 1.45 |
| 32 | 3.74e-02 | 2.65e-06 | 4.36 ms | 8.83 ms | 2.03 |
| 64 | 5.35e-02 | 4.54e-06 | 8.05 ms | 17.42 ms | 2.17 |
| 128 | 7.58e-02 | 8.33e-06 | 11.71 ms | 34.56 ms | 2.95 |
| 256 | 1.06e-01 | 1.56e-05 | 23.34 ms | 35.78 ms | 1.53 |
| 512 | 1.50e-01 | 3.06e-05 | 47.30 ms | 63.16 ms | 1.34 |
| 1024 | 2.04e-01 | 5.91e-05 | 95.30 ms | 127.99 ms | 1.34 |

With the usual layer the gradient is wrong by 0.37 % in a single layer, by 1.1 % at 4 layers and by 20 % at 1024. With `stable_var` it is wrong by 3.5e-07 in a single layer and by 5.9e-05 at 1024: between 3450 and 16900 times smaller at every depth tested. The largest error of the output of one normalisation, against float64 on the same input, is between 1.9e-02 and 5.3e-02 for the usual layer and 4.2e-07 for the layer with `stable_var`, at every depth. No gradient is `NaN` or infinite in either variant.

If a gradient error of 1 % is the most that can be accepted, the usual layer reaches it at 4 layers (1.1 %; at 2 layers it is 0.79 %); with `stable_var` the error is still 5.9e-05 at 1024 layers, the deepest tested. So the number of layers that can be built is at least 256 times larger, 1024 against 4, and this test did not find the limit with `stable_var`.

The price is time. A step with the layer that uses `stable_var` takes between 1.2 and 3.0 times as long as one with the usual layer on this GPU, and 1.34 times at 1024 layers (128.0 ms against 95.3 ms). The ratio at 128 layers (2.95) is out of line with its neighbours (1.53 at 256), and was not investigated. For activations with a large offset and a small spread, where the usual layer gives gradients that are wrong by 1 % to 20 %, a step that takes at most three times as long and a gradient that is 3450 to 16900 times more accurate is a good price; for activations without a large offset this test says nothing, because it was not run.

This is one synthetic stack, one GPU session and one size (128 rows of 512 values). In this stack the gradient grows with the depth, up to about 1e15 at 1024 layers, and stays finite in `float32`; it is not a trained network, and the page does not measure the accuracy of one.

## Gamestory

Saturday afternoon at the market. The young shopkeeper and his assistant are selling a super-rare series of collectible action figures that all cost around 1,000 euros (one costs 1000.00, one 1000.01, one 1000.02).

At the end of the day they want to know how much those few cents varied from one figure to another. But there is a problem: to do the math super fast, the assistant uses a pocket calculator with a short screen (called float32), which can hold at most 7 digits on display!

If the assistant uses the normal textbook formula (mean(x**2) - mean(x)**2), a disaster happens: squaring 1000 creates giant numbers (1,000,000), the short screen cuts off all the cents to fit the zeros, and at the end it spits out a crazy negative number (-196608)! To save the cents from the short screen, the assistant uses the stable_var trick, made of four clever moves:

- **1. Removing the giant weight at the start (y = x - x[0]):** Before doing any math, the assistant takes the price of the first figure (1000.00) and subtracts it from all the others. Only the cents of difference remain on the sheet (0.00, 0.01, 0.02)! That way the short screen no longer wastes space writing the thousand euros that are the same for everyone.
- **2. Centering the cents (m):** Now he takes the average of only those tiny cents (m): this way every tiny rounding stays super small and never blows up.
- **3. Catching the crumbs that fall off (TwoSum and kahan):** When adding up lots of tiny cents one after another, on a short-screen calculator some crumbs risk falling off the edge. The assistant uses a special trick (TwoSum and kahan) that catches every single crumb of remainder and puts it back into the final sum, without losing even a thousandth!
- **4. Exact zero on identical prices (0.0):** In the end, if all the figures had the exact same price ([1e10, 1e10, 1e10]), the stable_var function answers 0.0 exact, without making up fake wobbles.

In the end, what is stable_var (Stable variance in float32)? It is a super-accurate math function that lets computers calculate how much numbers vary while using very little memory (float32), by first removing the giant base value and saving even the tiniest rounding crumbs!

## Description

Actually, `stable_var` is useful wherever there is a large number with small variations around it, regardless of the field:

- **In finance:** a stock or an index quoted at 4,500 euros that fluctuates by 25 cents (0.25). With the normal formula in `float32`, the variance comes out `0.0000` (as if the market were completely still!), while `stable_var` finds the exact value (`0.2528`).
- **In robots and drones:** a joint at rest around 1,000 that vibrates by 0.01. The normal formula is wrong by 630 times, while `stable_var` gives the exact number.
- **In deep networks in `float32`:** when a signal with a large base value goes through many layers, one after another.

So it is not only for deep networks: it holds in finance, in robots and in deep networks every time you have "large numbers + small variations" in `float32`!

How do I compute a variance in float32 that is exactly zero on identical
values?

![A three-dimensional surface: the number of layers (log2) and the offset over the spread of the activations (log10) on the horizontal axes, the relative error of the gradient (log10) on the vertical axis. The pink surface of the usual layer normalisation rises with the depth and with the offset over the spread, up to about 20 percent; the blue surface of the layer with stable_var stays between 3e-07 and 6e-05.](../assets/stable_variance/gradient_3d.png)

The picture is the depth test on a GPU in float32, for 11 depths from 1 to 1024 layers and 6 values of the offset over the spread, from 800 to 250000 (66 cells). The pink surface is the gradient error of the usual layer, the blue one is the error of the layer with `stable_var`. The usual layer degrades with the depth and with the offset over the spread, and is at about 20 % at 1024 layers and 250000; the layer with `stable_var` is better in all 66 cells, by a factor between 35 and 17000 (median 960), and goes from 3e-07 to 6e-05. A step with `stable_var` takes between 0.76 and 3.1 times as long as one with the usual layer (median 1.4).

## A deep learning example: the gradient of a normalisation layer

A normalisation layer divides each row of activations by the square root of its variance. When the activations of a row sit around a large offset with a small spread, the variance and the subtraction of the mean in `float32` lose the digits of the spread, and the gradient that goes back to the earlier layers inherits the error. The test takes 64 rows of 1024 values around 1000 with a spread of 0.004 and applies three layers: the usual one, with `jnp.mean` and `jnp.var`; one that takes only the variance from `stable_var`; and one that takes the variance from `stable_var` and subtracts the mean on the data shifted by the first element, `(x - x[0]) - mean(x - x[0])`. The output and the gradient (by `jax.grad`) of each are compared with the same layer in float64:

```python
import jax
import jax.numpy as jnp
import numpy as np
from dense_armor.core.signals.stats.stable import stable_var

EPS = 1e-6


def usual(x):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + EPS)


def stable_variance_only(x):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = stable_var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + EPS)


def stable(x):
    y = x - x[..., :1]
    d = y - jnp.mean(y, axis=-1, keepdims=True)
    return d / jnp.sqrt(stable_var(x, axis=-1, keepdims=True) + EPS)


rng = np.random.default_rng(0)
x = rng.normal(1000.0, 0.004, (64, 1024)).astype(np.float32)
g = rng.normal(0.0, 1.0, (64, 1024)).astype(np.float32)
x64, g64 = x.astype(np.float64), g.astype(np.float64)
inv = 1.0 / np.sqrt(x64.var(-1, keepdims=True) + EPS)
xh = (x64 - x64.mean(-1, keepdims=True)) * inv
dx = inv * (g64 - g64.mean(-1, keepdims=True) - xh * (g64 * xh).mean(-1, keepdims=True))
for name, layer in (
    ("usual", usual),
    ("stable variance only", stable_variance_only),
    ("stable", stable),
):
    out = np.asarray(layer(jnp.asarray(x)), np.float64)
    grad = np.asarray(jax.grad(lambda a: jnp.sum(g * layer(a)))(jnp.asarray(x)), np.float64)
    print(f"{name:22s} output {np.abs(out - xh).max():.1e}  gradient {np.abs(grad - dx).max():.1e}")
print(f"largest true gradient {np.abs(dx).max():.0f}")
```

```
usual                  output 1.9e-02  gradient 8.8e-01
stable variance only   output 1.9e-02  gradient 7.8e-01
stable                 output 3.5e-07  gradient 5.1e-03
largest true gradient 1122
```

The usual layer is wrong in the output by `1.9e-02` and in the gradient by `0.88`, on a largest true gradient of 1122. Taking only the variance from `stable_var` changes nothing (`1.9e-02` and `0.78`): the digits are lost in the subtraction `x - mean`, not in the variance. When the layer also subtracts the mean on the shifted data, the output error falls to `3.5e-07` and the gradient error to `5.1e-03`, that is `4.5e-06` of the largest gradient against `7.8e-04` for the usual layer: about 170 times smaller. `stable_var` can be differentiated by `jax.grad`, and no gradient in this test is `NaN`.

This is one layer, in one setting, in `float32`; it is not a training run, and the page does not measure the accuracy of a trained network.

### How many layers

The same layer was stacked, `L` times: each layer is `x = 1000 + 0.004 * tanh(LN(x) @ W)`, with `W` orthogonal (times 0.7), 128 rows of 512 values, so that every layer receives an offset of 1000 and a spread of about 0.004. The test is `benchmarks/stable_variance_depth.py`, run on a Kaggle GPU with JAX 0.11.1 (the matrix products at the highest precision, one of the two visible devices in use). For each depth the gradient of the output with respect to the input is computed in `float32` by `jax.grad` and, as a reference, in float64 by replaying the backward pass on the same `float32` inputs. The table gives the relative error of the gradient and the time of one forward and backward step (median of 11 runs):

![The relative error of the gradient and the time of one forward and backward step against the number of layers, for the usual layer normalisation and for the one with stable_var: the usual layer is at 0.4 % with one layer, above 1 % from 4 layers and at 20 % with 1024; with stable_var the error goes from 3e-07 to 6e-05; the step takes between 1.2 and 3 times as long.](../assets/stable_variance/depth_gradient.png)

| layers | gradient error, usual layer | gradient error, `stable_var` | step time, usual | step time, `stable_var` | time ratio |
|---|---|---|---|---|---|
| 1 | 3.66e-03 | 3.45e-07 | 0.31 ms | 0.37 ms | 1.20 |
| 2 | 7.87e-03 | 4.88e-07 | 0.57 ms | 0.74 ms | 1.29 |
| 4 | 1.11e-02 | 6.95e-07 | 0.96 ms | 1.37 ms | 1.43 |
| 8 | 1.77e-02 | 1.04e-06 | 1.71 ms | 2.46 ms | 1.44 |
| 16 | 2.54e-02 | 1.62e-06 | 3.15 ms | 4.56 ms | 1.45 |
| 32 | 3.74e-02 | 2.65e-06 | 4.36 ms | 8.83 ms | 2.03 |
| 64 | 5.35e-02 | 4.54e-06 | 8.05 ms | 17.42 ms | 2.17 |
| 128 | 7.58e-02 | 8.33e-06 | 11.71 ms | 34.56 ms | 2.95 |
| 256 | 1.06e-01 | 1.56e-05 | 23.34 ms | 35.78 ms | 1.53 |
| 512 | 1.50e-01 | 3.06e-05 | 47.30 ms | 63.16 ms | 1.34 |
| 1024 | 2.04e-01 | 5.91e-05 | 95.30 ms | 127.99 ms | 1.34 |

With the usual layer the gradient is wrong by 0.37 % in a single layer, by 1.1 % at 4 layers and by 20 % at 1024. With `stable_var` it is wrong by 3.5e-07 in a single layer and by 5.9e-05 at 1024: between 3450 and 16900 times smaller at every depth tested. The largest error of the output of one normalisation, against float64 on the same input, is between 1.9e-02 and 5.3e-02 for the usual layer and 4.2e-07 for the layer with `stable_var`, at every depth. No gradient is `NaN` or infinite in either variant.

If a gradient error of 1 % is the most that can be accepted, the usual layer reaches it at 4 layers (1.1 %; at 2 layers it is 0.79 %); with `stable_var` the error is still 5.9e-05 at 1024 layers, the deepest tested. So the number of layers that can be built is at least 256 times larger, 1024 against 4, and this test did not find the limit with `stable_var`.

The price is time. A step with the layer that uses `stable_var` takes between 1.2 and 3.0 times as long as one with the usual layer on this GPU, and 1.34 times at 1024 layers (128.0 ms against 95.3 ms). The ratio at 128 layers (2.95) is out of line with its neighbours (1.53 at 256), and was not investigated. For activations with a large offset and a small spread, where the usual layer gives gradients that are wrong by 1 % to 20 %, a step that takes at most three times as long and a gradient that is 3450 to 16900 times more accurate is a good price; for activations without a large offset this test says nothing, because it was not run.

This is one synthetic stack, one GPU session and one size (128 rows of 512 values). In this stack the gradient grows with the depth, up to about 1e15 at 1024 layers, and stays finite in `float32`; it is not a trained network, and the page does not measure the accuracy of one.

## Gamestory

Saturday afternoon at the market. The young shopkeeper and his assistant are selling a super-rare series of collectible action figures that all cost around 1,000 euros (one costs 1000.00, one 1000.01, one 1000.02).

At the end of the day they want to know how much those few cents varied from one figure to another. But there is a problem: to do the math super fast, the assistant uses a pocket calculator with a short screen (called float32), which can hold at most 7 digits on display!

If the assistant uses the normal textbook formula (mean(x**2) - mean(x)**2), a disaster happens: squaring 1000 creates giant numbers (1,000,000), the short screen cuts off all the cents to fit the zeros, and at the end it spits out a crazy negative number (-196608)! To save the cents from the short screen, the assistant uses the stable_var trick, made of four clever moves:

- **1. Removing the giant weight at the start (y = x - x[0]):** Before doing any math, the assistant takes the price of the first figure (1000.00) and subtracts it from all the others. Only the cents of difference remain on the sheet (0.00, 0.01, 0.02)! That way the short screen no longer wastes space writing the thousand euros that are the same for everyone.
- **2. Centering the cents (m):** Now he takes the average of only those tiny cents (m): this way every tiny rounding stays super small and never blows up.
- **3. Catching the crumbs that fall off (TwoSum and kahan):** When adding up lots of tiny cents one after another, on a short-screen calculator some crumbs risk falling off the edge. The assistant uses a special trick (TwoSum and kahan) that catches every single crumb of remainder and puts it back into the final sum, without losing even a thousandth!
- **4. Exact zero on identical prices (0.0):** In the end, if all the figures had the exact same price ([1e10, 1e10, 1e10]), the stable_var function answers 0.0 exact, without making up fake wobbles.

In the end, what is stable_var (Stable variance in float32)? It is a super-accurate math function that lets computers calculate how much numbers vary while using very little memory (float32), by first removing the giant base value and saving even the tiniest rounding crumbs!

## A first example: the quotes of a stock index

A stock index future is quoted around 4500 euros and moves in steps of 0.25, the tick. A trading desk takes 3000 consecutive quotes, in `float32`, and wants their variance, the first step of a volatility estimate. The first number is the variance in float64 of the same `float32` values (the reference), the second is the textbook formula in `float32`, the third is `stable_var`:

```python
import numpy as np
import jax.numpy as jnp
from dense_armor.core.signals.stats.stable import stable_var

rng = np.random.default_rng(0)
quotes = np.round(rng.normal(4500.0, 0.5, 3000) * 4.0) / 4.0
x = jnp.asarray(quotes.astype(np.float32))
ref = float(np.var(np.asarray(x, dtype=np.float64)))
naive = float(jnp.mean(x * x) - jnp.mean(x) ** 2)
print(f"{ref:.4f} {naive:.4f} {float(stable_var(x)):.4f}")
```

```
0.2528 0.0000 0.2528
```

The textbook formula returns `0.0000`: the quotes are around 4500, their squares are of the order of twenty million, and the variance of a quarter of a euro is below the last digits of that number, so it disappears in the subtraction. A volatility of zero on a moving market is the kind of answer that nothing in the number says is wrong. `stable_var` returns `0.2528`, equal to the reference.

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
from dense_armor.core.signals.stats.stable import stable_var

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
from dense_armor.core.signals.stats.moments import RunningMoments

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

## 6. A robot joint around its home position

A robot joint reports its angle as a `float32` every millisecond. The angle sits around a large offset, the wobble around it is small, and the controller needs the variance of the wobble. Three thousand readings around 1000 with a spread of 0.01, in the same three numbers as above:

```python
import numpy as np
import jax.numpy as jnp
from dense_armor.core.signals.stats.stable import stable_var

rng = np.random.default_rng(0)
x = jnp.asarray(rng.normal(1000.0, 0.01, 3000).astype(np.float32))
ref = float(np.var(np.asarray(x, dtype=np.float64)))
naive = float(jnp.mean(x * x) - jnp.mean(x) ** 2)
print(f"{ref:.4e} {naive:.4e} {float(stable_var(x)):.4e}")
```

```
9.8841e-05 6.2500e-02 9.8841e-05
```

The textbook formula gives `6.2500e-02`, about 630 times the true variance `9.8841e-05`, and the sign of the error is positive, so nothing in the number says that it is wrong. `stable_var` returns `9.8841e-05`, the same as the reference at the digits shown.

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
`dense_armor/core/signals/stats` is described in the CHANGELOG;
`RunningMoments.learn_many` uses `stable_moments` for a `float32` block.

### The figure at the top

The image `docs/assets/stable_variance/gradient_3d.png` is generated by `docs/assets/stable_variance/make_gradient_3d.py` from `benchmarks/stable_variance_depth_grid_gpu.json`, the result of `python benchmarks/stable_variance_depth.py --grid` on a Kaggle GPU: 6 spreads (0.004 to 1.26) around an offset of 1000 and 11 depths, the same stack as in the depth test below, with 3 timed runs per cell.

### The depth test

`benchmarks/stable_variance_depth.py` writes `benchmarks/stable_variance_depth.json`; the GPU numbers on this page are in `benchmarks/stable_variance_depth_gpu.json`, obtained with the same script, with `stable.py` and the script unchanged, in a Kaggle session with a GPU. The image `docs/assets/stable_variance/depth_gradient.png` is made from that file by `docs/assets/stable_variance/make_depth_figure.py`.
