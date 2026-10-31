# How do I build the blocks of a current Transformer?

A Transformer of the original paper normalises with `LayerNorm`, feeds each
position through a two-matrix network, adds the position to the input and
trains with Adam. Current models change four of these parts: they normalise
with the root mean square only, they gate the feed-forward network, they
rotate the queries and the keys by an angle that grows with the position,
and they train with weight decay that does not pass through the moments of
Adam. This page builds each part, one step at a time, with the classes of
the library.

## 1. Normalise with the root mean square

`RMSNorm` divides each sample by the root mean square of its features and
multiplies by a learned gain. It does not subtract the mean, so it has no
bias.

```python
import jax.numpy as jnp
from dense_armor.utility.neural.norm_rms import RMSNorm

x = jnp.array([[3.0, 4.0], [1.0, 1.0]])
n = RMSNorm(2, eps=1e-12)
print(n.forward(x))
```

```
[[0.84852815 1.1313709 ]
 [1.         1.        ]]
```

The first row has a root mean square of `sqrt((9 + 16) / 2) = 3.5355`, and
`3 / 3.5355 = 0.8485`. The second row is already at root mean square `1`, so
it does not change. The gain starts at `1`; it is the only parameter.

The formula is `y = x / sqrt(mean(x**2) + eps) * g`, where `x` is one input row,
`mean` averages over its features, `eps` is a small constant that keeps the
root away from zero, and `g` is the gain. The small case is the first row
above: `mean(x**2) = (9 + 16) / 2 = 12.5`, its root is `3.5355`, and
`[3, 4] / 3.5355 = [0.8485, 1.1314]`.

## 2. Gate the feed-forward network

A gated feed-forward layer computes two projections of the input, passes one
through an activation, multiplies the two element by element, and projects
back. With the sigmoid it is called GLU, with the GELU GEGLU, with the SiLU
SwiGLU.

```python
import jax
from dense_armor.utility.neural.glu import GLUFeedForward

f = GLUFeedForward(4, 8, "silu", jax.random.PRNGKey(0))
x = jax.random.normal(jax.random.PRNGKey(1), (3, 4))
print(f.forward(x).shape, len(f.params()))
```

```
(3, 4) 3
```

The output has the shape of the input, and the layer has three weight
matrices, as the paper says. The argument `bias=True` adds three bias
vectors; `activation` is `"sigmoid"`, `"gelu"` or `"silu"`.

The formula is `y = (act(x W) * (x V)) W2`, where `x` is one input row of
`d_model` numbers, `W` and `V` are matrices of shape `(d_model, d_ff)`, `W2`
has shape `(d_ff, d_model)`, `act` is the sigmoid, the GELU or the SiLU, and
`*` is the product element by element. A small case: `x = [1, 2]`, `W` the
identity, `V` half the identity, `W2` the identity, and the SiLU. Then
`x W = [1, 2]`, the SiLU gives `[0.7311, 1.7616]`, `x V = [0.5, 1.0]`, the
product is `[0.3655, 1.7616]`, and `W2` leaves it as it is.

## 3. Rotate the queries and the keys

The attention of the library has no notion of order: if the tokens are
reversed, the output is reversed and nothing else changes. The rotary
embedding rotates the queries and the keys of every head by an angle
proportional to the position, so that the score between two positions
depends on their distance.

```python
import jax
import jax.numpy as jnp
from dense_armor.utility.neural.attention import MultiHeadSelfAttention

x = jax.random.normal(jax.random.PRNGKey(1), (6, 8))
plain = MultiHeadSelfAttention(8, 2, jax.random.PRNGKey(0))
rot = MultiHeadSelfAttention(8, 2, jax.random.PRNGKey(0), rotary=True)
a = float(jnp.max(jnp.abs(plain.forward(x) - plain.forward(x[::-1])[::-1])))
b = float(jnp.max(jnp.abs(rot.forward(x) - rot.forward(x[::-1])[::-1])))
print(round(a, 4), round(b, 4))
```

```
0.0 1.5552
```

For the attention without positions the difference between "reverse the
input" and "reverse the output" is `0.0`: the order is invisible. With
`rotary=True` it is `1.5552`: the output depends on the order. The default
is `rotary=False`, and then the layer gives the same values as before.

The formula rotates each pair of adjacent numbers `(a, b)` of a query or a key
at position `m` by the angle `m * theta_i`: `(a cos(m theta_i) - b sin(m
theta_i), a sin(m theta_i) + b cos(m theta_i))`, with `theta_i = 10000^(-2 i /
d)` for the pair `i = 0, 1, ...` and `d` the dimension of one head. A small
case: `d = 2`, the pair `(1, 0)` at position `1` becomes `(cos 1, sin 1) =
(0.5403, 0.8415)`. A rotation does not change lengths, and rotating by `m` and
by `n` differs by a rotation of `m - n`, so the product of a query at `m` and a
key at `n` depends only on `m - n`:

```python
import jax
import jax.numpy as jnp
from dense_armor.utility.neural.rotary import apply_rotary, rotary_angles

q = jax.random.normal(jax.random.PRNGKey(0), (8,))
k = jax.random.normal(jax.random.PRNGKey(1), (8,))
cos, sin = rotary_angles(10, 8)
rq = apply_rotary(jnp.tile(q, (10, 1)), cos, sin)
rk = apply_rotary(jnp.tile(k, (10, 1)), cos, sin)
print([round(float(rq[m] @ rk[n]), 4) for m, n in [(5, 2), (9, 6), (5, 3)]])
```

```
[-0.2266, -0.2266, 0.0591]
```

The first two products are equal: the distance is `3` in both `(5, 2)` and
`(9, 6)`. The third, at distance `2`, is different.

## 4. Train with AdamW

`AdamW` is Adam with the weight decay applied to the weights directly, not
added to the gradient, so that the decay does not change with the size of the
gradients.

```python
import jax
from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.neural.models import mlp
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.optim import AdamW

X = jax.random.normal(jax.random.PRNGKey(1), (32, 4))
y = jax.random.normal(jax.random.PRNGKey(2), (32, 2))
t = Trainer(mlp([4, 8, 2], jax.random.PRNGKey(0)), MSE(), AdamW(0.01, weight_decay=0.1))
t.fit(X, y, epochs=5, batch_size=8, key=jax.random.PRNGKey(0))
print([round(v, 4) for v in t.history_["train_loss"]])
```

```
[4.961, 1.4344, 1.0681, 1.0447, 1.0493]
```

The loss falls in the first epochs. The decay is `eta * weight_decay *
weight` at every step, where `eta` is an attribute of the optimizer that
equals `1.0` until it is set.

The update is `w <- w - lr * m_hat / (sqrt(v_hat) + eps) - eta * weight_decay *
w`, where `w` is a weight, `m_hat` and `v_hat` are the bias-corrected moving
averages of the gradient and of its square (as in Adam), `lr` is the step
size, `eps` a small constant, `weight_decay` the decay coefficient and `eta`
the multiplier of the decay. A small case, the first step: `w = 1.0`, gradient
`0.5`, `lr = 0.1`, `weight_decay = 0.5`, `eta = 1`. At the first step `m_hat`
is the gradient and `sqrt(v_hat)` its absolute value, so the Adam term is `0.1
* 0.5 / 0.5 = 0.1`; the decay is `1 * 0.5 * 1.0 = 0.5`; and `w = 1.0 - 0.1 -
0.5 = 0.4`.

## Details

### References

- Loshchilov, I., Hutter, F. (2019). Decoupled Weight Decay Regularization.
  ICLR 2019, arXiv:1711.05101. Algorithm 2, page 3 (line 12 is the decoupled
  decay; line 6 adds the decay to the gradient and is the L2 variant).
- Zhang, B., Sennrich, R. (2019). Root Mean Square Layer Normalization.
  NeurIPS 2019, arXiv:1910.07467. Section 4, Eq.(4), page 3.
- Shazeer, N. (2020). GLU Variants Improve Transformer. arXiv:2002.05202.
  Eq.(6), page 2 (the feed-forward layers); Eq.(4), page 1, and Eq.(5), page 2 (the
  units with biases).
- Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., Liu, Y. (2021). RoFormer:
  Enhanced Transformer with Rotary Position Embedding. arXiv:2104.09864.
  Section 3.2.2, Eq.(14) to (16), page 5; Section 3.4.2, Eq.(34), page 7.

### Choices stated explicitly

- `RMSNorm`: `eps` inside the root is our choice (`1e-5`, the value of
  `LayerNorm` in this library); the equation of the paper has no constant.
- `GLUFeedForward`: the biases are off by default, as in Eq.(6); with
  `bias=True` the layer follows the forms of Eq.(4) and Eq.(5) (our choice). `d_ff` is
  not reduced by `2/3` as the paper does to match the parameters of a
  two-matrix layer.
- Rotary: the pairs are adjacent dimensions, as in Eq.(34); the angles are
  `10000^(-2(i-1)/d)`. The rotation is applied to the queries and the keys
  of every head, with the same angles for every head. `step` rotates the new
  query and the new key at the position equal to the length of the cache and
  stores the rotated key.
- `AdamW`: the decay is `eta * weight_decay * weight`, as in line 12 of the
  algorithm, and not `lr * weight_decay * weight`. `Trainer` sets `lr` at each
  step to the value of the schedule, which contains the multiplier of the
  paper; the decay uses the attribute `eta`, which the caller sets.
- `AdamW` decays every array it is given. `Trainer(decay="decoupled")` with
  `Adam` decays only the arrays with two or more dimensions. The two give
  the same trajectory when all the parameters of the model are such arrays
  and the multiplier is the same (tested in `test/optim/test_adamw.py`); they
  differ when the model has biases, or when a schedule decays the learning
  rate and `eta` is not set.

### Measured cost

Not measured.

### Declared limits

- `TransformerBlock` has no `rotary` argument; the option is on
  `MultiHeadSelfAttention` only.
- `Trainer(compile=True)` does not support `AdamW`: the compiled step knows
  SGD, Adam, RMSProp and AdaGrad.
- The classes were tested on a CPU. No GPU, no training-quality comparison
  with `LayerNorm`, a two-matrix feed-forward layer or sinusoidal positions.
