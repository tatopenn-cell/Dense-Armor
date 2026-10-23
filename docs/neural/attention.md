# Attention: how does a network look at the other steps of a sequence, and where is each piece in the library?

A recurrent layer carries a memory from step to step. Attention does the opposite: each step looks directly at every other step and decides how much each one matters. This page starts from one head by hand, goes up to the multi-head layer, the block and the encoder, and ends with the two ways to tell the network where each step sits in the sequence. The recurrent layers are on the page [Recurrent layers and attention](sequence_layers.md).

## 1. One head, by hand

The scaled dot-product attention of Eq.(1), page 4 of Vaswani et al. 2017 is `Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V`. A query is compared with every key, the scores become weights that sum to 1, and the output is the average of the values under those weights. With one query `[1, 0]`, two keys `[1, 0]` and `[0, 1]` (so `d_k = 2`), and values `10` and `20`, the score of the first key is `1/sqrt(2) = 0.7071` and the score of the second is `0`. The softmax gives the weights `0.6698` and `0.3302`, and the output is `0.6698 * 10 + 0.3302 * 20 = 13.3024`.

~~~python
import jax.numpy as jnp
from dense_armor.utility.neural.attention import scaled_dot_product_attention

q = jnp.array([[1.0, 0.0]])
k = jnp.array([[1.0, 0.0], [0.0, 1.0]])
v = jnp.array([[10.0], [20.0]])
y = scaled_dot_product_attention(q, k, v)
print(round(float(y[0, 0]), 4))
~~~

~~~
13.3024
~~~

The query is closer to the first key than to the second, so the first value gets the larger weight and the output sits between the two values, nearer to 10. The division by `sqrt(d_k)` keeps the scores small when the vectors grow, so the softmax stays in a region where its gradient is not negligible.

## 2. Several heads, the shapes

A `MultiHeadSelfAttention` projects the input into queries, keys and values, splits them into `n_heads` heads of size `d_model / n_heads`, runs the attention of step 1 on each head, joins the heads and projects the result. The four projection matrices `W_Q`, `W_K`, `W_V` and `W_O` are learned (page 5 of the paper).

~~~python
import jax
import jax.numpy as jnp
from dense_armor.utility.neural.attention import MultiHeadSelfAttention

m = MultiHeadSelfAttention(8, 2, jax.random.PRNGKey(0))
print(m.forward(jnp.ones((4, 8))).shape)
print(len(m.params()))
~~~

~~~
(4, 8)
4
~~~

A sequence of 4 steps with `d_model = 8` comes back with the same shape, because self-attention keeps the width. The four parameters are the four projection matrices. A `d_model` that `n_heads` does not divide raises `ValueError`.

## 3. A causal cache, for a stream

With `causal = True`, step `t` looks only at the steps up to `t`. The method `step` takes one sample, appends its key and value to a cache and returns the output with the new cache, so a control loop pays for one step and not for the whole sequence. The output of `step` called `T` times equals the output of `forward` on the whole sequence.

~~~python
import jax
import jax.numpy as jnp
import numpy as np
from dense_armor.utility.neural.attention import MultiHeadSelfAttention

m = MultiHeadSelfAttention(8, 2, jax.random.PRNGKey(0), causal=True)
seq = jax.random.normal(jax.random.PRNGKey(1), (5, 8))
yf = m.forward(seq)
cache = m.init_cache()
ys = []
for t in range(5):
    yt, cache = m.step(seq[t], cache)
    ys.append(yt)
print(bool(np.allclose(np.asarray(jnp.stack(ys)), np.asarray(yf), atol=1e-5)))
~~~

~~~
True
~~~

`init_cache` returns two empty arrays of shape `(0, d_model)`, and each step adds one row to both. The mask of `forward` is lower-triangular and is applied before the softmax (page 5 of the paper).

## 4. The block: attention, then a small network, with a residual

A `TransformerBlock` is the unit that a transformer stacks. It computes `y = LayerNorm(x + Attention(x))` and then `z = LayerNorm(y + FFN(y))`, where `FFN(x) = max(0, x W_1 + b_1) W_2 + b_2` (page 3 and Eq.(2), page 5). The addition `x + ...` is the residual connection: the block can learn to change little, and the gradient has a direct path back.

~~~python
import jax
from dense_armor.utility.neural.attention import TransformerBlock

b = TransformerBlock(8, 2, 16, jax.random.PRNGKey(0))
x = jax.random.normal(jax.random.PRNGKey(1), (4, 8))
y = b.forward(x)
print(y.shape, len(b.params()))
print(round(float(y.mean()), 4), round(float(y.std(axis=-1).mean()), 4))
~~~

~~~
(4, 8) 12
0.0 1.0
~~~

The 12 parameters are the 4 of the attention, 2 for each of the two `LayerNorm` and 2 for each of the two dense layers of the feed-forward. The last operation is a `LayerNorm`, so every position leaves the block with mean `0` and standard deviation `1`. The block takes `causal=True` and `dropout`, and has `init_cache` and `step` like the attention layer.

## 5. Telling the network where each step sits

Attention sees a set of steps and not an order: if the steps are shuffled, the outputs are shuffled in the same way. The order has to be added to the input, either as a fixed pattern or as a learned vector for each position (page 6 of the paper).

~~~python
import jax
import jax.numpy as jnp
from dense_armor.utility.neural.attention import sinusoidal_positions
from dense_armor.utility.neural.positions import LearnedPositions

p = sinusoidal_positions(4, 4)
print([round(float(v), 4) for v in p[1]])
m = LearnedPositions(4, 3, jax.random.PRNGKey(0))
z = m.forward(jnp.zeros((2, 3)))
print(z.shape, len(m.params()), bool((z == m.P.value[:2]).all()))
~~~

~~~
[0.8415, 0.5403, 0.01, 0.9999]
(2, 3) 1 True
~~~

`sinusoidal_positions(length, d_model)` returns the fixed pattern of sines and cosines of different wavelengths; row 1 starts with `sin(1) = 0.8415` and `cos(1) = 0.5403`. The layer `AddPositions(max_len)` of `dense_armor.utility.neural.models` adds that pattern to its input and has no parameters. `LearnedPositions(max_len, d_model, key)` holds a table `P` of shape `(max_len, d_model)` that training changes, and adds its first `T` rows to a sequence of `T` steps: on a zero input the output is exactly `P[:2]`. The paper reports nearly identical results for the two (page 6 and page 9, Table 3 row (E)).

## 6. The encoder, from a sequence to a label

`transformer_encoder` joins the pieces of the page in one `Sequential`: a dense projection, the positions, `depth` blocks, a mean over the time axis and a dense head.

~~~python
import jax
import jax.numpy as jnp
from dense_armor.utility.neural.models import transformer_encoder

m = transformer_encoder(4, 8, 2, 16, 1, 3, jax.random.PRNGKey(0), 16)
print(m.forward(jnp.ones((5, 4))).shape)
~~~

~~~
(3,)
~~~

A sequence of 5 steps with 4 features gives one vector of 3 numbers: `MeanOverTime` collapses the time axis and the last `Dense(8, 3)` gives the three outputs. The arguments are, in order, `n_in`, `d_model`, `n_heads`, `d_ff`, `depth`, `n_out`, the key and `max_len`; `causal=True` makes every block causal.

## Details

### References

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., Polosukhin, I. (2017). *Attention Is All You Need.* arXiv:1706.03762. Page 3, `LayerNorm(x + Sublayer(x))`; page 4, Eq.(1); page 5, the multi-head attention, the masking and Eq.(2); page 6, the sinusoidal and the learned positional encodings; page 9, Table 3 row (E).
- He, K., Zhang, X., Ren, S., Sun, J. (2016). *Deep Residual Learning for Image Recognition.* arXiv:1512.03385. Page 2, Figure 2; page 3, Eqn.(1), for the residual connection.

### Choices stated explicitly

- The `TransformerBlock` applies the dropout to the output of each sub-layer before the addition, as page 8 of the paper does. The default is `dropout = 0.0`.
- A `MultiHeadSelfAttention` with `causal = True` sets the masked positions to `-inf` before the softmax. The method `step` uses the cache of the past and no mask, since the future is not there yet.
- The scale `std = 0.02` of `LearnedPositions` is our choice; the paper gives no value.

### Development diary

The cross-attention between two sequences and the gate between two signals are the next task of the library and have no page yet.
