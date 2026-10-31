# Decoder language models: how do I train a model that continues a sequence?

A language model reads a sequence of tokens and predicts the next one. A decoder Transformer is the model that does this without looking at the tokens it has not seen yet: at each position `t` the self-attention is masked so that the model can only look at positions up to `t`. This page builds the block, the model, the loss and the generation loop, and trains a small model to continue a repeating pattern.

## 1. The mask: each position looks only backwards

The self-attention of [Attention](attention.md) compares every query with every key. In a decoder, the comparison is masked: the score at position `(i, j)` is kept when `j <= i` and set to `-infinity` when `j > i`, so the softmax at position `i` has no weight on `j > i`. Vaswani et al. 2017, Section 3.1, page 3: "we also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions".

The check is exact, not approximate: change the tokens after `t` and the logits up to `t` do not change at all, bit for bit.

~~~python
import jax
import jax.numpy as jnp
from dense_armor.utility.neural.models import transformer_decoder

m = transformer_decoder(8, 16, 2, 1, 6, key=jax.random.PRNGKey(0))
a = jnp.array([0, 1, 2, 3, 4, 5])
c = jnp.array([0, 1, 2, 7, 6, 5])
la = m.forward(a)
lc = m.forward(c)
print(bool(jnp.array_equal(la[0], lc[0])))
print(bool(jnp.array_equal(la[1], lc[1])))
print(bool(jnp.array_equal(la[2], lc[2])))
print(bool(not jnp.array_equal(la[3], lc[3])))
~~~

~~~
True
True
True
True
~~~

The first three positions are identical because the change is at position 3 and later. Position 3 changes because it now sees the token `7` instead of `3`.

## 2. The decoder block: three sub-layers, post-norm

The encoder block of [Attention](attention.md) has two sub-layers: self-attention and feed-forward. The decoder block has a third sub-layer, cross-attention over the encoder output (Vaswani et al., Section 3.1, page 3). For a language model the encoder output is absent, and the block has two sub-layers again, with the self-attention masked.

The block follows the post-norm order of the paper: `y = LayerNorm(x + Sublayer(x))`.

~~~python
import jax
import jax.numpy as jnp
from dense_armor.utility.neural.decoder import TransformerDecoderBlock

b = TransformerDecoderBlock(8, 2, 16, jax.random.PRNGKey(0))
x = jax.random.normal(jax.random.PRNGKey(1), (4, 8))
print(b.forward(x).shape)
print(len(b.params()))
~~~

~~~
(4, 8)
12
~~~

The block keeps the width of the sequence. The twelve parameters are the four projections of the attention, the gain and the bias of the two layer norms, and the weight and the bias of the two dense layers of the feed-forward.

## 3. The model: embedding, positions, blocks, head

`transformer_decoder` builds the chain of a causal language model: token embedding, sinusoidal positions, `n_layers` decoder blocks, a final layer norm, and an output head to the vocabulary. With `tie=True` the head reuses the embedding matrix. The head gives logits of shape `(T, vocab)` for a prompt of `T` tokens.

## 4. The next-token loss, by hand

The target of position `t` is the token at `t+1`. With a two-token vocabulary, `y = [1, 0, 1]`, and logits `[[2, 0], [0, 2], [2, 0]]`, the log-softmax of the first position at the target `0` is `-0.1269` and the same for the second position at the target `1`. The mean over the two valid positions is `0.1269`, and the perplexity is `e^0.1269 = 1.1353`.

~~~python
import jax.numpy as jnp
from dense_armor.utility.neural.losses_lm import NextTokenLoss, perplexity

L = NextTokenLoss()
logits = jnp.array([[[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]]])
y = jnp.array([[1, 0, 1]])
val = float(L.forward(logits, y))
print(round(val, 4))
print(round(perplexity(val), 4))
~~~

~~~
0.1269
1.1353
~~~

A target of `-100` does not enter the loss and does not enter the count. This is the convention of the field, not a formula of the paper, and it is declared in the docstring.

## 5. Generation: greedy and sampled

At `temperature=0` the next token is the argmax of the last logits. At `temperature>0` the next token is sampled from `softmax(logits / temperature)` and a `jax.random` key is required. The returned sequence never exceeds `max_len`.

~~~python
import jax
import jax.numpy as jnp
from dense_armor.utility.neural.models import transformer_decoder
from dense_armor.utility.neural.generate import generate

m = transformer_decoder(4, 8, 2, 1, 4, key=jax.random.PRNGKey(0))
out = generate(m, jnp.array([0, 1]), max_new=2, max_len=4)
print(out.tolist())
~~~

~~~
[0, 1, 3, 3]
~~~

An untrained model returns arbitrary tokens. Training is what makes the output continue the pattern.

## 6. Training the repeating pattern

A period-4 pattern `0, 1, 2, 3, 0, 1, 2, 3` on a four-token vocabulary, a model with one block, and Adam with `lr=0.05`. The loss crosses `0.05` in a few dozen steps, and `generate` continues the pattern from a two-token prompt.

~~~python
import numpy as np
import jax
import jax.numpy as jnp
from dense_armor.utility.neural.models import transformer_decoder
from dense_armor.utility.neural.losses_lm import NextTokenLoss
from dense_armor.utility.neural.generate import generate
from dense_armor.utility.optim import Adam

m = transformer_decoder(4, 16, 2, 1, 8, key=jax.random.PRNGKey(0))
tokens = jnp.asarray((np.arange(8) % 4).astype(np.int32))
opts = [Adam(lr=0.05) for _ in m.params()]
L = NextTokenLoss()
for step in range(300):
    m.zero_grad()
    logits = m.forward(tokens)
    loss = float(L.forward(logits, tokens))
    if loss < 0.05:
        break
    m.backward(L.backward())
    for p, o in zip(m.params(), opts):
        p.update(o)
print(step + 1)
print(round(loss, 4))
print(generate(m, jnp.array([0, 1]), max_new=5, max_len=8).tolist())
~~~

~~~
28
0.0244
[0, 1, 2, 3, 0, 1, 2]
~~~

Twenty-eight steps, loss `0.0244`, and the prompt `[0, 1]` is continued with `[2, 3, 0, 1, 2]`, the period-4 pattern.

## Details

### References

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., Polosukhin, I. (2017). Attention Is All You Need. arXiv:1706.03762. Section 3.1, page 3: the decoder stacks and the masking of the self-attention. Section 3.2, page 4: the scaled dot-product attention, Eq.(1). Section 3.3, page 5: the base model, `d_model=512`, `d_ff=2048`. Section 5.4, page 8: label smoothing.
- Ba, J. L., Kiros, J. R., Hinton, G. E. (2016). Layer Normalization. arXiv:1607.06450. Eq.(3), page 2: the layer norm, `y = gain * (x - mean) / sqrt(var + eps) + bias`. The citation is in `norm.py`.

### Choices stated explicitly

- **Positional encoding.** Sinusoidal, Vaswani et al. Section 3.5, page 6. The `LearnedPositions` layer of `positions.py` is available as an alternative and is not used here.
- **Weight tying.** `tie=True` reuses the embedding matrix as the output projection. This is not in Vaswani et al. 2017; it is a common practice in language models, and it is declared in the `_TiedHead` docstring. The default is `tie=False`.
- **`ignore_index = -100`.** A target with this value does not enter the next-token loss and does not enter the count. This is not in Vaswani et al. 2017; it is the convention of the reference implementations of the field, and it is declared in the `NextTokenLoss` docstring.
- **`d_ff` default.** `4 * d_model`, from the base model of Vaswani et al., Section 3.3, page 5 (`d_model=512`, `d_ff=2048`). The parameter is exposed so a caller can choose another value.
- **Generation.** Re-runs the whole prefix at every new token: `O(T**2)` in the length. The cache-based `step` of `MultiHeadSelfAttention` exists but is not exposed by `TransformerDecoderBlock`; the choice is documented in the `generate` docstring.

### Measured cost

No benchmark in this page. The numbers printed by the code blocks come from runs of the blocks themselves, on this page. The training block on a small model (`d_model=16`, one block, sequence 8) takes about ten seconds on the Colab CPU.

### Not measured

GPU speed, larger models (`d_model=512`, six blocks), longer sequences (`max_len > 64`), the quality of the model on a real corpus. The page shows the block, the loss and the generation loop, not a trained language model.

### Development diary

The page was written after the code: the block, the loss and the generation loop passed their tests (`test/neural/test_decoder.py`, 37 tests) before any prose was written. The outputs shown are the outputs of the last run of the blocks, not written by hand.
