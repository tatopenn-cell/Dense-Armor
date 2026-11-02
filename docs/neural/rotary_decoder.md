# Rotary decoder: how do I generate text without running the whole sequence again at every token?

A language model writes one token at a time. To choose the next token it looks at every token before it, and the numbers it computes for the old tokens (their keys and values in each attention) do not change when a new token arrives. The rotary decoder keeps those numbers in a cache of fixed shape, so a new token costs one step and not a pass over the whole text. The page builds the decoder, runs one token against the cache, checks that the cache gives the same text as the full pass, and measures what it saves.

## 1. Build the decoder

The decoder is an embedding, a stack of blocks, a final norm and a linear head that gives one score per word of the vocabulary.

```python
import jax
from dense_armor.utility.deep.neural.rotary_decoder import rotary_decoder

m = rotary_decoder(65, 64, 4, 2, 256, key=jax.random.PRNGKey(0))
print(sum(int(p.value.size) for p in m.params()))
```

```
139777
```

The arguments are the vocabulary (65), the width `d_model` (64), the heads (4), the blocks (2) and the longest text `max_len` (256). Each block normalises with `RMSNorm`, runs causal attention whose queries and keys are rotated by their position (the rotary embedding), adds the result to its input, then normalises again and runs a feed-forward layer with a gate (`SwiGLU`) and adds that too. The feed-forward width is four times `d_model` unless `d_ff` is given.

## 2. Run one token against the cache

The cache holds, for every block, the keys and the values of the tokens seen so far, in two arrays of shape `(max_len, d_model)`.

```python
import jax
import jax.numpy as jnp
from dense_armor.utility.deep.neural.rotary_decoder import rotary_decoder

m = rotary_decoder(65, 64, 4, 2, 256, key=jax.random.PRNGKey(0))
ids = jnp.arange(8, dtype=jnp.int32)
full = m.forward(ids)
c = m.init_static_cache(256)
for t in range(8):
    last, c = m.step_static(ids[t], c, t)
print(float(jnp.max(jnp.abs(last - full[-1]))) < 1e-5)
print(c[0][0].shape)
```

```
True
(256, 64)
```

`forward` reads the eight tokens at once. `step_static` takes one token and its position `t`, writes its key and value in row `t` of the cache, rotates the query and the key for position `t`, and attends to the rows `0..t`; the rows after `t` are masked. After eight steps the logits of the last token are the same as the last row of the full pass. The shape of the cache never changes, which is why the step compiles once.

## 3. Let `generate` use the cache

`generate` uses the cache when the model has `init_static_cache` and `step_compiled`.

```python
import jax
import jax.numpy as jnp
from dense_armor.utility.deep.neural.generate import generate
from dense_armor.utility.deep.neural.rotary_decoder import rotary_decoder

m = rotary_decoder(65, 64, 4, 2, 256, key=jax.random.PRNGKey(0))
ids = jnp.arange(8, dtype=jnp.int32)
print(generate(m, ids[:4], max_new=6).tolist())
```

```
[0, 1, 2, 3, 36, 26, 46, 26, 46, 46]
```

The prompt is the first four tokens; the six that follow come from the decoder, which is not trained here, so they are numbers and not words. The prompt goes through the cache one token at a time, then each new token is chosen from the last logits (greedy, `temperature=0`) and fed back. The step reads the weights at each call, so training the model and generating again uses the new weights.

## 4. Check it against the pass without a cache

A model that has only `forward` is run again on the whole text at every token, as before.

```python
import jax
import jax.numpy as jnp
from dense_armor.utility.deep.neural.generate import generate
from dense_armor.utility.deep.neural.rotary_decoder import rotary_decoder

m = rotary_decoder(65, 64, 4, 2, 256, key=jax.random.PRNGKey(0))
ids = jnp.arange(8, dtype=jnp.int32)


class Plain:
    max_len = 256

    def forward(self, x):
        return m.forward(x, training=False)


a = generate(Plain(), ids[:4], max_new=6)
print(bool((a == generate(m, ids[:4], max_new=6)).all()))
```

```
True
```

`Plain` hides the cache methods, so `generate` takes the old path. The two give the same six tokens. With `temperature` above zero and the same `key`, the two paths also give the same text.

## 5. Spend time to save memory

`remat=True` makes each block keep only its input in the forward pass and compute the rest again in `backward`.

```python
import jax
from dense_armor.utility.deep.neural.rotary_decoder import rotary_decoder

m = rotary_decoder(65, 64, 4, 2, 256, key=jax.random.PRNGKey(0), remat=True)
print(m.blocks[0])
```

```
RotaryDecoderBlock(64, 4, d_ff=256, remat=True)
```

The gradients are the same as without it; the cost is one more forward pass per block while training. It does not change how `generate` works.

## 6. What the cache saves

The script `benchmarks/neural/kv_cache.py` generates 100 tokens from a prompt of 8 with both paths, five times each after one warm-up call. The numbers are in `benchmarks/kv_cache_cpu.json`.

```python
import json
from pathlib import Path

d = json.loads(Path("benchmarks/kv_cache_cpu.json").read_text())
print(f"{d['cache']['median_s']:.3f}")
print(f"{d['no_cache']['median_s']:.3f}")
print(f"{d['speedup_median']:.1f}")
```

```
0.055
1.383
25.0
```

The first number is the median time in seconds with the cache, the second without, the third how many times faster the cache is. The model is the one of step 1, on a CPU.

## Details

### References

- Su et al. 2021, "RoFormer: Enhanced Transformer with Rotary Position Embedding", Eq.(15) page 5 (the rotary matrix, with `theta_i = 10000^(-2(i-1)/d)`).
- Shazeer 2020, "GLU Variants Improve Transformer", Eq.(6) page 2 (`FFN_SwiGLU`, no biases).
- Zhang and Sennrich 2019, "Root Mean Square Layer Normalization", Eq.(4) page 3 (`RMSNorm` adds an `eps` inside the root, which the paper does not have).
- Vaswani et al. 2017, "Attention Is All You Need", section 3.1 page 3 (the decoder stack and the causal mask).

### Choices stated explicitly

- **The norm comes before each sub-layer (pre-norm).** Vaswani et al. 2017 put it after the residual sum. This is our choice and is not in the papers above.
- **The static cache is our choice.** The papers above do not describe one. It has the fixed shape `(max_len, d_model)`, the key and the value of position `t` are written in row `t`, and the rows after `t` are masked. A step reads all `max_len` rows, whatever the position.
- **The cache is for one sequence.** `step_static` takes one token of shape `(d_model,)`; there is no batch.
- **The prompt goes through the cache one token at a time,** not in one pass.
- **`d_ff` is `4 * d_model` by default,** as in Vaswani et al. 2017, Table 3 page 9 (the `d_ff = 2048` of the base model is `4 * 512`).
- **The same `max_len` bounds the text and the cache.** `generate` stops when the text reaches `max_len`.

### Measured cost

Machine: Kaggle CPU, `python` 3.13.15, `jax` 0.11.1, `numpy` 2.5.4. Model of step 1, 139,777 parameters. Prompt of 8 tokens, 100 new tokens, five repeats, not a statistical test.

| path | first call (s) | median (s) | range (s) |
|---|---|---|---|
| with the cache | 0.794 | 0.055 | 0.053-0.134 |
| without the cache | 2.258 | 1.383 | 1.363-231.617 |

Both paths give the same tokens. The first of the five repeats without the cache took 231.617 s against about 1.4 s for the others; the cause was not measured.

### Not measured

- A GPU or a TPU, a longer context than 256, a larger model, and the memory the cache uses.
- A batch of sequences.
- The speed of `remat=True` and the memory it saves.

### Development diary

The first version of the step wrote the cache with `dynamic_update_slice(cache, row, (pos, 0))`. With `jax_enable_x64`, which the test suite sets, `pos` is `int32` and the literal `0` is `int64`, and JAX refuses indices of different types; the column index is now `zeros_like(pos)`. The rotary angles were built with `jnp.arange(offset, offset + length)`, which needs a concrete offset, and a compiled step has a traced one; they are now `offset + jnp.arange(length)`. Both showed up as failing tests on the first run and are covered by the tests of `test_rotary_decoder.py`.
