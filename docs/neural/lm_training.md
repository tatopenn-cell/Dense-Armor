# Language-model training: how do I train a decoder on a GPU?

A 14.2-million-parameter decoder trained on the tiny Shakespeare text with the modern blocks of this library: RMSNorm, SwiGLU, rotary attention, a causal mask. The page shows the data, the baselines, the model, the run on a Tesla T4, the text the model produces after 500 steps, and how the variance of the training loss shrinks.

## 1. The data: 1.1 MB of Shakespeare, 65 characters

The tiny Shakespeare text (public domain) has 1,115,394 bytes and 65 distinct characters. A character-level model needs no tokenizer: each character is an integer in [0, 65).

```python
from benchmarks.train_small_lm import _load_dataset

data = _load_dataset()
print(data["n_bytes"])
print(data["sha256"][:16])
print(len(data["chars"]))
```

```
1115394
86c4e6aa9db7c042
65
```

The script aborts if the SHA-256 of the downloaded file differs from 86c4e6aa...565ed: a corrupted download is caught before training starts.

## 2. The baselines: unigram and bigram

Before training, two character models give the floor. The unigram counts each character; the bigram counts each pair. Both use add-1 smoothing and are scored on the last 10 % of the text, 111,540 characters, in nats per character.

```python
from benchmarks.train_small_lm import (
    _bigram_loss, _load_dataset, _unigram_loss,
)

data = _load_dataset()
print(round(_unigram_loss(data["train"], data["val"]), 4))
print(round(_bigram_loss(data["train"], data["val"]), 4))
```

```
3.3473
2.4819
```

The two numbers come from benchmarks/train_small_lm_gpu_1dev.json. The model must go below the bigram, 2.4819, to be doing anything more than a pair count.

## 3. The model: 14.2 million parameters, six modern blocks

The model is an embedding, six decoder blocks, a final RMSNorm and a linear head. Each block is a pre-norm residual: RMSNorm followed by causal rotary attention, then RMSNorm followed by SwiGLU feed-forward. The class lives in `benchmarks/train_small_lm.py`; here is its parameter count.

```python
import jax
from benchmarks.train_small_lm import _build_model, _config

m = _build_model(_config(False), jax.random.PRNGKey(0))
print(sum(int(p.value.size) for p in m.params()))
```

```
14210753
```

The config: `d_model` 384, six heads, six layers, `d_ff` 1536, `max_len` 256, vocabulary 65. The `d_ff = 4 * d_model` ratio of Vaswani et al. Table 3 is kept.

## 4. The run: 500 steps, loss from 4.80 to 1.90

The training uses `Adam` with decoupled weight decay (`Trainer(decay="decoupled")`, the AdamW update of Loshchilov and Hutter 2019), bfloat16 compute with float32 master weights, the compiled step, batch 16, sequence 256. The recorded loss, from the JSON:

```python
import json
from pathlib import Path

d = json.loads(Path("benchmarks/train_small_lm_gpu_1dev.json").read_text())
for step, loss in d["loss_curve"].items():
    print(f"step {int(step):>3}  loss {loss:.4f}")
```

```
step   1  loss 4.8044
step 100  loss 2.5033
step 200  loss 2.2151
step 300  loss 2.0058
step 400  loss 1.9720
step 500  loss 1.8979
```

The final validation loss is 1.9956. The bigram baseline is 2.4819, the ratio is 0.804; the target of 0.7 set before the run is not reached on 500 steps. The loss is still falling.

## 5. The variance: the training stabilises

The variance of the per-step training loss, in the first 100 steps and in the last 100, computed by `stable_var` from `dense_armor.utility.signals.stats.stable`:

```python
import json
from pathlib import Path

d = json.loads(Path("benchmarks/train_small_lm_gpu_1dev.json").read_text())
print(f"{d['loss_variance_first_100']:.6f}")
print(f"{d['loss_variance_last_100']:.10f}")
print(f"{d['loss_variance_ratio']:.4e}")
```

```
0.291021
0.0011562244
3.9730e-03
```

The variance drops by a factor of 250. That is what the stabilising run means: the loss moves less from step to step as the model settles. On two devices the ratio is 4.0238e-03, essentially the same. The claims C5 (the variance shrinks, ratio < 1) holds on both runs.

## 6. The text: 44 characters after the prompt

The model generates 30 new characters from the prompt `First Citizen:`. Greedy and temperature 1.0 on the same prompt, from the JSON:

```python
import json
from pathlib import Path

d = json.loads(Path("benchmarks/train_small_lm_gpu_1dev.json").read_text())
print(repr(d["generated_greedy"]))
print(repr(d["generated_sample"]))
```

```
'First Citizen:\nAnd the son the son the son t'
"First Citizen:\n\nDo hnou; but and.\n\nBinCeu,, "
```

The greedy text repeats "the son the": the mark of a 14.2-million-parameter model after 500 steps. The pattern is there, the grammar is not.

## Details

### References

- Vaswani et al. 2017, "Attention is all you need", section 3.1 page 3 (decoder stacks and causal mask), Table 3 page 9 (`d_ff = 4 * d_model`).
- Su et al. 2021, "RoFormer", Eq.(15) page 5, Eq.(34) page 7 (the pair rotation).
- Shazeer 2020, "GLU variants improve Transformer", Eq.(6) page 2 (SwiGLU with Swish-1), page 2 (no biases).
- Loshchilov and Hutter 2019, "Decoupled weight decay regularization", Algorithm 1 page 4. The script uses `Adam` plus `Trainer(decay="decoupled")`, which is the same update (line 252 of `trainer.py`).
- Ogita, Rump, Oishi 2005, "Accurate Sum and Dot Product", Algorithm 3.1 page 7 (TwoSum). The compensated tree in `stable_var` uses it at every level.
- Shewchuk 1997, "Adaptive Precision Floating-Point Arithmetic", Theorem 7 page 12 (TWO-SUM) and Lemma 5 page 10 (Sterbenz, the shift is exact).
- Healy 2022, "Accurate and consistent calculation of the mean and variance in Monte-Carlo simulations", arXiv:2206.10662, Algorithm 2 (shifted data with K = x[0]). The kernel `stable_var` of `dense_armor.utility.signals.stats.stable` follows it.
- Karpathy's `char-rnn` repository, `input.txt` of "tiny Shakespeare" (public domain), SHA-256 `86c4e6aa...565ed`.

### Choices stated explicitly

- **`transformer_decoder` is not used.** The builder in `models.py` composes the 2017 blocks (LayerNorm, two Dense, sinusoidal positions). The script composes RMSNorm, SwiGLU and rotary with the blocks of `attention.py`, `norm_rms.py`, `glu.py`, `layers.py`.
- **AdamW via `Adam` plus decoupled decay.** `compiled.py::_init_state` dispatches `SGD`, `Adam`, `RMSProp`, `AdaGrad`; `AdamW` is not wired.
- **bfloat16, not float16.** 8-bit exponent, no loss scaling needed, native on CPU, T4, TPU.
- **30 new tokens, not 200.** `generate` has no KV cache: 2.4 s/token on the T4, 200 tokens cost about 15 minutes.
- **Claim 5 uses `stable_var` from `dense_armor.utility.signals.stats.stable`**, the same kernel as the float32 variance task. When the two windows overlap (`2 * min(100, n_steps) > n_steps`, i.e. fewer than 200 steps), the claim is `None`, not a number.
- **Speed is measured with one `fit()` covering all steps, not one `fit()` per batch.** With more than one device the trainer copies weights and Adam state to every device at each `fit()` and syncs back at the end; a per-batch `fit()` measures the copy, not the training.

### Measured cost

Machine: Kaggle, `device_kind` "Tesla T4", `device_count` 2, `python` 3.13.15, `jax` 0.11.1, `numpy` 2.2.6.

| run | devices | steps | batch | seq | tokens/s | peak bytes | val_loss | loss var ratio |
|---|---|---|---|---|---|---|---|---|
| 1 device | 1 | 500 | 16 | 256 | 28,746 | 1,424,270,592 | 1.9956 | 3.9730e-03 |
| 2 devices | 2 | 500 | 16 | 256 | 27,743 | 1,456,666,368 | 1.9943 | 4.0238e-03 |

The two runs use the same seed, data and steps. The 2-device run is 0.97 times the 1-device run on tokens per second, well below the 1.5 set before the run: claim 3 does not hold on batch 16. The 2-device run does not pay off with 8-sample shards per device.

### Not measured

- 2 GPUs with a larger batch (32 or 64) or a larger model, where data parallelism usually pays off.
- The TPU: a TPU v3-8 has 8 devices, more than the 2-device story of `data_parallel.py`.
- bfloat16 tensor-core kernels tuned for the T4; the numbers are JAX 0.11 default.

### Development diary

The first training step is 6.3 s on the T4 with XLA compile, the next 10 are 2.4 s: 240 ms/step. `generate` re-runs the whole sequence per token; 20 tokens cost 47.7 s and 100 cost 230.3 s, both about 2.4 s/token. A first run with `max_new=200` on both greedy and sampled generation would have taken 15 minutes on generation alone; the committed JSON uses 30. Claim 5 is what was missing: without `stable_var` a "proof of scale" says the model runs, not that it stabilises.
