# Language-model training: how do I train a decoder on a GPU?

A 14.2-million-parameter decoder trained on the tiny Shakespeare text with the modern blocks of this library: RMSNorm, SwiGLU, rotary attention, a causal mask. The page shows the data, the baselines, the model, the run on one and two Tesla T4, the text the model produces, and what changes when the training moves to two devices.

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

The model is an embedding, six decoder blocks, a final RMSNorm and a linear head. Each block is a pre-norm residual: RMSNorm followed by causal rotary attention, then RMSNorm followed by SwiGLU feed-forward.

```python
import jax
from benchmarks.train_small_lm import _build_model, _config

m = _build_model(_config(False), jax.random.PRNGKey(0))
print(sum(int(p.value.size) for p in m.params()))
```

```
14210753
```

The config: d_model 384, six heads, six layers, d_ff 1536, max_len 256, vocabulary 65. The d_ff = 4 * d_model ratio of Vaswani et al. Table 3 is kept.

## 4. The run: 500 steps, loss from 4.80 to 1.91

The training uses Adam with decoupled weight decay (Trainer(decay="decoupled"), the AdamW update of Loshchilov and Hutter 2019), bfloat16 compute with float32 master weights, the compiled step, batch 16, sequence 256. The recorded loss, from the JSON:

```python
import json
from pathlib import Path

d = json.loads(Path("benchmarks/train_small_lm_gpu_1dev.json").read_text())
for step, loss in d["loss_curve"].items():
    print(f"step {int(step):>3}  loss {loss:.4f}")
```

```
step   1  loss 4.8044
step 100  loss 2.5032
step 500  loss 1.9125
```

On one device the final validation loss is 2.0124, the bigram baseline is 2.4819, the ratio is 0.811. On two devices the final validation loss is 2.2235, the ratio is 0.896: the same 500 steps, the same seed, the same batches, and a worse model. The 2-device run is not only slower, it trains less. The cause is not identified in this task; it is declared in the Details.

## 5. The variance: the training stabilises on one device, not on two

The variance of the per-step training loss, computed by `stable_var` from `dense_armor.core.signals.stats.stable`. Two windows: the first 100 steps against the last 100, and the last 100 against the 100 before them (401-500 vs 301-400). The second is the honest one: the first window includes the descent from 4.80 to 2.50, so its variance is large by construction.

```python
import json
from pathlib import Path

for name in ("gpu_1dev", "gpu_2dev"):
    d = json.loads(Path(f"benchmarks/train_small_lm_{name}.json").read_text())
    print(f"--- {name} ---")
    print(f"first 100 : {d['loss_variance_first_100']:.6f}")
    print(f"last 100  : {d['loss_variance_last_100']:.8f}")
    print(f"tail ratio: {d['loss_variance_ratio_tail']:.4f}")
    print(f"claim 5   : {d['claim_5_holds']}")
```

```
--- gpu_1dev ---
first 100 : 0.290992
last 100  : 0.00124835
tail ratio: 0.7696
claim 5   : True
--- gpu_2dev ---
first 100 : 0.291017
last 100  : 0.00188044
tail ratio: 1.8584
claim 5   : False
```

On one device the last 100 steps move less than the 100 before them: ratio 0.77, the training is settling. On two devices the last 100 move more: ratio 1.86, the run is noisier at the end than it was in the middle. Claim 5 (the tail variance shrinks) holds on one device and fails on two.

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
'First Citizen:\nAnd the see the so the son th'
"First Citizen:\n'Dey,nos; brot shall in this "
```

The greedy text repeats "the": the mark of a 14.2-million-parameter model after 500 steps. The pattern is there, the grammar is not. On two devices the greedy output is worse: `First Citizen:\nAnd the the the the the the t`, the same collapse seen in the loss.

## Details

### References

- Vaswani et al. 2017, "Attention is all you need", section 3.1 page 3 (decoder stacks and causal mask), Table 3 page 9 (`d_ff = 4 * d_model`).
- Su et al. 2021, "RoFormer", Eq.(15) page 5, Eq.(34) page 7 (the pair rotation).
- Shazeer 2020, "GLU variants improve Transformer", Eq.(6) page 2 (SwiGLU with Swish-1), page 2 (no biases).
- Loshchilov and Hutter 2019, "Decoupled weight decay regularization", Algorithm 1 page 4. The script uses `Adam` plus `Trainer(decay="decoupled")`.
- Ogita, Rump, Oishi 2005, "Accurate Sum and Dot Product", Algorithm 3.1 page 7 (TwoSum).
- Shewchuk 1997, Theorem 7 page 12 (TWO-SUM) and Lemma 5 page 10 (Sterbenz).
- Healy 2022, arXiv:2206.10662, Algorithm 2 (shifted data, K = x[0]). The `stable_var` kernel follows it.
- Karpathy's `char-rnn`, `input.txt` of "tiny Shakespeare", SHA-256 `86c4e6aa...565ed`.

### Choices stated explicitly

- **`transformer_decoder` is not used.** The builder in `deep/neural/models.py` composes the 2017 blocks. The script composes RMSNorm, SwiGLU and rotary with the blocks of `deep/neural/`.
- **AdamW via `Adam` plus decoupled decay.** `deep/neural/compiled.py` does not dispatch `AdamW`.
- **bfloat16, not float16.** No loss scaling needed.
- **30 new tokens, not 200.** `generate` has no KV cache: about 2 s/token on T4.
- **Claim 5 uses the tail window** (401-500 vs 301-400), not the first window, so the descent from 4.80 to 2.50 does not enter the denominator. On fewer than 400 steps the claim is `None`.
- **Speed is one `fit()` per session**, not one `fit()` per batch: with more than one device the trainer copies weights and Adam state at each `fit()`.

### Measured cost

Machine: Kaggle, `device_kind` "Tesla T4", `device_count` 2, `python` 3.13.15, `jax` 0.11.1, `numpy` 2.2.6, 3 sessions per run.

| run | devices | tokens/s median | tokens/s range | peak bytes | val_loss | ratio to bigram | claim 5 |
|---|---|---|---|---|---|---|---|
| 1 device | 1 | 35,983 | 28,843-36,271 | 1,427,065,088 | 2.0124 | 0.811 | True |
| 2 devices | 2 | 35,700 | 27,218-35,932 | 1,578,250,752 | 2.2235 | 0.896 | False |

The two runs use the same seed, data and steps. The 2-device run is 0.992 times the 1-device run on the median, well below the 1.5 set before the run: claim 3 does not hold on batch 16.

### Not measured

- Why the 2-device run trains worse (val_loss 2.22 against 2.01) at the same seed, data and 500 steps. Possible causes: the shard/all-reduce round of `Trainer.fit` with `devices=2` applied to the members of `RotaryDecoder`, the batch 16 split in two 8-sample halves, or a different path in `compiled.py::_train_parallel`. The two runs use the same script and flags except `--devices`; the difference is not explained by any number in the JSON.
- A 2-device run with batch 32 or 64, where data parallelism usually pays off.
- The TPU: a TPU v3-8 has 8 devices.

### Development diary

The first training step is 6.3 s on T4 with XLA compile, the next 10 are 2.4 s: 240 ms/step. `generate` has no KV cache: 20 tokens cost 47.7 s and 100 cost 230.3 s. The 3-session run with 500 steps took 179 s on 1 device and 530 s on 2 devices. The 2-device number and the loss difference point in the same direction: the parallel path is doing something different, not just slower.
