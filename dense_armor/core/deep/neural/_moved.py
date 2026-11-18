"""Old module paths of the neural package, kept as aliases of the new ones.

The modules of ``dense_armor.core.deep.neural`` live in sections (``layers``,
``attention``, ``transformer``, ``losses``, ``training``, ``online``, ``models``).
``dense_armor.core.deep.neural.moe`` still works: it is the same module object as
``dense_armor.core.deep.neural.attention.moe``.
"""

import importlib
import sys

_SECTION_OF = {
    "activations": "layers",
    "activations_more": "layers",
    "adapter": "layers",
    "autoencoder": "models",
    "background": "training",
    "batching": "training",
    "bidirectional": "layers",
    "bottleneck": "attention",
    "chain": "layers",
    "clip": "training",
    "compiled": "training",
    "continual": "online",
    "conv": "layers",
    "conv_more": "layers",
    "convtranspose": "layers",
    "cross_attention": "attention",
    "data_parallel": "training",
    "decoder": "transformer",
    "gated_attention": "attention",
    "generate": "transformer",
    "generative_replay": "online",
    "glu": "layers",
    "gradcheck": "layers",
    "hedge": "online",
    "init": "layers",
    "latent_attention": "attention",
    "losses_lm": "losses",
    "losses_more": "losses",
    "matryoshka": "losses",
    "moe": "attention",
    "norm": "layers",
    "norm_more": "layers",
    "norm_rms": "layers",
    "online_autoencoder": "online",
    "online_vector": "online",
    "params": "layers",
    "positions": "attention",
    "quantize": "models",
    "recurrent": "layers",
    "rotary": "attention",
    "rotary_decoder": "transformer",
    "schedules": "training",
    "schedules_more": "training",
    "seq2seq": "transformer",
    "stream": "training",
    "test_time": "transformer",
    "trainer": "training",
}

for _old, _section in _SECTION_OF.items():
    _module = importlib.import_module(f"{__package__}.{_section}.{_old}")
    sys.modules[f"{__package__}.{_old}"] = _module
    if not hasattr(sys.modules[__package__], _old):
        setattr(sys.modules[__package__], _old, _module)
