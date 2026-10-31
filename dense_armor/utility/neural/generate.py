"""Greedy and sampled generation for a causal language model.

The generation re-runs the model on the whole prefix at every new
token: ``O(T**2)`` in the length, chosen for clarity. The cache-based
``step`` of ``MultiHeadSelfAttention`` would need a ``step`` on
``TransformerDecoderBlock``, which is out of the scope of this item.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def generate(
    model: Any,
    prompt: jax.Array,
    max_new: int,
    temperature: float = 0.0,
    key: Any = None,
    max_len: int | None = None,
) -> jax.Array:
    """Continue ``prompt`` with at most ``max_new`` new tokens.

    At ``temperature == 0`` the next token is the argmax of the last
    logits (greedy). At ``temperature > 0`` the next token is sampled
    from ``softmax(logits / temperature)`` and ``key`` must be given.
    The returned sequence never exceeds ``max_len``; when ``max_len`` is
    ``None`` the model's ``max_len`` attribute is used.

    Args:
        model: an object with ``forward(tokens) -> logits`` of shape
            ``(T, V)``, and optionally a ``max_len`` attribute.
        prompt: an integer array of token ids, shape ``(P,)``.
        max_new: how many tokens to add. ``max_new = 0`` returns the
            prompt unchanged.
        temperature: ``0.0`` for greedy, positive for sampling.
        key: a ``jax.random`` key. Required when ``temperature > 0``.
        max_len: the maximum length of the returned sequence. If
            ``None``, the ``max_len`` attribute of ``model`` is used.

    Returns:
        An integer array of shape ``(min(P + max_new, max_len),)``.

    Raises:
        ValueError: on empty prompt, on non 1-D prompt, on ``max_new < 0``,
            on negative temperature, on ``temperature > 0`` without
            ``key``, or when ``max_len`` is smaller than the prompt.
    """
    prompt = jnp.asarray(prompt, dtype=jnp.int32)
    if prompt.ndim != 1:
        raise ValueError(f"prompt must be 1-D, got shape {prompt.shape}")
    if prompt.shape[0] == 0:
        raise ValueError("prompt is empty")
    if max_new < 0:
        raise ValueError(f"max_new must be >= 0, got {max_new}")
    if temperature < 0.0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    if temperature > 0.0 and key is None:
        raise ValueError("temperature > 0 requires a jax.random key")
    if max_len is None:
        max_len = getattr(model, "max_len", None)
        if max_len is None:
            raise ValueError("max_len is None and the model has no max_len attribute")
    if max_len < prompt.shape[0]:
        raise ValueError(
            f"max_len {max_len} is smaller than the prompt ({prompt.shape[0]})"
        )

    seq = prompt
    for _ in range(max_new):
        if seq.shape[0] >= max_len:
            break
        logits = jnp.asarray(model.forward(seq))
        last = logits[-1]
        if temperature == 0.0:
            nxt = jnp.argmax(last)
        else:
            probs = jax.nn.softmax(last / temperature, axis=-1)
            key, sub = jax.random.split(key)
            nxt = jax.random.categorical(sub, jnp.log(probs), axis=-1)
        seq = jnp.concatenate([seq, jnp.atleast_1d(jnp.asarray(nxt, dtype=seq.dtype))])
    return seq
