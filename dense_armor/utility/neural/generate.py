"""Greedy and sampled generation for a causal language model.

A model that has ``init_static_cache`` and ``step_compiled`` (the
``ModernDecoder`` of ``modern_decoder.py``) is run one token at a time
against a key/value cache of fixed shape: the step compiles once and a
new token costs the same at every length. Any other model with a
``forward`` is run again on the whole prefix at every new token,
``O(T**2)`` in the length.
"""

from __future__ import annotations

import inspect
from typing import Any

import jax
import jax.numpy as jnp


def _pick(last: jax.Array, temperature: float, key: Any) -> tuple[jax.Array, Any]:
    if temperature == 0.0:
        return jnp.argmax(last), key
    key, sub = jax.random.split(key)
    return jax.random.categorical(sub, last / temperature, axis=-1), key


def _generate_cached(
    model: Any,
    prompt: jax.Array,
    max_new: int,
    temperature: float,
    key: Any,
    max_len: int,
) -> jax.Array:
    caches = model.init_static_cache(max_len)
    logits = None
    for i in range(prompt.shape[0]):
        logits, caches = model.step_compiled(prompt[i], caches, i)
    new = []
    n = prompt.shape[0]
    while len(new) < max_new and n < max_len:
        nxt, key = _pick(jnp.asarray(logits), temperature, key)
        nxt = jnp.asarray(nxt, dtype=prompt.dtype)
        new.append(nxt)
        n += 1
        if len(new) < max_new and n < max_len:
            logits, caches = model.step_compiled(nxt, caches, n - 1)
    if not new:
        return prompt
    return jnp.concatenate([prompt, jnp.stack(new)])


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
            ``(T, V)``, and optionally a ``max_len`` attribute. If its
            ``forward`` has a ``training`` argument, it is called with
            ``training=False``.
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

    if hasattr(model, "init_static_cache") and hasattr(model, "step_compiled"):
        return _generate_cached(model, prompt, max_new, temperature, key, max_len)
    takes_training = "training" in inspect.signature(model.forward).parameters
    seq = prompt
    for _ in range(max_new):
        if seq.shape[0] >= max_len:
            break
        if takes_training:
            logits = jnp.asarray(model.forward(seq, training=False))
        else:
            logits = jnp.asarray(model.forward(seq))
        last = logits[-1]
        if temperature == 0.0:
            nxt = jnp.argmax(last)
        else:
            key, sub = jax.random.split(key)
            nxt = jax.random.categorical(sub, last / temperature, axis=-1)
        seq = jnp.concatenate([seq, jnp.atleast_1d(jnp.asarray(nxt, dtype=seq.dtype))])
    return seq
