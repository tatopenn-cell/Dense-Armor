"""TensorFlow bridge for Dense-Armor.

Converts a TensorFlow tensor (or a pytree of tensors) to a jax array (or
a pytree of jax arrays) and back, keeping the dtype, and wraps a
Dense-Armor model as a ``tf.keras`` layer through ``jax2tf.convert``.

Needs the optional dependency: ``pip install dense-armor[tensorflow]``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import tensorflow as tf
    from jax.experimental import jax2tf
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "dense_armor.bridges.tensorflow needs tensorflow and jax: "
        "pip install dense-armor[tensorflow]"
    ) from exc

__all__ = ["KerasLayer", "KerasAdapter", "from_keras", "to_jax", "to_tf"]


def _to_jax_leaf(t: Any) -> Any:
    if isinstance(t, tf.Tensor):
        return jax.dlpack.from_dlpack(t)
    if isinstance(t, tf.Variable):
        return jax.dlpack.from_dlpack(tf.convert_to_tensor(t))
    return t


def _to_tf_leaf(a: Any) -> Any:
    if isinstance(a, jax.Array):
        return tf.experimental.dlpack.from_dlpack(a.__dlpack__())
    return a


def to_jax(x: Any) -> Any:
    """Convert a TensorFlow tensor, or a pytree of them, to jax.

    Args:
        x: a ``tf.Tensor`` or ``tf.Variable``, or any pytree whose leaves
            may be one of them.

    Returns:
        A ``jax.Array`` for a tensor input, or a pytree with the same
        structure and ``jax.Array`` leaves. The dtype is kept: a float32
        tensor becomes a float32 array, no float64 is introduced.

    Example:
        >>> import tensorflow as tf
        >>> from dense_armor.bridges.tensorflow import to_jax
        >>> t = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
        >>> a = to_jax(t)
        >>> a.dtype, a.shape
        (dtype('float32'), (3,))
    """
    return jax.tree_util.tree_map(_to_jax_leaf, x)


def to_tf(a: Any) -> Any:
    """Convert a jax array, or a pytree of them, to TensorFlow.

    The JAX side exports a DLPack capsule (``a.__dlpack__()``); TensorFlow
    2.21 expects the capsule, not the array.

    Args:
        a: a ``jax.Array``, or any pytree whose leaves may be one.

    Returns:
        A ``tf.Tensor`` for an array input, or a pytree with the same
        structure and ``tf.Tensor`` leaves. The dtype is kept.

    Example:
        >>> import jax.numpy as jnp
        >>> from dense_armor.bridges.tensorflow import to_tf
        >>> a = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        >>> t = to_tf(a)
        >>> t.dtype, tuple(t.shape)
        (tf.float32, (3,))
    """
    return jax.tree_util.tree_map(_to_tf_leaf, a)


class KerasLayer(tf.keras.layers.Layer):
    """A Dense-Armor model as a ``tf.keras.layers.Layer``.

    Every Dense-Armor ``Param`` becomes a ``tf.Variable``. The forward
    runs the JAX model through ``jax2tf.convert``; the gradient with
    respect to the input and to the weights is computed by TensorFlow's
    ``GradientTape`` through the converted function.
    """

    def __init__(self, model: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        import copy

        self._model = copy.deepcopy(model)
        self._w = [
            tf.Variable(
                np.asarray(p.value, dtype=np.float32),
                trainable=True,
                dtype=tf.float32,
            )
            for p in model.params()
        ]
        self._trainable_variables_list = list(self._w)

    @property
    def trainable_variables(self) -> list:
        return list(self._w)

    def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        import copy

        def f(x_arg: Any, *w_args: Any) -> Any:
            m = copy.deepcopy(self._model)
            for param, w in zip(m.params(), w_args):
                param.value = w
            return m.forward(x_arg, training=False)

        f_tf = jax2tf.convert(f)
        return f_tf(x, *[tf.convert_to_tensor(w) for w in self._w])


def _make_apply_jax(module: Any) -> Any:
    cache = {"out_dim": None}

    @jax.custom_vjp
    def _apply(x: Any) -> Any:
        return _forward(x)

    def _forward(x: Any) -> Any:
        x_t = tf.convert_to_tensor(np.asarray(x))
        if cache["out_dim"] is None:
            probe = module(x_t)
            cache["out_dim"] = int(probe.shape[-1])
        out_shape = jax.ShapeDtypeStruct((x.shape[0], cache["out_dim"]), jnp.float32)

        def cb(x_np: Any) -> Any:
            return module(tf.convert_to_tensor(x_np)).numpy()

        return jax.pure_callback(cb, out_shape, x)

    def _fwd(x: Any) -> tuple:
        return _forward(x), (x,)

    def _bwd(res: tuple, g: Any) -> tuple:
        (x,) = res
        g_shape = jax.ShapeDtypeStruct(x.shape, jnp.float32)

        def cb(x_np: Any, g_np: Any) -> Any:
            x_t = tf.Variable(x_np, dtype=tf.float32)
            with tf.GradientTape() as tape:
                tape.watch(x_t)
                y = module(x_t)
                y = y * tf.convert_to_tensor(g_np)
            g_x = tape.gradient(y, x_t)
            return g_x.numpy() if g_x is not None else np.zeros_like(x_np)

        g_x = jax.pure_callback(cb, g_shape, x, g)
        return (g_x,)

    _apply.defvjp(_fwd, _bwd)
    return _apply


class KerasAdapter:
    """A ``tf.keras`` model usable inside Dense-Armor.

    ``apply(x)`` runs the Keras model on ``x``. For a jax array input the
    gradient is computed with ``tf.GradientTape`` and exposed to jax
    through ``jax.custom_vjp``, so a Dense-Armor pipeline can back
    propagate through the adapter.

    Args:
        module: a compiled ``tf.keras.Model``.

    Example:
        >>> import tensorflow as tf
        >>> from dense_armor.bridges.tensorflow import from_keras
        >>> module = tf.keras.Sequential([
        ...     tf.keras.layers.Dense(8, activation="relu"),
        ...     tf.keras.layers.Dense(2)])
        >>> adapter = from_keras(module)
        >>> tuple(adapter.apply(tf.zeros((3, 4))).shape)
        (3, 2)
    """

    def __init__(self, module: Any) -> None:
        self.module = module
        self._apply_jax = _make_apply_jax(module)

    def apply(self, x: Any) -> Any:
        if isinstance(x, tf.Tensor) or isinstance(x, tf.Variable):
            return self.module(x)
        return self._apply_jax(x)


def from_keras(module: Any) -> KerasAdapter:
    """Wrap a ``tf.keras`` model as a Dense-Armor adapter.

    Args:
        module: a compiled ``tf.keras.Model``.

    Returns:
        A ``KerasAdapter`` whose ``apply`` runs the model on the input
        and, for a jax array input, exposes the gradient through
        ``jax.custom_vjp``.
    """
    return KerasAdapter(module)
