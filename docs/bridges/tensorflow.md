# TensorFlow bridge

TensorFlow and JAX both hold an array as a typed buffer; the difference is the framework around it. A bridge converts a tensor from one side to the other, and lets a Dense-Armor model be a `tf.keras.layers.Layer` that trains under a Keras optimiser.

## 1. A tensor from one side to the other

`to_jax(t)` converts a `tf.Tensor` or a `tf.Variable` to a `jax.Array`, and `to_tf(a)` does the reverse. The conversion uses DLPack: TensorFlow 2.21 expects the capsule (`a.__dlpack__()`), not the jax array, so `to_tf` calls it explicitly. Both functions accept a pytree of tensors.

```python
import tensorflow as tf
from dense_armor.bridges.tensorflow import to_jax, to_tf

t = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
a = to_jax(t)
back = to_tf(a)
print(a.dtype, tuple(a.shape), bool(tf.reduce_all(t == back)))
```

```
float32 (3,) True
```

The dtype is kept: a float32 tensor stays float32, no float64 is introduced. A non-contiguous tensor is exported through DLPack and comes out with the same values.

## 2. A Dense-Armor model as a Keras layer

`KerasLayer(model)` wraps a Dense-Armor `Sequential` as a `tf.keras.layers.Layer`. Every Dense-Armor `Param` becomes a `tf.Variable`, so a Keras optimiser sees it. `call` runs the JAX model through `jax2tf.convert`; the gradient of the input and of the weights is computed by TensorFlow's `GradientTape` through the converted function.

```python
import tensorflow as tf, jax
from dense_armor.bridges.tensorflow import KerasLayer
from dense_armor.utility.neural import Dense, Sequential
from dense_armor.utility.neural.activations import Relu

model = Sequential([Dense(4, 8, key=jax.random.PRNGKey(0)), Relu(),
                    Dense(8, 3, key=jax.random.PRNGKey(1))])
layer = KerasLayer(model)
x = tf.zeros((2, 4), dtype=tf.float32)
print(tuple(layer(x).shape), len(layer.trainable_variables))
```

```
(2, 3) 4
```

The gradient of the layer output with respect to the input equals `jax.grad` of the same JAX call, and a step of `tf.keras.optimizers.SGD` lowers the loss on a synthetic problem. Both are tested.

## 3. A Keras model inside Dense-Armor

`from_keras(module)` returns an adapter with `apply(x)` that runs the Keras model on `x`. For a jax array input the gradient is computed with `tf.GradientTape` and exposed to jax through `jax.custom_vjp`, so a Dense-Armor pipeline can back propagate through the adapter.

```python
import tensorflow as tf
from dense_armor.bridges.tensorflow import from_keras

module = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(2)])
adapter = from_keras(module)
print(tuple(adapter.apply(tf.zeros((3, 4))).shape))
```

```
(3, 2)
```

## 4. What the bridge costs

`benchmarks/bridge_tensorflow.py` measures the forward of the same model (a `Sequential` of `Dense(64,128)`, `Relu`, `Dense(128,32)`) on batch 256, float32, seven runs: the JAX model called directly, an equivalent `tf.keras.Sequential` called directly, and the bridge.

The claim declared in the first commit (the bridge costs less than 20% of the JAX forward) **does not hold**. On batch 256 the bridge takes 10.5 ms against 0.19 ms of the JAX forward, a factor of 53; the TensorFlow model called directly takes 3.4 ms on the same data, so the bridge adds about 7 ms on top of TensorFlow's own cost. The `jax2tf.convert` of the forward is cached per (shape, training) inside the layer, so only the first call pays the conversion. The full run at batch 256 is in `benchmarks/bridge_tensorflow.json`. The cost is dominated by the TensorFlow layer machinery around the JAX call, not by the DLPack conversion; making it cheaper is a separate task.

## Details

### References

- TensorFlow 2.21, `tf.experimental.dlpack`: <https://www.tensorflow.org/api_docs/python/tf/experimental/dlpack>.
- TensorFlow 2.21, `tf.custom_gradient`: <https://www.tensorflow.org/api_docs/python/tf/custom_gradient>.
- TensorFlow 2.21, `tf.keras.Model`: <https://www.tensorflow.org/api_docs/python/tf/keras/Model>.
- JAX 0.11.1, `jax2tf`: <https://jax.readthedocs.io/en/latest/jax.experimental.jax2tf.html>.

### Choices stated explicitly

- **DLPack with the capsule.** TensorFlow 2.21 `tf.experimental.dlpack.from_dlpack` wants a DLPack capsule, not a jax array: `to_tf` calls `a.__dlpack__()` explicitly. A jax array passed directly raises a `TypeError`.
- **`jax2tf.convert` for the layer, not `custom_gradient`.** The audit shows that `jax2tf.convert` converts `Dense` and `Sequential` without error; the four activations fail only because their `forward` has no `training` keyword, which is not a `jax2tf` limitation. `KerasLayer` uses `jax2tf.convert` and lets TensorFlow's `GradientTape` compute the gradients, so the layer is differentiable end to end without a hand-written backward.
- **The weights are plain `tf.Variable`, not Keras weights.** A Keras `add_weight` returns a `KerasVariable`, which is not a valid argument for `jax2tf.convert`; plain `tf.Variable` works, and the layer exposes them through `trainable_variables`.
- **`from_keras` uses `tf.GradientTape` with `tape.watch`.** The gradient of the Keras model with respect to a jax input is computed by `GradientTape`, and the tape must `watch` the input explicitly (a `tf.Variable` created inside `pure_callback` is not watched automatically). The result matches `jax.grad` of the same function.
- **`KerasLayer.call` deep-copies the Dense-Armor model at every call.** The Dense-Armor layers hold mutable `Param.value`, so a shared model would leak the tracers of the previous call; a copy per call avoids the leak and costs about 1 ms. The `jax2tf.convert` of the forward function is also rebuilt at every call, and that is the larger term of the 68x overhead. A Keras wrapper that caches the converted function is a separate task.
- **`KerasLayer.call` passes `training=False` to the Dense-Armor model.** A `Dropout` or a `BatchNorm` inside the wrapped model stays in evaluation mode even when the Keras layer is in `train()`. A full integration with Keras's training mode is a separate task.
- **The TensorFlow tests need `tensorflow` installed.** In CI the package is not in the `dev` extra, so the tests are skipped by `pytest.importorskip("tensorflow")` and codecov sees 0% on `dense_armor/bridges/tensorflow.py`. Installing the extra in the CI job is a separate change to the workflow, and the workflow file needs a token with the `workflow` scope.
- **`from_keras` does not work under `jax.jit`.** The adapter uses `jax.pure_callback` with `np.asarray` inside, and `np.asarray` on a tracer raises `TracerArrayConversionError`. A jitted version needs `jax.pure_callback` with a `ShapeDtypeStruct` computed without a trace and no `np.asarray` on the tracer, which is a separate task.
- **The output of `from_keras` is 2D (`(batch, out_dim)`).** A model that returns a 3D tensor (a `Reshape`, a small sequence) raises a `JaxRuntimeError`: the `ShapeDtypeStruct` in the callback is built from `module.output_shape[-1]` and the leading dimension is `x.shape[0]`. A general shape is a separate task.
- **The dtype is float32.** A float64 input comes out as float32, because the callback builds the result with `dtype=jnp.float32` and the TensorFlow module runs in float32. There is no warning; the bridge assumes the float32 default of the library.
- **`KerasLayer.call` ignores `training=True`.** The JAX model is called with `training=bool(training)` but the flag is captured when the converted function is built, and the Keras layer does not switch the inner `Dropout` or `BatchNorm` to training mode. A full integration with Keras's training mode is a separate task.
- **The optional dependency is `tensorflow-cpu`.** The package is smaller than the GPU wheel and the bridge is tested on the CPU; a reader with a GPU installs `tensorflow` instead.

### Measured cost

The JSON is `benchmarks/bridge_tensorflow.json`. The claim does not hold and the delivery says so.

### Development diary

The first version of `KerasLayer` used `tf.custom_gradient` with a jax `vjp` inside. TensorFlow 2.21 expects the gradient function of a `custom_gradient` that uses variables to return a pair `(input_grads, variable_grads)`, and the exact shape of that contract differs between calls with and without variables; the version with `jax2tf.convert` and `GradientTape` is shorter and works for the classes the audit names.
