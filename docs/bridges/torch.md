# PyTorch bridge

PyTorch and JAX both hold an array as a typed buffer; the difference is the framework around it. A bridge converts a tensor from one side to the other without copying it, and lets a Dense-Armor model be a `torch.nn.Module` that trains under `torch.optim`.

## 1. A tensor from one side to the other

`to_jax(t)` converts a `torch.Tensor` to a `jax.Array`, and `to_torch(a)` does the reverse. A contiguous CPU tensor keeps its buffer: the two sides point at the same memory, and no copy happens. Both functions also accept a pytree of tensors, so a dict of parameters crosses in one call.

```python
import torch
from dense_armor.bridges.torch import to_jax, to_torch

t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
a = to_jax(t)
back = to_torch(a)
print(a.dtype, tuple(a.shape), t.data_ptr() == back.data_ptr())
```

```
float32 (3,) True
```

The dtype is kept: a float32 tensor stays float32, no float64 is introduced. A tensor that requires grad is detached before the conversion; the caller re-attaches the gradient on the torch side if it needs it. A non-contiguous tensor is made contiguous first, which copies it.

## 2. A Dense-Armor model as a torch module

`TorchModule(model)` wraps a Dense-Armor `Sequential` (or any model with `forward`, `params()`) as a `torch.nn.Module`. Every Dense-Armor `Param` becomes a `torch.nn.Parameter`, so a `torch.optim` optimiser sees it. `forward` calls the JAX model through a `torch.autograd.Function` whose backward is `jax.vjp` of the same call.

```python
import torch, jax
from dense_armor.bridges.torch import TorchModule
from dense_armor.utility.neural import Dense, Sequential
from dense_armor.utility.neural.activations import Relu

model = Sequential([Dense(4, 8, key=jax.random.PRNGKey(0)), Relu(),
                    Dense(8, 3, key=jax.random.PRNGKey(1))])
tm = TorchModule(model)
x = torch.randn(2, 4)
print(tuple(tm(x).shape), len(list(tm.parameters())))
```

```
(2, 3) 4
```

The gradient of the torch output with respect to the torch input equals `jax.grad` of the same JAX call, and a step of `torch.optim.SGD` lowers the loss on a synthetic problem. Both are tested.

## 3. A torch module inside Dense-Armor

`from_torch(module)` returns an adapter with `apply(x)` that runs the torch module on the converted input. The gradient of `apply` with respect to the input is `torch.autograd` of the module, so a Dense-Armor pipeline can compute through it.

```python
import torch
from dense_armor.bridges.torch import from_torch

module = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(),
                             torch.nn.Linear(8, 2))
adapter = from_torch(module)
print(tuple(adapter.apply(torch.zeros(3, 4)).shape))
```

```
(3, 2)
```

## 4. What the bridge costs

`benchmarks/bridge_torch.py` measures the forward of the same model (a `Sequential` of `Dense(64,128)`, `Relu`, `Dense(128,32)`) on batch 256, float32, seven runs: the JAX model called directly, an equivalent `torch.nn.Sequential` called directly, and the bridge.

```
$ python benchmarks/bridge_torch.py
{
  "backend": "cpu",
  "torch": "2.11.0+cpu",
  "jax": "0.11.1",
  "numpy": "2.1.3",
  "batch": 256,
  "n_runs": 7,
  "jax_direct_ms":    {"median": 0.160, "iqr": 0.065},
  "torch_direct_ms":  {"median": 0.306, "iqr": 0.040},
  "bridge_ms":        {"median": 9.026, "iqr": 0.355},
  "overhead_fraction": 55.4,
  "claim_20pct_holds": false
}
```

The claim declared in the first commit (the conversion costs less than 20% of the forward) **does not hold**: the bridge forward takes 9.0 ms against 0.16 ms of the JAX forward, a factor of 56. The overhead is not the DLPack conversion (0.16 ms); it is `jax.vjp` in eager mode. Making it cheaper would need `jax.jit` of the forward, which is incompatible with the mutable `Param.value` the Dense-Armor layers use, and a functional version of the layers is a separate task. The numbers are the ones above.

## Details

### References

- PyTorch 2.11, `torch.utils.dlpack`: <https://pytorch.org/docs/2.11/torch.utils.dlpack.html>.
- PyTorch 2.11, `torch.autograd.Function`: <https://pytorch.org/docs/2.11/autograd.html#torch.autograd.Function>.
- PyTorch 2.11, `torch.nn.Module`: <https://pytorch.org/docs/2.11/generated/torch.nn.Module.html>.
- PyTorch 2.11, `torch.func`: <https://pytorch.org/docs/2.11/func.html>.
- JAX 0.11.1, `jax.dlpack`: <https://jax.readthedocs.io/en/latest/jax.dlpack.html>.

### Choices stated explicitly

- **DLPack, not numpy.** The conversion is `jax.dlpack.from_dlpack(t)` and `torch.from_dlpack(a)`, which share the buffer for a contiguous CPU tensor. A numpy round trip would copy on both sides.
- **`requires_grad=True` is detached, not refused.** `torch.utils.dlpack` refuses a tensor that requires grad (`BufferError`); the bridge detaches it and lets the caller re-attach the gradient on the torch side, which is what `TorchModule` does through its own `autograd.Function`.
- **Non-contiguous is copied.** A transposed tensor is not contiguous; the bridge calls `.contiguous()` before the DLPack export, which copies it. The alternative (an error) is worse for the reader.
- **`TorchModule.backward` uses `jax.vjp` and copies the gradient.** `torch.from_dlpack` of a JAX array shares the buffer, and JAX invalidates it when the array goes out of scope; `.clone()` on the gradient avoids the stale buffer.
- **`from_torch` does not wrap the module's parameters as Dense-Armor params.** The Dense-Armor estimator only computes through the module; the module's parameters are updated by torch. A full conversion (torch params inside the Dense-Armor `Param` object) is out of scope.

### Measured cost

The JSON is `benchmarks/bridge_torch.json`. The claim does not hold and the delivery says so: 9.0 ms against 0.16 ms, a factor of 56, on a 2-core CPU.

### Development diary

The first version of `TorchModule.forward` called `copy.deepcopy(model)` at every call, which raised a tracer leak the second time: the `Param` values held the tracers of the previous `jax.vjp`, and the next call found them invalid. The copy is now inside the function that `jax.vjp` traces, so each trace sees a fresh structure.

