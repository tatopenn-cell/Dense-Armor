# Choosing and reshaping channels

`Select`, `Discard` and `FuncTransformer` keep, drop and change the channels of a sample; their code is in `core/infrastructure/compose`. The overview of the section is on [Streams](index.md).

## 3. Choosing and reshaping channels

A sample with a joint position, a torque and a temperature; the model needs only the
first two, with the torque scaled.

```python
from dense_armor.roles import Signal
from dense_armor.core.infrastructure.compose import FuncTransformer, Select

sig = Signal(values=[0.5, 2.0, 9.0], names=["q0", "tau0", "temp"], units=["rad", "N m", "C"], t=1.0)
pipe = Select(keys=("q0", "tau0")) | FuncTransformer(lambda v: v * 2.0, keys=("tau0",))
out = pipe.transform_one(sig)
print({k: float(v) for k, v in out.items()}, out.t, out.units)
```

```
{'q0': 0.5, 'tau0': 4.0} 1.0 ['rad', 'N m']
```

`Select` keeps the named channels (`Discard` drops them), `FuncTransformer` applies a
function to some channels; joined with `|` they form a pipeline like any other in the library.
The timestamp and the units of the kept channels travel with the sample.

## API reference

::: dense_armor.core.infrastructure.compose
