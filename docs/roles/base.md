# The base: Root

Every tool of Dense-Armor derives from `Root`. It gives each tool the same few abilities: tell its parameters, make a fresh copy, change a setting while it runs, save what it has learned, know its own clock and describe itself.

## 1. Read and set the parameters

`get_params` returns the parameters that the constructor accepts, and `set_params` changes them and returns the object.

```python
from dense_armor.core.online.learn.online_classifiers import OnlineGaussianNB

m = OnlineGaussianNB()
print(m.get_params())
m.set_params(var_smoothing=1e-8)
print(m.var_smoothing)
```

```
{'alpha': 0.0, 'var_smoothing': 0.0, 'eps': 1e-12, 'dtype': None}
1e-08
```

A parameter is whatever `__init__` takes, and it is read from the attribute of the same name. That is why a tool must store each constructor argument unchanged.

## 2. Make a fresh copy

`clone` builds a new object from the same parameters and none of what was learned. `new_params` overrides some of them.

```python
from dense_armor.core.online.learn.online_classifiers import OnlineGaussianNB

m = OnlineGaussianNB()
m.learn_one({"a": 1.0}, 0)
c = m.clone(new_params={"alpha": 1.0})
print(c is m, c.alpha, c.predict_one({"a": 1.0}))
```

```
False 1.0 None
```

The copy has the new `alpha` and has learned nothing, so `predict_one` answers `None`. Nested tools are cloned too.

## 3. Change a setting while it runs

`mutate` changes only the attributes that the class lists in `_mutable_attributes`; the list is empty by default.

```python
from dense_armor.roles import Transformer


class Gain(Transformer):
    """Multiply the value of v by k."""

    _mutable_attributes = ("k",)

    def __init__(self, k=1.0):
        self.k = k

    def transform_one(self, x, t=None):
        return {"v": x["v"] * self.k}


g = Gain()
g.mutate({"k": 3.0})
print(g.transform_one({"v": 2.0}))
try:
    g.mutate({"zz": 1})
except ValueError as e:
    print(e)
```

```
{'v': 6.0}
Gain: 'zz' is not mutable
```

Anything that is not on the list raises `ValueError`, so a running system cannot change by accident something that it should not.

## 4. Save and restore what was learned

`state_dict` returns what the tool has learned: its attributes whose name ends with an underscore. `load_state_dict` puts it back.

```python
from dense_armor.core.signals.preprocessing.scale import StandardScaler

s = StandardScaler()
for v in (1.0, 2.0, 3.0):
    s.learn_one({"v": v})
c = StandardScaler()
c.load_state_dict(s.state_dict())
print(list(s.state_dict()), c.transform_one({"v": 3.0}) == s.transform_one({"v": 3.0}))
```

```
['stats_'] True
```

The parameters stay in `get_params`, the learned state in `state_dict`: a program can restart, build the tool from its parameters and restore the state.

## 5. Know the clock

When you pass the time `t` of each sample, the tool keeps the last 64 intervals. `dt` is their median, `rate` is `1 / dt`, `jitter` is a robust spread of the intervals, and `window_samples` converts seconds to a number of samples.

```python
from dense_armor.core.signals.preprocessing.scale import StandardScaler

s = StandardScaler()
for i in range(100):
    s.learn_one({"v": float(i)}, t=i * 0.01)
print(round(s.dt, 4), round(s.rate, 1), round(s.jitter, 6), s.window_samples(0.5))
try:
    s.learn_one({"v": 1.0}, t=0.5)
except ValueError as e:
    print(e)
```

```
0.01 100.0 0.0 50
non-increasing t: previous 0.99, got 0.5
```

A time that does not increase raises `ValueError`.

## 6. Describe the tool

`describe` returns a dict that holds the name, the parameters with their type and default, the methods and the input and output. It is plain data, ready to be written as JSON for a program or an agent that must choose a tool.

```python
from dense_armor.core.signals.preprocessing.scale import StandardScaler

d = StandardScaler().describe()
print(list(d), d["parameters"])
```

```
['name', 'module', 'parameters', 'methods', 'input_schema', 'output_schema'] {'features': {'type': 'list[str] | None', 'default': None}, 'eps': {'type': 'float', 'default': 1e-12}}
```

## Details

- `fit(X, y)` is the offline entry point of every online tool: it learns from a whole table and returns the tool. It also takes `t`, `epochs`, `shuffle` and `seed`.
- A tool that is saved with `pickle` and loaded under another version of the library gives an `InconsistentVersionWarning`.
- `repr` prints the parameters one per line. `_raw_memory_usage` counts the bytes of the object, with the arrays included.
- The code is in `dense_armor/roles/root.py`.
