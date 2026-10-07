# The Dense-Armor base

Every Dense-Armor estimator shares one small API, so you can swap one
algorithm for another without changing the code around it. The design
follows scikit-learn's parameter / clone / inspect contract
(Buitinck et al., [arXiv:1309.0238](https://arxiv.org/abs/1309.0238))
and river's one-sample-at-a-time interface
(Montiel et al., [arXiv:2012.04740](https://arxiv.org/abs/2012.04740)).

## What the base gives you

A class deriving from `dense_armor.roles.Base` automatically has:

- **Hyper-parameters from the constructor.** `get_params()` walks the
  `__init__` signature and reads each parameter from the attribute of
  the same name. `set_params(**p)` writes them back and returns `self`.
- **Cloning without learned state.** `clone()` returns a fresh instance
  built from the current parameters; `clone(new_params={...})` overrides
  some of them, recursively for nested estimators.
- **Live mutation.** `mutate({...})` changes only attributes listed in
  `_mutable_attributes` (empty by default). Anything else raises
  `ValueError`. This is how a parameter is changed while a robot moves.
- **Reproducibility.** `seed=None` at any nesting level marks the
  estimator stochastic (`_is_stochastic` is `True`).
- **Robust pickling.** `pickle.dumps/loads` round-trips cleanly and
  warns (`InconsistentVersionWarning`) when the library version changed.
- **Memory accounting.** `_raw_memory_usage` is the size of the object
  graph in bytes (`ndarray.nbytes` included); `_memory_usage` is human
  readable.
- **Readable `repr`.** Parameters one per line, floats in scientific
  notation only when they should be, `nan`/`inf` printed as such.

## Roles

One file per role under `dense_armor/roles/`:

| Role | File | Key methods |
|------|------|-------------|
| `Classifier` | `classifier.py` | `learn_one(x, y)`, `predict_proba_one(x)`, `predict_one(x)` |
| `Regressor` | `regressor.py` | `learn_one(x, y)`, `predict_one(x, return_std=False)` |
| `Transformer` | `transformer.py` | `learn_one(x)`, `transform_one(x)`, `a + b`, `a * b` |
| `AnomalyDetector` | `anomaly_detector.py` | `learn_one(x)`, `score_one(x)` (high = anomalous) |
| `DriftDetector` | `drift_detector.py` | `update(x)`, `drift_detected` (bool) |
| `ModelWrapper` | `wrapper.py` | delegates to `_wrapped_model` |

Every classifier returns `predict_one(x) is None` and
`predict_proba_one(x) == {}` until the first `learn_one`, for every
classifier — no accidental default labels.

## Pipelines

Transformers compose with `+` (union) and `*` (product); anything
composes with `|`:

```python
from dense_armor.utility.learn.online_classifiers import OnlineGaussianNB
from dense_armor.utility.anomaly.filters import HampelFilter

model = HampelFilter(radius=5, feature="v") | OnlineGaussianNB()
model.learn_one({"v": 1.0, "other": 2.0}, 0)
```

`isinstance(pipeline, Classifier)` is `True` when the last step is a
classifier.

## Writing a new estimator

Subclass a role, set hyper-parameters in `__init__`, store learned
state in attributes with a trailing underscore:

```python
from dense_armor.roles import Classifier

class RunningMean(Classifier):
    """Predict the majority label seen so far."""

    def __init__(self, prior: float = 1.0):
        self.prior = prior
        self.counts_: dict = {}

    def learn_one(self, x, y, t=None):
        self.counts_[y] = self.counts_.get(y, 0) + 1
        return self

    def predict_proba_one(self, x, t=None):
        total = sum(self.counts_.values())
        if total == 0:
            return {}
        return {c: n / total for c, n in self.counts_.items()}
```

`dense_armor.checks.check_estimator(RunningMean())` runs every check
for the classifier role.

## Robot additions

### Time base

Pass `t` (seconds) to any `learn_one` / `predict_one` / `update`. The
base keeps the last 64 intervals and exposes `dt` (median, robust to
scheduler jitter), `rate = 1 / dt`, `jitter = 1.4826 * MAD(dt)`.
`window_samples(W)` converts a window in seconds to samples. A
non-increasing `t` raises `ValueError`.

```python
for i in range(100):
    model.learn_one(x, y, t=i * 0.01)
print(round(model.dt, 4), round(model.rate, 1))
```

### Protection

`Protected(model, detector, fallback)` scores each sample first;
flagged samples are not learned and the prediction is the fallback (or
the last prediction made on an unflagged sample).

```python
from dense_armor.roles import Protected
from dense_armor.utility.anomaly.filters import HampelScorer

safe = Protected(OnlineGaussianNB(), HampelScorer(radius=5), fallback=0)
```

### State

`state_dict()` returns the learned state (trailing-underscore
attributes); `load_state_dict(s)` restores it. Separate from
`get_params()`, so a robot can be restored after a restart without
losing its hyper-parameters.

```python
snap = model.state_dict()
model.load_state_dict(snap)
```

### Batches

`learn_many` / `predict_many` / `predict_proba_many` / `score_many` /
`transform_many` loop over the `_one` methods by default; models can
override them with a vectorised version.

### Tool description

`describe()` returns a JSON-serialisable name, parameters (type,
default), methods and input/output schema — ready for an LLM agent.

## Bridges

The core has no external online-learning dependency. `dense_armor.bridges`
holds optional interop layers, each installing as an extra:

- `dense_armor.bridges.river` (`pip install dense-armor[river]`):
  our estimators and river's share `learn_one` / `predict_one` /
  `predict_proba_one`, so pipelines and evaluators interoperate
  without wrapping. If river is installed, our role classes are
  registered as virtual subclasses of river's ABCs at import time.
- `dense_armor.bridges.llm` (`pip install dense-armor[llm]`):
  an estimator whose features come from any `embed(text) -> vector`
  callable, with JSON-in / JSON-out calls built on `describe()`.
