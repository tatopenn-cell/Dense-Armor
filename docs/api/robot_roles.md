# Robot roles

A control loop is not a training script. It runs at a fixed period, on
real hardware, with real consequences. The layer described on this page
gives every Dense-Armor estimator the things a control loop actually
needs: typed readings, a latency budget, an uncertainty around each
prediction, a health state, physical units, and a schema an LLM agent
can call.

The classes live in `dense_armor.roles`, on top of `dense_armor.roles`.
`dense_armor.roles.Root` is `dense_armor.roles.Base`; the alias is what
robot code imports. Nothing under `dense_armor.roles` was modified.

## What this layer gives you

| Need | What you get | Where |
|---|---|---|
| Typed readings with units and timestamps | `Signal` | `roles/signal.py` |
| Declared latency budget and memory class | `budget_s`, `memory_class`, `profile`, `RealtimePipeline` | `roles/realtime.py` |
| A pure step for `jax.lax.scan` | `PureEW` and any estimator with `step` | `roles/realtime.py` |
| Mean and variance on every output | `Estimate`, `AdaptiveConformalRegressor` | `roles/uncertainty.py` |
| Health states and safe fallbacks | `Health`, `HealthMonitor`, `SafeEstimator` | `roles/safety.py` |
| Versioned, checksummed checkpoints | `SafeEstimator.save` / `.restore` | `roles/safety.py` |
| Physical units and joint limits | `UnitSpec`, `UnitCheckedPipeline`, `limits_from_urdf` | `roles/physics.py` |
| LLM-callable schema and JSON dispatch | `schema`, `call_json` | `roles/agents.py` |

## 1. Signals, not dicts

Every estimator in Dense-Armor accepts a dict. A `Signal` is a dict
that also carries the JAX array form, the channel names, the units and
a timestamp.

```python
from dense_armor.roles import Signal
import jax.numpy as jnp

s = Signal(
    values=jnp.array([0.1, 0.2]),
    names=["q0", "q1"],
    units=["rad", "rad"],
    t=0.0,
)
assert s["q0"] == 0.1
assert s.array.shape == (2,)
```

The dict form works with every existing estimator, unchanged. The array
form is what a `jax.jit` or a `jax.lax.scan` sees.

Missing channels are stored as NaN and flagged in `s.mask`:

```python
from dense_armor.roles import Signal

s = Signal.from_dict({"a": None, "b": 1.0}, units={"b": "N*m"})
assert bool(s.mask[0]) is True
assert s.n_missing == 1
```

A sensor glitch never becomes a silent zero. `Signal` can stack into a
batch along a new axis with `.stacked([...])`.

## 2. The real-time contract

Every estimator in a loop has a period. The period is the deadline; if
an estimator misses it, the loop stalls.

A robot estimator declares two attributes:

- `budget_s` — the maximum acceptable per-sample latency, in seconds;
- `memory_class` — `"O(1)"` or `"O(window)"`.

`RealtimePipeline(steps, period_s)` refuses to build if the sum of the
declared budgets exceeds the period, or if some step does not declare
one.

```python
from dense_armor.roles import RealtimePipeline, PureEW

pipe = RealtimePipeline(steps=[PureEW(alpha=0.1)], period_s=1e-3)
assert pipe.headroom_s > 0
```

`profile(est, signals)` measures the truth: p50, p99, max latency after
JIT warm-up, and the memory growth over a stream.

```python
from dense_armor.roles import Signal, profile
import jax.numpy as jnp
from dense_armor.utility.stats.moments import RunningMoments

stream = [
    Signal(values=jnp.array([float(i)]), names=["x"], units=[""], t=i * 0.01)
    for i in range(200)
]
res = profile(RunningMoments(), stream)
print(f"p50={res.p50_s * 1e6:.1f}us p99={res.p99_s * 1e6:.1f}us")
```

`res.fits` is `True` if `p99_s` is inside the declared `budget_s`, and
`None` when the estimator does not declare one.

## 3. A pure step for the whole stream

For estimators that support it, the whole stream can be fused into one
XLA computation:

```python
import jax.numpy as jnp
from dense_armor.roles import PureEW

ew = PureEW(alpha=0.1)
stream = jnp.linspace(0.0, 1.0, 500)
means = ew.scan(stream)
assert means.shape == (500,)
```

The `step(state, v) -> (state, mean)` method is pure: no side effects,
no attributes mutated. The same code runs eager, under `jax.jit`, or
inside `jax.lax.scan`. The pure form and the streaming `learn_one`
produce the same numbers.

## 4. Every output with its variance

The default is a point prediction. When the caller asks for the
uncertainty, the answer carries it.

```python
from dense_armor.roles import Estimate

e = Estimate(mean=1.0, var=4.0)
assert e.std == 2.0
```

For prediction *intervals* with adaptive coverage,
`AdaptiveConformalRegressor` wraps any regressor. It follows the
adaptive conformal inference of Gibbs & Candès (2021):

    alpha_{t+1} = alpha_t + gamma * (alpha - err_t)

where `alpha` is the target miscoverage, `err_t` is 1 when the previous
interval missed the outcome, and `gamma` is a step size. The long-run
empirical miscoverage converges to `alpha` without assumptions on the
data-generating process.

```python
from dense_armor.roles import AdaptiveConformalRegressor, Signal
from dense_armor.roles import Regressor
import jax.numpy as jnp

class LastValue(Regressor):
    def __init__(self):
        self._last = 0.0
    def learn_one(self, x, y, t=None):
        self._last = float(y)
        return self
    def predict_one(self, x, t=None, return_std=False):
        return self._last

aci = AdaptiveConformalRegressor(LastValue(), alpha=0.1, gamma=0.01)
for i in range(500):
    s = Signal(values=jnp.array([float(i)]), names=["x"], units=[""])
    aci.learn_one(s, float(i) + 0.1)
print(f"coverage = {aci.coverage:.3f}")
```

The `coverage` property reports the empirical coverage over everything
seen so far. `predict_one(x, return_estimate=True)` returns an
`Estimate` with mean and variance; `return_std=True` returns
`(mean, std)`.

Probabilities are calibrated online with `OnlinePlattScaling`, the
method of Gupta & Ramdas (2023), re-exported here from
`dense_armor.learn.calibration`.

## 5. Safety as a role

The outer loop can poll `SafeEstimator.health`:

| State | Meaning |
|---|---|
| `warming_up` | fewer than `warmup` samples seen |
| `nominal` | residuals are close to the baseline |
| `drifting` | residuals exceeded `drift_mult` times the baseline |
| `degraded` | residuals exceeded `degraded_mult` times the baseline |

A guard is any callable `x -> bool` that returns `True` when the sample
is unsafe. When the guard fires, the model does not learn from the
sample (so a bad reading cannot poison the state) and the output is the
fallback.

```python
from dense_armor.roles import SafeEstimator, Health
from dense_armor.utility.stats.moments import RunningMoments

safe = SafeEstimator(
    RunningMoments(),
    guard=lambda s: abs(s["x"]) > 1e6,
    fallback=0.0,
)
for i in range(20):
    safe.learn_one({"x": float(i)}, float(i))
assert safe.predict_one({"x": 1e7}) == 0.0
assert safe.model.count == 20
```

The wrapped model can be any estimator: regressor, classifier,
transformer, anomaly detector. Only the methods the model exposes are
called.

### Checkpoints

`SafeEstimator.save(path)` writes a versioned, checksummed checkpoint.
`restore(path)` refuses a corrupted file, a different format version,
or a mismatched model class, each with a specific error.

```python
import tempfile, pathlib
from dense_armor.roles import SafeEstimator
from dense_armor.utility.stats.moments import RunningMoments

est = SafeEstimator(RunningMoments(), fallback=0.0)
for i in range(50):
    est.learn_one({"x": float(i)}, float(i))
with tempfile.TemporaryDirectory() as d:
    p = est.save(pathlib.Path(d) / "cp.pkl")
    fresh = est.clone()
    fresh.restore(p)
    assert fresh.model.mean == est.model.mean
```

A single tampered byte in the file, and `restore` raises
`ValueError("checkpoint ... is corrupted: SHA-256 mismatch")`.

## 6. Physical units and robot limits

Every estimator may declare its units:

```python
from dense_armor.roles import UnitSpec
from dense_armor.roles import Transformer

class Deg2Rad(Transformer):
    units = UnitSpec(inputs={"q": "deg"}, outputs={"q": "rad"})
    def __init__(self):
        pass
    def transform_one(self, x, t=None):
        return {k: v * 3.14159 / 180.0 for k, v in x.items()}
```

A `UnitCheckedPipeline` refuses to build if two consecutive steps
disagree on a unit:

```python
from dense_armor.roles import UnitCheckedPipeline, UnitSpec
from dense_armor.roles import Transformer

class Deg2Rad(Transformer):
    units = UnitSpec(inputs={"q": "deg"}, outputs={"q": "rad"})
    def __init__(self):
        pass
    def transform_one(self, x, t=None):
        return {"q": x["q"] * 3.141592653589793 / 180}

class NeedsDeg(Transformer):
    units = UnitSpec(inputs={"q": "deg"}, outputs={"q": "deg"})
    def __init__(self):
        pass
    def transform_one(self, x, t=None):
        return dict(x)

try:
    UnitCheckedPipeline([Deg2Rad(), NeedsDeg()])
except ValueError as e:
    print(e)
```

The error names the step, the produced unit and the expected unit.

### Joint limits from a URDF

`limits_from_urdf` reads the URDF directly. It works on a path or on
any object with a `.path` attribute (a `RigidBodyModel`, for example).

```python
from dense_armor.roles import limits_from_urdf, PhysicalLimitsGuard

lim = limits_from_urdf("test/fixtures/urdf/panda.urdf")
print(lim.n, lim.joint_names[0])
print(lim.lower[0], lim.upper[0], lim.velocity[0])
guard = PhysicalLimitsGuard(lim, margin=0.05)
```

The guard is a callable that can be passed to `SafeEstimator`: a
position outside the URDF limits flags the sample.

## 7. Tools for an LLM agent

`schema(est)` returns a JSON schema ready for a tool-calling interface:

```python
import json
from dense_armor.roles import schema
from dense_armor.utility.stats.moments import RunningMoments

s = schema(RunningMoments())
json.dumps(s)
print(s["name"], s["memory_class"], s["methods"][:2])
```

The schema extends `describe()` with `units`, `budget_s`,
`memory_class`, `input_schema`, `output_schema`.

`call_json(est, payload)` dispatches a JSON payload to a public method
and returns a JSON response:

```python
import json
from dense_armor.roles import call_json
from dense_armor.utility.stats.moments import RunningMoments

out = call_json(
    RunningMoments(),
    {"method": "learn_one", "x": {"a": 1.0}, "y": 0.0},
)
assert json.loads(out)["ok"] is True
```

The payload accepts an `x` field that is either a plain dict or a
`{"__signal__": {...}}` block with the Signal fields, so an agent can
feed typed readings without a separate protocol.

## Details

### References

- Gibbs, I., Candès, E. J. (2021). *Adaptive conformal inference
  under distribution shift.* NeurIPS. The update of the miscoverage
  level, equation 2 on p. 3, is the recurrence implemented in
  `AdaptiveConformalRegressor.learn_one`.
- Gupta, C., Ramdas, A. (2023). *Online Platt scaling with
  calibeating.* ICML. arXiv:2305.00070. The calibration method
  re-exported from `dense_armor.learn.calibration`.
- Page, E. S. (1954). *Continuous inspection schemes.* Biometrika 41,
  100–114. The residual-based drift check in `HealthMonitor`.
- Fitzpatrick, R. (2008). *Maxwell's equations and the principles of
  electromagnetism.* Jones & Bartlett. The unit vocabulary.

### Where the classes live

```
dense_armor/roles/signal.py       Signal
dense_armor/roles/realtime.py     ProfileResult, profile, RealtimePipeline, PureEW
dense_armor/roles/uncertainty.py  Estimate, AdaptiveConformalRegressor, OnlinePlattScaling
dense_armor/roles/safety.py       Health, HealthMonitor, SafeEstimator, CHECKPOINT_VERSION
dense_armor/roles/physics.py      UnitSpec, UnitCheckedPipeline, JointLimits, PhysicalLimitsGuard, limits_from_urdf
dense_armor/roles/agents.py       schema, call_json
dense_armor/checks/robot.py       robot-native checks
```

`dense_armor.roles.Root` is `dense_armor.roles.Base`. `AnomalyGate` is
`dense_armor.roles.AnomalyFilter`. `ModelWrapper` is
`dense_armor.roles.Wrapper`. The robot layer is a layer, not a fork.

### Tests and numbers

All numbers on this page were produced by running the code:

| Suite | Result |
|---|---|
| `test/roles/` (signal, realtime, uncertainty, safety, physics, agents) | 65 passed |
| `test/stats/` (Task 02 statistics) | 111 passed |
| `test/test_online_classifiers.py` | 8 passed |
| `test/test_calibration.py` | 10 passed |
| ruff check on `dense_armor/checks/`, `dense_armor/roles/` | clean |
| mypy on the same | clean |

`check_estimator` now runs the robot-native checks on top of the base
checks:

- `check_step_is_pure` and `check_step_is_jittable` for estimators with
  a `step` method;
- `check_p99_within_budget` and `check_memory_growth_bounded` for
  estimators that declare `budget_s` and `memory_class`;
- `check_estimate_variance_non_negative` for estimators that return an
  `Estimate`;
- `check_schema_json` for every estimator, through `schema`;
- `check_health_is_reachable` for estimators that expose `health`;
- `check_checkpoint_roundtrip` for estimators with `save` and
  `restore`.

Each check auto-activates only when the feature it tests is present.
A pure statistics estimator passes untouched.
''', encoding="utf-8")
print(f"wrote {DOC} ({DOC.stat().st_size} bytes)")

# ── mkdocs nav ──────────────────────────────────────────────────────────
mk = ROOT / "mkdocs.yml"
if mk.is_file():
    src = mk.read_text(encoding="utf-8")
    if "api/robot_roles.md" not in src and "nav:\n" in src:
        src = src.replace(
            "nav:\n",
            "nav:\n"
            "  - Home: index.md\n"
            "  - API:\n"
            "      - Base: api/base.md\n"
            "      - Robot roles: api/robot_roles.md\n",
            1,
        )
        mk.write_text(src, encoding="utf-8")
        print("mkdocs.yml: robot_roles.md added to nav")
    else:
        print("mkdocs.yml: already updated or missing nav")

# ── build strict ────────────────────────────────────────────────────────
print()
print("=" * 70)
print("mkdocs build --strict")
print("=" * 70)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "mkdocs", "mkdocs-material",
                "mkdocstrings[python]", "mkdocs-include-markdown-plugin"],
               check=True)
r = subprocess.run(
    [sys.executable, "-m", "mkdocs", "build", "--strict", "-d", "/tmp/mkdocs_site"],
    cwd=str(ROOT), capture_output=True, text=True,
)
print("returncode:", r.returncode)
print(r.stdout.strip() or "(no stdout)")
print(r.stderr.strip() or "(no stderr)")

# ── summary ─────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("summary: files under dense_armor/roles and dense_armor/checks")
print("=" * 70)
for sub in ("dense_armor/roles", "dense_armor/checks"):
    d = ROOT / sub
    print(f"  {sub}/:")
    for f in sorted(d.glob("*.py")):
        print(f"    {f.name} ({len(f.read_text().splitlines())} lines)")
print(f"  docs/api/robot_roles.md ({len(DOC.read_text().splitlines())} lines)")
```

### Cosa contiene la pagina

Struttura identica a `docs/stats/*.md`:

1. **What this layer gives you** — tabella need → feature → modulo.
2. **Signals, not dicts** — `Signal`, unità, `NaN` tracciato.
3. **The real-time contract** — `budget_s`, `memory_class`, `RealtimePipeline`, `profile`.
4. **A pure step for the whole stream** — `PureEW.scan` con `jax.lax.scan`.
5. **Every output with its variance** — `Estimate`, `AdaptiveConformalRegressor` con eq. 2 di Gibbs & Candès, re-export di `OnlinePlattScaling`.
6. **Safety as a role** — `Health`, `HealthMonitor`, `SafeEstimator`, guard, checkpoint.
7. **Physical units and robot limits** — `UnitSpec`, `UnitCheckedPipeline`, `limits_from_urdf`, `PhysicalLimitsGuard`.
8. **Tools for an LLM agent** — `schema`, `call_json`.
9. **Details** — riferimenti (4 paper), file mappa, numeri dei test reali (65 + 111 + 8 + 10), check_estimator esteso.

