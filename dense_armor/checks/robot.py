"""Robot-native checks that extend :func:`dense_armor.checks.check_estimator`.

Each check is a function that takes an estimator and returns silently if
the estimator does not have the feature being tested. This keeps the
extension non-invasive: a pure statistics estimator passes untouched,
while a robot estimator with ``step``, ``budget_s``, ``save``/``restore``
gets the new checks for free.

The checks use only duck typing — no import of :mod:`dense_armor.roles`
— so the module has no circular dependency with the robot layer.
"""
import inspect
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


def _learn(est: Any, sig: Any) -> None:
    """Call ``learn_one`` with ``y=0.0`` when the signature requires it.

    An anomaly detector's second parameter is the timestamp ``t``: it
    receives the sample alone.
    """
    learn = est.learn_one
    names = list(inspect.signature(learn).parameters)
    if len(names) > 1 and names[1] == "t":
        learn(sig)
        return
    try:
        learn(sig, 0.0)
    except TypeError:
        learn(sig)


def check_step_is_pure(est: Any) -> None:
    """The ``step`` method must be pure: same input, same output."""
    step = getattr(est, "step", None)
    init = getattr(est, "init_state", None)
    if step is None or init is None:
        return
    import jax
    import jax.numpy as jnp

    state = init()
    v = jnp.array(1.5)
    s1, o1 = step(state, v)
    s2, o2 = step(state, v)
    flat_o1 = jax.tree_util.tree_leaves(o1)
    flat_o2 = jax.tree_util.tree_leaves(o2)
    assert len(flat_o1) == len(flat_o2)
    for a, b in zip(flat_o1, flat_o2):
        assert float(a) == float(b), "step is not pure: outputs differ"
    flat_s1 = jax.tree_util.tree_leaves(s1)
    flat_s2 = jax.tree_util.tree_leaves(s2)
    assert len(flat_s1) == len(flat_s2)
    for a, b in zip(flat_s1, flat_s2):
        assert float(a) == float(b), "step is not pure: states differ"


def check_step_is_jittable(est: Any) -> None:
    """The ``step`` method must be usable under ``jax.jit``."""
    step = getattr(est, "step", None)
    init = getattr(est, "init_state", None)
    if step is None or init is None:
        return
    import jax
    import jax.numpy as jnp

    state = init()
    v = jnp.array(2.0)
    _, o_eager = step(state, v)
    _, o_jit = jax.jit(step)(state, v)
    assert float(o_eager) == float(o_jit), "step eager vs jitted differ"


def check_p99_within_budget(est: Any) -> None:
    """If the estimator declares ``budget_s``, p99 must fit it."""
    budget = getattr(est, "budget_s", None)
    if budget is None:
        return
    # The skill sets a hard floor of 1e-3 s on `budget_s`: a smaller
    # declared budget is a contract violation, not a measurement. Reject
    # it before measuring, so the check is deterministic.
    if budget < 1e-3:
        raise AssertionError(
            f"declared budget {budget:.3e} s is below the 1e-3 s "
            f"lower bound"
        )
    learn = getattr(est, "learn_one", None)
    if learn is None:
        return
    import jax.numpy as jnp

    from dense_armor.roles.signal import Signal

    n_warmup = 64
    n_measured = 200
    n_rounds = 3
    signals = [
        Signal(
            values=jnp.array([float(i)]),
            names=["x"],
            units=[""],
            t=i * 0.01,
        )
        for i in range(n_warmup + n_rounds * n_measured)
    ]
    for sig in signals[:n_warmup]:
        _learn(est, sig)
    # CI runners are slower and more jittery than the machine the
    # budgets are measured on. With 200 measurements per round, the 99th
    # percentile is the second-largest value, not the maximum: a single
    # JAX compile, a GC pause or a preemption does not decide it. When
    # the first round is over the limit, the measurement is repeated on
    # new samples, up to three rounds, and the smallest p99 counts: a
    # burst of preemptions in one round does not decide it, while an
    # estimator that regresses by 2x or more is slow in every round. The
    # tolerance of half the budget stays.
    tolerance = max(1e-4, 0.5 * budget)
    p99 = float("inf")
    for k in range(n_rounds):
        start = n_warmup + k * n_measured
        latencies: list[float] = []
        for sig in signals[start : start + n_measured]:
            t0 = time.perf_counter()
            _learn(est, sig)
            latencies.append(time.perf_counter() - t0)
        p99 = min(p99, float(np.percentile(latencies, 99)))
        if p99 <= budget + tolerance:
            break
    assert p99 <= budget + tolerance, (
        f"p99 latency {p99:.3e} s exceeds declared budget "
        f"{budget:.3e} s (tolerance {tolerance:.3e} s)"
    )


def check_memory_growth_bounded(est: Any) -> None:
    """Declared ``memory_class`` must match the observed growth."""
    mc = getattr(est, "memory_class", None)
    if mc not in ("O(1)", "O(window)"):
        return
    learn = getattr(est, "learn_one", None)
    if learn is None:
        return
    if getattr(est, "_raw_memory_usage", None) is None:
        return
    import jax.numpy as jnp

    from dense_armor.roles.signal import Signal

    def _sig(i: int) -> Any:
        return Signal(values=jnp.array([float(i)]), names=["x"], units=[""])

    n_fill = 32
    if mc == "O(window)":
        window = getattr(est, "window", None)
        if not isinstance(window, int) or isinstance(window, bool) or window < 1:
            return
        n_fill = max(32, window + 8)
    for i in range(n_fill):
        _learn(est, _sig(i))
    start = int(est._raw_memory_usage)
    for i in range(n_fill, n_fill + 32):
        _learn(est, _sig(i))
    end = int(est._raw_memory_usage)
    slack = 8192
    assert end - start <= slack, (
        f"declared {mc} but memory grew by {end - start} bytes "
        f"after {n_fill} samples"
    )


def check_estimate_variance_non_negative(est: Any) -> None:
    """If ``predict_one`` accepts ``return_estimate=True``, variance ≥ 0."""
    predict = getattr(est, "predict_one", None)
    if predict is None:
        return
    import jax.numpy as jnp

    from dense_armor.roles.signal import Signal

    sig = Signal(values=jnp.array([0.0]), names=["x"], units=[""])
    try:
        out = predict(sig, return_estimate=True)
    except (TypeError, NotImplementedError):
        return
    var = getattr(out, "var", None)
    if var is None:
        return
    assert var >= 0.0, f"variance is negative: {var}"


def check_schema_json(est: Any) -> None:
    """If ``schema`` is available, it must be JSON-serialisable."""
    try:
        from dense_armor.roles.agents import schema
    except ImportError:
        return
    s = schema(est)
    json.dumps(s)


def check_health_is_reachable(est: Any) -> None:
    """If ``health`` is exposed, it must be a non-empty string."""
    health = getattr(est, "health", None)
    if health is None:
        return
    val = getattr(health, "value", health)
    assert isinstance(val, str) and val, "health returned an empty value"


def check_checkpoint_roundtrip(est: Any) -> None:
    """If ``save``/``restore`` exist, a roundtrip must work and a
    corrupted checkpoint must be refused."""
    save = getattr(est, "save", None)
    restore = getattr(est, "restore", None)
    learn = getattr(est, "learn_one", None)
    clone = getattr(est, "clone", None)
    if save is None or restore is None or learn is None or clone is None:
        return
    for i in range(16):
        try:
            learn({"x": float(i)}, float(i))
        except TypeError:
            try:
                learn({"x": float(i)})
            except TypeError:
                return
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cp.pkl"
        save(p)
        fresh = clone()
        fresh.restore(p)
        p.write_bytes(p.read_bytes()[:-8] + b"\x00" * 8)
        rejected = False
        try:
            fresh.restore(p)
        except Exception:  # noqa: BLE001
            rejected = True
        assert rejected, "corrupted checkpoint was accepted by restore()"

