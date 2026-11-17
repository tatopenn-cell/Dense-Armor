"""Tests for the linear scaling rule and the data-parallel step.

The multi-device scenarios run in one subprocess with four virtual CPU
devices: the number of devices is fixed when JAX starts, and the process of
pytest has already started it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

from dense_armor.core.deep.neural.data_parallel import linear_scaling
from dense_armor.core.deep.neural.schedules_more import LinearWarmup


def test_linear_scaling_hand_case():
    """Goyal 2017, page 2: a batch of 256 with a base of 32 scales lr by 8."""
    assert linear_scaling(0.1, 256, 32) == pytest.approx(0.8)


def test_linear_scaling_identity_at_base_batch():
    assert linear_scaling(0.05, 32, 32) == pytest.approx(0.05)


@pytest.mark.parametrize(
    "args",
    [(0.1, 256, 0), (0.1, 16, 32), (0.0, 256, 32), (-0.1, 256, 32)],
)
def test_linear_scaling_rejects_invalid_arguments(args):
    with pytest.raises(ValueError):
        linear_scaling(*args)


def test_linear_warmup_reaches_the_scaled_rate():
    """The warmup reaches the scaled learning rate at ``warmup`` epochs."""
    lr0 = 0.1
    lr_end = linear_scaling(lr0, 256, 32)
    s = LinearWarmup(lr_start=lr0, lr_end=lr_end, warmup=5)
    assert s(0.0) == pytest.approx(lr0)
    assert s(2.5) == pytest.approx((lr0 + lr_end) / 2.0)
    assert s(5.0) == pytest.approx(lr_end)
    assert s(7.0) == pytest.approx(lr_end)


_PRELUDE = """
import json
import numpy as np
import jax
import jax.numpy as jnp
from dense_armor.core.deep.neural.models import mlp
from dense_armor.core.deep.neural.losses import MSE
from dense_armor.core.deep.neural.trainer import Trainer
from dense_armor.core.deep.neural.stream import DataStream
from dense_armor.core.deep.neural.compiled import (
    make_step,
    make_step_parallel,
    _read_values,
    _init_state,
)
from dense_armor.core.deep.neural.schedules import ConstantLR
from dense_armor.core.online.optim import SGD, Adam

CHECK = (1, 10, 50, 200)
rng = np.random.default_rng(0)
X = rng.normal(size=(32, 2)).astype(np.float32)
Y = rng.normal(size=(32, 1)).astype(np.float32)
OUT = {}


def trainer(devices=None, opt=None, **kw):
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    t = Trainer(m, MSE(), opt or SGD(0.01), compile=True, devices=devices, **kw)
    return m, t


def weights(m):
    return [np.array(p.value) for p in m.params()]


def spread(a, b):
    return float(max(np.abs(x - y).max() for x, y in zip(a, b)))


def curve(devices, **kw):
    m, t = trainer(devices, **kw)
    t.fit(X, Y, epochs=200, batch_size=32, key=jax.random.PRNGKey(0), shuffle=False)
    h = t.history_["train_loss"]
    return [float(h[s - 1]) for s in CHECK]


def rel(a, b):
    return [abs(x - y) / abs(x) for x, y in zip(a, b)]
"""

_SCENARIOS = """
one = curve(None)
for n in (2, 4):
    many = curve(n)
    OUT["plain_%d" % n] = {"one": one, "many": many, "rel": rel(one, many)}
one_b = curve(None, compute_dtype="bfloat16")
for n in (2, 4):
    many = curve(n, compute_dtype="bfloat16")
    OUT["bf16_%d" % n] = {"one": one_b, "many": many, "rel": rel(one_b, many)}

Yc = Y.copy()
Yc[:16] *= 10.0


def clipped(devices, clip):
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    t = Trainer(m, MSE(), SGD(0.01), compile=True, devices=devices, clip_norm=clip)
    t.fit(X, Yc, epochs=3, batch_size=32, key=jax.random.PRNGKey(0), shuffle=False)
    return weights(m)


w1, w2, w0 = clipped(None, 0.05), clipped(2, 0.05), clipped(None, None)
OUT["clip"] = {"parallel_vs_single": spread(w1, w2), "clip_effect": spread(w1, w0)}

rng2 = np.random.default_rng(1)
Xd = rng2.normal(size=(16, 2)).astype(np.float32)
Yd = rng2.normal(size=(16, 1)).astype(np.float32)
m, t = trainer(None)
t.fit(Xd, Yd, epochs=3, batch_size=8, key=jax.random.PRNGKey(0))
OUT["default"] = {
    "loss": t.history_["train_loss"],
    "keys": list(t.history_),
    "x64": bool(jax.config.jax_enable_x64),
}


def adam_weights(devices, epochs, calls=1, **kw):
    m, t = trainer(devices, opt=Adam(0.01), **kw)
    for _ in range(calls):
        t.fit(X, Y, epochs=epochs, batch_size=16, key=jax.random.PRNGKey(0), shuffle=False)
    return weights(m)


OUT["adam"] = spread(adam_weights(None, 3), adam_weights(2, 3))
OUT["state_plain"] = spread(adam_weights(2, 2), adam_weights(2, 1, calls=2))
OUT["state_mixed"] = spread(
    adam_weights(2, 2, compute_dtype="bfloat16"),
    adam_weights(2, 1, calls=2, compute_dtype="bfloat16"),
)
OUT["adam_bf16"] = spread(
    adam_weights(None, 3, compute_dtype="bfloat16"),
    adam_weights(2, 3, compute_dtype="bfloat16"),
)

errors = {}


def attempt(name, fn):
    try:
        fn()
        errors[name] = None
    except Exception as e:
        errors[name] = type(e).__name__ + ": " + str(e)


attempt("batch", lambda: trainer(2)[1].fit(X, Y, epochs=1, batch_size=3, key=jax.random.PRNGKey(0)))
attempt("tail", lambda: trainer(2)[1].fit(X[:11], Y[:11], epochs=1, batch_size=4, key=jax.random.PRNGKey(0)))
attempt("too_many", lambda: trainer(8))
attempt("one_device", lambda: trainer(1))
attempt("repeated", lambda: trainer([0, 0]))
attempt("unknown", lambda: trainer([0, 9]))
attempt(
    "no_compile",
    lambda: Trainer(mlp([2, 4, 1], jax.random.PRNGKey(0)), MSE(), SGD(0.01), devices=2),
)
OUT["errors"] = errors

m_ids, t_ids = trainer([3, 1])
t_ids.fit(X, Y, epochs=3, batch_size=16, key=jax.random.PRNGKey(0), shuffle=False)
m_two, t_two = trainer(2)
t_two.fit(X, Y, epochs=3, batch_size=16, key=jax.random.PRNGKey(0), shuffle=False)
OUT["ids"] = {"spread": spread(weights(m_ids), weights(m_two)), "devices": t_ids.devices}

m_tail, t_tail = trainer(2)
t_tail.fit(X[:28], Y[:28], epochs=2, batch_size=16, key=jax.random.PRNGKey(0), shuffle=False)
m_tail1, t_tail1 = trainer(None)
t_tail1.fit(X[:28], Y[:28], epochs=2, batch_size=16, key=jax.random.PRNGKey(0), shuffle=False)
OUT["tail_ok"] = spread(weights(m_tail), weights(m_tail1))

m_es, t_es = trainer(2)
t_es.fit(X, Y, epochs=8, batch_size=16, key=jax.random.PRNGKey(0), X_val=X[:8], y_val=Y[:8], patience=2)
OUT["early"] = {"epochs": len(t_es.history_["train_loss"]), "best": t_es.best_epoch_}

rng3 = np.random.default_rng(3)
Xf = rng3.normal(size=(64, 2)).astype(np.float32)
Yf = rng3.normal(size=(64, 1)).astype(np.float32)
m_f, t_f = trainer(2, compute_dtype="float16", loss_scale="dynamic")
t_f.fit(Xf, Yf, epochs=25, batch_size=8, key=jax.random.PRNGKey(0))
h = t_f.history_["train_loss"]
OUT["fp16"] = {"first": h[0], "last": h[-1], "scale": t_f.history_["loss_scale"], "finite": all(np.isfinite(h))}


def overflow(shard):
    Xo = X[:16].copy()
    Yo = Y[:16].copy()
    if shard == 0:
        Xo[0:8] *= 1e30
    else:
        Xo[8:16] *= 1e30
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    opt = SGD(0.01)
    step = make_step(
        m, MSE(), opt, ConstantLR(0.01), 0.0, "l2", None,
        compute_dtype="float16", loss_scale="dynamic",
    )
    pmapped = jax.pmap(
        step.mixed_devices_inner, axis_name="devices", in_axes=(0, 0, 0, 0, None, None, None)
    )
    values = _read_values(m)
    state = _init_state(opt, values)
    n = 2
    bc = lambda v: jnp.broadcast_to(jnp.asarray(v)[None], (n,) + jnp.shape(v))
    values_d = jax.tree_util.tree_map(bc, values)
    state_d = jax.tree_util.tree_map(bc, state)
    v_out, s_out, loss_d, finite_d = pmapped(
        values_d, state_d, jnp.asarray(Xo).reshape((n, 8, 2)),
        jnp.asarray(Yo).reshape((n, 8, 1)), 0.01, 1.0, jnp.float32(1024.0),
    )
    unchanged = all(
        np.array_equal(np.asarray(values[k]), np.asarray(v_out[k][i]))
        for k in range(len(values)) for i in range(n)
    )
    equal = all(
        np.array_equal(np.asarray(v_out[k][0]), np.asarray(v_out[k][1]))
        for k in range(len(values))
    )
    return {"unchanged": bool(unchanged), "equal": bool(equal), "finite": [bool(f) for f in np.asarray(finite_d)]}


m_p = mlp([2, 4, 1], jax.random.PRNGKey(0))
step_p = make_step_parallel(
    m_p, MSE(), SGD(0.01), ConstantLR(0.01), 0.0, "l2", None, 2, devices=[3, 1]
)
vals = _read_values(m_p)
vals_d = jax.tree_util.tree_map(
    lambda v: jnp.broadcast_to(jnp.asarray(v)[None], (2,) + jnp.shape(v)), vals
)
v_out, _, _ = step_p(
    vals_d, None, jnp.asarray(X[:16]).reshape((2, 8, 2)),
    jnp.asarray(Y[:16]).reshape((2, 8, 1)), 0.0,
)
OUT["pmap_devices"] = sorted(int(d.id) for d in v_out[0].devices())

seen = []
orig_pmap = jax.pmap


def spy(*a, **k):
    seen.append(k.get("devices"))
    return orig_pmap(*a, **k)


jax.pmap = spy
m_s, t_s = trainer([3, 1])
t_s.fit(X, Y, epochs=1, batch_size=16, key=jax.random.PRNGKey(0), shuffle=False)
jax.pmap = orig_pmap
OUT["trainer_pmap_devices"] = [
    None if ds is None else [int(d.id) for d in ds] for ds in seen
]

def acc_weights(devices, micro, k, epochs=3, **kw):
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    tr = Trainer(m, MSE(), SGD(0.05), compile=True, devices=devices, accumulate=k, **kw)
    tr.fit(X, Y, epochs=epochs, batch_size=micro, key=jax.random.PRNGKey(0), shuffle=False)
    return weights(m), tr


w_big, _ = acc_weights(None, 32, 1)
w_dev, _ = acc_weights(2, 8, 4)
w_dev4, _ = acc_weights(4, 8, 4)
OUT["acc_devices"] = {"two": spread(w_big, w_dev), "four": spread(w_big, w_dev4)}

w_mixed_big, _ = acc_weights(None, 32, 1, compute_dtype="bfloat16")
w_mixed_dev, _ = acc_weights(2, 8, 4, compute_dtype="bfloat16")
OUT["acc_mixed_devices"] = spread(w_mixed_big, w_mixed_dev)


def acc_overflow(shard):
    Xo = X.copy()
    if shard == 0:
        Xo[0:4] *= 1e30
    else:
        Xo[4:8] *= 1e30
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    tr = Trainer(
        m, MSE(), SGD(0.05), compile=True, devices=2, accumulate=2,
        compute_dtype="float16", loss_scale="dynamic",
    )
    tr._scale = 1024.0
    before = [w.astype(np.float32) for w in weights(m)]
    tr.fit(Xo[:16], Y[:16], epochs=1, batch_size=8, key=jax.random.PRNGKey(0), shuffle=False)
    after = [w.astype(np.float32) for w in weights(m)]
    return {"unchanged": spread(before, after) == 0.0, "scale": tr.history_["loss_scale"]}


OUT["acc_overflow_0"] = acc_overflow(0)
OUT["acc_overflow_1"] = acc_overflow(1)

attempt(
    "generator_batch",
    lambda: trainer(2)[1].fit(
        None, None, epochs=1, batch_size=None, key=jax.random.PRNGKey(0),
        stream=DataStream(lambda: zip(X[:11], Y[:11]), 4),
    ),
)
OUT["errors"] = errors

OUT["overflow_0"] = overflow(0)
OUT["overflow_1"] = overflow(1)
print(json.dumps(OUT))
"""


def _run(code: str, n_devices: int = 4, timeout: int = 900) -> dict:
    env = dict(os.environ)
    env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={n_devices}"
    env["JAX_PLATFORMS"] = "cpu"
    r = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-1500:])
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def out():
    return _run(_PRELUDE + _SCENARIOS)


@pytest.mark.parametrize("n", [2, 4])
def test_parallel_equals_single_at_steps_1_10_50_200(out, n):
    r = out[f"plain_{n}"]
    print(f"\n  {n} devices, steps 1/10/50/200")
    print(f"  one : {r['one']}")
    print(f"  many: {r['many']}")
    print(f"  rel : {r['rel']}")
    assert max(r["rel"]) < 1e-5


@pytest.mark.parametrize("n", [2, 4])
def test_mixed_bf16_parallel_equals_single_at_steps_1_10_50_200(out, n):
    """The bound 1e-2 is about twice the unit roundoff of bfloat16 (2**-8,
    3.9e-3): the sum of the shard gradients is taken in another order."""
    r = out[f"bf16_{n}"]
    print(f"\n  bf16, {n} devices, steps 1/10/50/200")
    print(f"  one : {r['one']}")
    print(f"  many: {r['many']}")
    print(f"  rel : {r['rel']}")
    assert max(r["rel"]) < 1e-2


def test_clipping_acts_on_the_averaged_gradient(out):
    """The two shards have gradients of very different norms; clipping the
    average gives the single-device result, and the clip is active."""
    r = out["clip"]
    print(
        f"\n  parallel vs single: {r['parallel_vs_single']}, clip effect: {r['clip_effect']}"
    )
    assert r["parallel_vs_single"] < 1e-6
    assert r["clip_effect"] > 1e-3


def test_default_gives_the_values_stored_from_master(out):
    """compile=True with devices=None: the three epoch losses of the MLP
    [2, 4, 1] on 16 samples, SGD(0.01), batch 8, shuffle key 0, taken by
    running master with x64 on and with x64 off. The subprocess inherits the
    setting of the environment (the CI turns x64 on), and reports it."""
    if out["default"]["x64"]:
        stored, tol = [1.8287844438741407, 1.4960505551335324, 1.325572929081289], 1e-9
    else:
        stored, tol = [4.009725093841553, 3.231865406036377, 2.795619010925293], 1e-5
    assert out["default"]["loss"] == pytest.approx(stored, rel=tol)
    assert out["default"]["keys"] == ["train_loss", "val_loss", "lr"]


def test_adam_state_is_replicated_and_equal_to_single(out):
    assert out["adam"] < 1e-5
    assert out["adam_bf16"] < 1e-3


@pytest.mark.parametrize("key", ["state_plain", "state_mixed"])
def test_the_optimizer_state_survives_between_fit_calls(out, key):
    assert out[key] < 1e-6


@pytest.mark.parametrize(
    "key,match",
    [
        ("batch", "batch_size 3 is not divisible by devices 2"),
        ("tail", "last batch has 3 samples"),
        ("too_many", "exceeds the available count 4"),
        ("one_device", "at least 2 devices"),
        ("repeated", "repeated"),
        ("unknown", "not among the 4 available devices"),
        ("no_compile", "devices requires compile=True"),
    ],
)
def test_invalid_devices_raise(out, key, match):
    message = out["errors"][key]
    assert message is not None and message.startswith("ValueError"), message
    assert match in message


def test_a_list_of_device_ids_is_used_and_does_not_change_the_result(out):
    assert out["ids"]["devices"] == [3, 1]
    assert out["ids"]["spread"] < 1e-6


def test_the_step_runs_on_the_devices_that_were_asked_for(out):
    assert out["pmap_devices"] == [1, 3]


def test_the_trainer_asks_pmap_for_the_given_device_ids(out):
    assert out["trainer_pmap_devices"] == [[3, 1]]


def test_a_last_batch_that_the_devices_divide_is_trained_on(out):
    assert out["tail_ok"] < 1e-5


def test_early_stopping_runs_on_several_devices(out):
    assert out["early"]["best"] is not None
    assert 1 <= out["early"]["epochs"] <= 8


def test_float16_dynamic_scale_trains_on_two_devices(out):
    r = out["fp16"]
    assert r["finite"]
    assert r["last"] < 0.9 * r["first"]
    assert all(s > 0 for s in r["scale"])


@pytest.mark.parametrize("shard", [0, 1])
def test_mixed_overflow_on_one_device_skips_all(out, shard):
    r = out[f"overflow_{shard}"]
    print(f"\n  overflow on shard {shard}: {r}")
    assert r["finite"] == [False, False]
    assert r["unchanged"]
    assert r["equal"]


@pytest.mark.parametrize("key", ["two", "four"])
def test_accumulation_over_devices_equals_one_large_batch(out, key):
    """4 micro-batches of 8, each split over 2 or 4 devices, against one batch of 32."""
    assert out["acc_devices"][key] < 1e-5


def test_accumulation_with_devices_and_bfloat16_follows_the_single_device_batch(out):
    assert out["acc_mixed_devices"] < 5e-2


@pytest.mark.parametrize("shard", [0, 1])
def test_an_overflow_on_one_device_skips_the_whole_accumulated_group(out, shard):
    r = out[f"acc_overflow_{shard}"]
    assert r["unchanged"]
    assert r["scale"] == [512.0]


def test_a_generator_batch_that_the_devices_do_not_divide_raises(out):
    message = out["errors"]["generator_batch"]
    assert message is not None and message.startswith("ValueError")
    assert "a batch of 3 samples is not divisible by devices 2" in message
