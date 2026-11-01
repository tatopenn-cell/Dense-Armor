"""Tests for the linear scaling rule and the data-parallel step."""

from __future__ import annotations

import pytest

from dense_armor.utility.neural.data_parallel import linear_scaling
from dense_armor.utility.neural.schedules_more import LinearWarmup


def test_linear_scaling_hand_case():
    """Goyal 2017, page 2: a batch of 256 with a base of 32 scales lr by 8."""
    assert linear_scaling(0.1, 256, 32) == pytest.approx(0.8)


def test_linear_scaling_identity_at_base_batch():
    assert linear_scaling(0.05, 32, 32) == pytest.approx(0.05)


def test_linear_scaling_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        linear_scaling(0.1, 256, 0)
    with pytest.raises(ValueError):
        linear_scaling(0.1, 16, 32)
    with pytest.raises(ValueError):
        linear_scaling(0.0, 256, 32)
    with pytest.raises(ValueError):
        linear_scaling(-0.1, 256, 32)


def test_linear_warmup_reaches_the_scaled_rate():
    """The warmup reaches the scaled learning rate at ``warmup`` epochs."""
    lr0 = 0.1
    lr_end = linear_scaling(lr0, 256, 32)
    s = LinearWarmup(lr_start=lr0, lr_end=lr_end, warmup=5)
    assert s(0.0) == pytest.approx(lr0)
    assert s(2.5) == pytest.approx((lr0 + lr_end) / 2.0)
    assert s(5.0) == pytest.approx(lr_end)
    assert s(7.0) == pytest.approx(lr_end)


import os
import subprocess
import sys


def _run_in_subprocess(code: str, n_devices: int, timeout: int = 120) -> str:
    env = dict(os.environ)
    env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={n_devices}"
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-800:])
    return r.stdout


_EQUIV = """
import numpy as np
import jax
from dense_armor.utility.neural.models import mlp
from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.optim import SGD
rng = np.random.default_rng(0)
X = rng.normal(size=(32, 2)).astype(np.float32)
y = rng.normal(size=(32, 1)).astype(np.float32)

def run(devices):
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    t = Trainer(m, MSE(), SGD(0.01), compile=True, devices=devices)
    t.fit(X, y, epochs=5, batch_size=8, key=jax.random.PRNGKey(0))
    return t.history_["train_loss"]

one = run(None)
many = run({n})
print("one:", [round(float(v), 8) for v in one])
print("many:", [round(float(v), 8) for v in many])
print("max_rel:", max(abs(a - b) / max(abs(a), 1e-9) for a, b in zip(one, many)))
"""


def test_parallel_equals_single_at_2_and_4_devices():
    for n in (2, 4):
        out = _run_in_subprocess(_EQUIV.format(n=n), n)
        print(f"n_devices={n}")
        for line in out.strip().splitlines():
            print("  " + line)
        rel = float(out.strip().split("max_rel:")[1].strip())
        assert rel < 1e-5, f"n={n}: rel {rel}"


def test_parallel_rejects_undivisible_batch():
    code = """
import numpy as np
import jax
from dense_armor.utility.neural.models import mlp
from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.optim import SGD
m = mlp([2, 4, 1], jax.random.PRNGKey(0))
t = Trainer(m, MSE(), SGD(0.01), compile=True, devices=2)
X = np.zeros((16, 2), dtype=np.float32)
y = np.zeros((16, 1), dtype=np.float32)
try:
    t.fit(X, y, epochs=1, batch_size=3, key=jax.random.PRNGKey(0))
    print("NO_ERROR")
except ValueError as e:
    print("OK", str(e)[:60])
"""
    out = _run_in_subprocess(code, 2)
    print(out.strip())
    assert "OK" in out


def test_parallel_rejects_too_many_devices():
    code = """
import jax
from dense_armor.utility.neural.models import mlp
from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.optim import SGD
m = mlp([2, 4, 1], jax.random.PRNGKey(0))
try:
    Trainer(m, MSE(), SGD(0.01), compile=True, devices=8)
    print("NO_ERROR")
except ValueError as e:
    print("OK", str(e)[:80])
"""
    out = _run_in_subprocess(code, 2)
    print(out.strip())
    assert "OK" in out


def test_parallel_rejects_devices_without_compile():
    code = """
import jax
from dense_armor.utility.neural.models import mlp
from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.optim import SGD
m = mlp([2, 4, 1], jax.random.PRNGKey(0))
try:
    Trainer(m, MSE(), SGD(0.01), compile=False, devices=2)
    print("NO_ERROR")
except ValueError as e:
    print("OK", str(e)[:60])
"""
    out = _run_in_subprocess(code, 2)
    print(out.strip())
    assert "OK" in out


_WEIGHT_DIFF = """
import numpy as np
import jax
from dense_armor.utility.neural.models import mlp
from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.optim import SGD
rng = np.random.default_rng(0)
X = rng.normal(size=(32, 2)).astype(np.float32)
y = rng.normal(size=(32, 1)).astype(np.float32)

def run(devices):
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    t = Trainer(m, MSE(), SGD(0.01), compile=True, devices=devices)
    t.fit(X, y, epochs=3, batch_size=8, key=jax.random.PRNGKey(0))
    return [np.array(p.value) for p in m.params()]

a = run(None)
b = run(2)
print("max_diff:", max(float(np.max(np.abs(x - y))) for x, y in zip(a, b)))
"""


def test_parallel_master_weights_match_single():
    out = _run_in_subprocess(_WEIGHT_DIFF, 2)
    print(out.strip())
    diff = float(out.strip().split("max_diff:")[1].strip())
    assert diff < 1e-4, f"weight diff {diff}"


# ---------------------------------------------------------------------
# F: the mixed parallel step (bfloat16/float16 + several devices)
# ---------------------------------------------------------------------


_BF16_EQUIV = """
import numpy as np, jax
from dense_armor.utility.neural.models import mlp
from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.optim import SGD
rng = np.random.default_rng(0)
X = rng.normal(size=(32, 2)).astype(np.float32)
y = rng.normal(size=(32, 1)).astype(np.float32)

def run(devices):
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    t = Trainer(m, MSE(), SGD(0.01), compile=True, devices=devices,
                compute_dtype="bfloat16")
    t.fit(X, y, epochs=50, batch_size=8, key=jax.random.PRNGKey(0))
    loss = t.history_["train_loss"]
    return loss[0], loss[2], loss[12], loss[49], loss[-1]

one = run(None)
many = run(@N@)
for i, s in enumerate([1, 10, 50, 200]):
    rel = abs(one[i] - many[i]) / max(abs(one[i]), 1e-9)
    print("step " + str(s) + ": 1dev=" + str(round(one[i], 8))
          + " @N@dev=" + str(round(many[i], 8))
          + " rel=" + str(round(rel, 6)))
"""


def test_mixed_parallel_bf16_matches_single():
    for n in (2, 4):
        code = _BF16_EQUIV.replace("@N@", str(n))
        out = _run_in_subprocess(code, n)
        print(f"n_devices={n}")
        for line in out.strip().splitlines():
            print("  " + line)
        for line in out.strip().splitlines():
            if "rel=" in line:
                rel = float(line.split("rel=")[1])
                assert rel < 1e-3, line


_OVERFLOW_SHARD = """
import numpy as np, jax
import jax.numpy as jnp
from dense_armor.utility.neural.models import mlp
from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.neural.compiled import (
    make_step, _read_values as read_values, _init_state as init_state,
)
from dense_armor.utility.optim import SGD
from dense_armor.utility.neural.schedules import ConstantLR

rng = np.random.default_rng(0)
X = rng.normal(size=(16, 2)).astype(np.float32)
y = rng.normal(size=(16, 1)).astype(np.float32)
if SHARD == 0:
    X[0:8] *= 1e30
else:
    X[8:16] *= 1e30

m = mlp([2, 4, 1], jax.random.PRNGKey(0))
opt = SGD(0.01)
sched = ConstantLR(0.01)
step = make_step(m, MSE(), opt, sched, 0.0, "l2", None,
                 compute_dtype="float16", loss_scale="dynamic")
pmapped = jax.pmap(step.mixed_devices_inner, axis_name="devices",
                   in_axes=(0, 0, 0, 0, None, None, None))

values = read_values(m)
state = init_state(opt, values)
n = 2
values_d = jax.tree_util.tree_map(
    lambda v: jnp.broadcast_to(jnp.asarray(v)[None], (n,) + jnp.shape(v)),
    values,
)
state_d = jax.tree_util.tree_map(
    lambda s: jnp.broadcast_to(jnp.asarray(s)[None], (n,) + jnp.shape(s)),
    state,
)
x_s = jnp.asarray(X).reshape((n, 8, 2))
y_s = jnp.asarray(y).reshape((n, 8, 1))
v_out, s_out, loss_d, finite_d = pmapped(
    values_d, state_d, x_s, y_s, 0.01, 1.0, jnp.float32(1024.0)
)

for k, v in enumerate(values):
    ref = jnp.asarray(v)
    for i in range(n):
        got = v_out[k][i]
        assert np.array_equal(np.asarray(ref), np.asarray(got)), (
            "device " + str(i) + " param " + str(k) + " changed"
        )
    for i in range(1, n):
        assert np.array_equal(np.asarray(v_out[k][0]), np.asarray(v_out[k][i])), (
            "param " + str(k) + " differs between devices"
        )
fin = np.asarray(finite_d)
assert fin[0] == fin[1], "finite flag differs between devices"
assert not bool(fin[0]), "step should be skipped"
print("OK shard=" + str(SHARD) + " finite=" + str(bool(fin[0])))
"""


import pytest as _pytest


@_pytest.mark.parametrize("shard", [0, 1])
def test_mixed_overflow_on_one_device_skips_all(shard):
    code = "SHARD=" + str(shard) + "\n" + _OVERFLOW_SHARD
    out = _run_in_subprocess(code, 2)
    print(out.strip())
    assert "OK shard=" + str(shard) in out
    assert "finite=False" in out
