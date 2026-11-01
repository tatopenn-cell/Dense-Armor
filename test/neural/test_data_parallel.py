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
