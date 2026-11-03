"""The engine computes in float64 on its own, whether x64 is on or off for the rest.

test/conftest.py turns x64 on for the whole session, so each case runs in a
fresh interpreter where the setting is the one under test.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

SCRIPT = """
import warnings
warnings.filterwarnings("error", message=".*requested dtype.*")
import numpy as np
from dense_armor.core.engine import AdaptiveSignalStabilizer

x = np.random.default_rng(0).normal(size=64)
x[20] = 40.0
s = AdaptiveSignalStabilizer()
stream = s.filter_data_stream(x)
batch = s.filter_batch_scenarios_independent(np.tile(x, (3, 1)))
print(stream.dtype, batch.dtype, bool(np.isfinite(stream).all()), bool(np.isfinite(batch).all()))
"""


def _run(x64: bool) -> list[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env.pop("JAX_ENABLE_X64", None)
    if x64:
        env["JAX_ENABLE_X64"] = "1"
    out = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr[-800:]
    return out.stdout.split()


@pytest.mark.parametrize(
    "x64, dtype", [(False, "float64"), (True, "float64")], ids=["x64_off", "x64_on"]
)
def test_engine_is_float64_with_x64_off_and_on(x64, dtype):
    stream_dtype, batch_dtype, stream_ok, batch_ok = _run(x64)
    assert stream_dtype == dtype
    assert batch_dtype == dtype
    assert stream_ok == "True" and batch_ok == "True"
