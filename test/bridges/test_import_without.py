"""The torch and TensorFlow bridges raise a clear error when the library is missing.

``import dense_armor`` must work without the other library, and the bridge
module must raise a ``ModuleNotFoundError`` that names the extra. The
library is blocked in a subprocess, so the test passes whether or not it is
installed on the machine.
"""

import subprocess
import sys
from pathlib import Path

import pytest

CODE = r"""
import sys

class _Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "{lib}":
            raise ModuleNotFoundError(f"blocked: {{name}}")
        return None

sys.meta_path.insert(0, _Block())

import dense_armor

try:
    import dense_armor.bridges.{lib}
except ModuleNotFoundError as exc:
    msg = str(exc)
    assert "pip install dense-armor[{extra}]" in msg, msg
    print("OK:", msg)
else:
    raise SystemExit("expected ModuleNotFoundError")
"""


@pytest.mark.parametrize(
    ("lib", "extra"), [("torch", "torch"), ("tensorflow", "tensorflow")]
)
def test_import_without_library_names_the_extra(lib, extra):
    r = subprocess.run(
        [sys.executable, "-c", CODE.format(lib=lib, extra=extra)],
        cwd=str(Path(__file__).resolve().parents[2]),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert r.returncode == 0, r.stderr[-400:]
    assert f"pip install dense-armor[{extra}]" in r.stdout
