"""`osqp` and `xacro` are optional (the `dynamics` extra): the library works without them."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _run(code):
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=REPO, capture_output=True, text=True,
    )


def test_base_dependencies_do_not_include_osqp_or_xacro():
    tomllib = pytest.importorskip("tomllib")
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    base = {d.split("[")[0].split(">")[0].split("=")[0] for d in data["project"]["dependencies"]}
    assert not {"osqp", "xacro"} & base
    assert {"osqp", "xacro"} <= set(data["project"]["optional-dependencies"]["dynamics"])


def test_dynamics_imports_without_osqp_and_xacro():
    r = _run(
        """
        import sys
        sys.modules["osqp"] = None
        sys.modules["xacro"] = None
        import dense_armor
        from dense_armor.dynamics import RigidBodyModel
        """
    )
    assert r.returncode == 0, r.stderr


def test_controller_without_osqp_names_the_extra():
    r = _run(
        """
        import sys
        sys.modules["osqp"] = None
        from dense_armor.dynamics import passivity_cbf_controller
        """
    )
    assert r.returncode != 0
    assert "dense-armor[dynamics]" in r.stderr


def test_xacro_file_without_xacro_names_the_extra():
    r = _run(
        """
        import sys
        sys.modules["xacro"] = None
        from dense_armor.dynamics import RigidBodyModel
        RigidBodyModel("test/fixtures/urdf/xacro_panda/panda_arm_hand.urdf.xacro")
        """
    )
    assert r.returncode != 0
    assert "dense-armor[dynamics]" in r.stderr
