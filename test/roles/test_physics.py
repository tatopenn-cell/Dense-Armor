"""Tests for dense_armor/roles/physics.py."""
import doctest
from pathlib import Path

import numpy as np
import pytest

import dense_armor.roles.physics as phys_mod
from dense_armor.roles import Transformer
from dense_armor.roles.physics import (
    JointLimits,
    PhysicalLimitsGuard,
    UnitCheckedPipeline,
    UnitSpec,
    limits_from_urdf,
    unit_spec_of,
)


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "urdf"


class _Deg2Rad(Transformer):
    units = UnitSpec(inputs={"q": "deg"}, outputs={"q": "rad"})

    def __init__(self) -> None:
        pass

    def transform_one(self, x, t=None):
        return {k: v * np.pi / 180.0 for k, v in x.items()}


class _NeedsRad(Transformer):
    units = UnitSpec(inputs={"q": "rad"}, outputs={"q": "rad"})

    def __init__(self) -> None:
        pass

    def transform_one(self, x, t=None):
        return dict(x)


class _NeedsDeg(Transformer):
    units = UnitSpec(inputs={"q": "deg"}, outputs={"q": "deg"})

    def __init__(self) -> None:
        pass

    def transform_one(self, x, t=None):
        return dict(x)


class _Untyped(Transformer):
    def __init__(self) -> None:
        pass

    def transform_one(self, x, t=None):
        return dict(x)


def test_unit_spec_of_none_when_undeclared():
    assert unit_spec_of(_Untyped()) is None


def test_unit_spec_of_returns_spec():
    spec = unit_spec_of(_Deg2Rad())
    assert isinstance(spec, UnitSpec)
    assert spec.inputs == {"q": "deg"}
    assert spec.outputs == {"q": "rad"}


def test_pipeline_accepts_matching_units():
    p = UnitCheckedPipeline([_Deg2Rad(), _NeedsRad()])
    assert p.step_units[1][0] == {"q": "rad"}


def test_pipeline_rejects_mismatched_units():
    with pytest.raises(ValueError, match="unit mismatch"):
        UnitCheckedPipeline([_Deg2Rad(), _NeedsDeg()])


def test_pipeline_skips_untyped_steps():
    p = UnitCheckedPipeline([_Untyped(), _NeedsRad()])
    assert p.step_units[0] == (None, None)
    assert p.step_units[1][0] == {"q": "rad"}


def test_pipeline_transform_runs():
    p = UnitCheckedPipeline([_Deg2Rad(), _NeedsRad()])
    out = p.transform_one({"q": 180.0})
    assert out == {"q": pytest.approx(np.pi)}


def test_limits_from_urdf_panda():
    path = FIXTURES / "panda.urdf"
    lim = limits_from_urdf(path)
    assert lim.n == 9
    assert "panda_joint1" in lim.joint_names
    assert lim.lower[0] == pytest.approx(-2.9671)
    assert lim.upper[0] == pytest.approx(2.9671)
    assert lim.velocity[0] == pytest.approx(2.1750)


def test_limits_from_urdf_gen3():
    path = FIXTURES / "GEN3-6DOF.urdf"
    lim = limits_from_urdf(path)
    assert lim.n == 6
    assert lim.joint_names[0] == "Actuator1"
    assert lim.upper[0] == np.inf
    assert lim.lower[0] == -np.inf


def test_limits_from_urdf_missing_file():
    with pytest.raises(FileNotFoundError):
        limits_from_urdf("/tmp/does-not-exist.urdf")


def test_joint_limits_in_bounds():
    path = FIXTURES / "panda.urdf"
    lim = limits_from_urdf(path)
    q_in = np.zeros(lim.n)
    assert lim.in_bounds(q_in)
    q_out = np.zeros(lim.n)
    q_out[0] = 10.0
    assert not lim.in_bounds(q_out)


def test_joint_limits_shape_mismatch():
    path = FIXTURES / "panda.urdf"
    lim = limits_from_urdf(path)
    with pytest.raises(ValueError):
        lim.in_bounds(np.zeros(lim.n - 1))


def test_physical_limits_guard_flags_out_of_bounds():
    path = FIXTURES / "panda.urdf"
    lim = limits_from_urdf(path)
    guard = PhysicalLimitsGuard(lim, margin=0.0)
    q_ok = {f"q{i}": 0.0 for i in range(lim.n)}
    assert guard(q_ok) is False
    q_bad = dict(q_ok)
    q_bad["q0"] = 10.0
    assert guard(q_bad) is True


def test_physical_limits_guard_margin_shrinks_bounds():
    path = FIXTURES / "panda.urdf"
    lim = limits_from_urdf(path)
    guard_tight = PhysicalLimitsGuard(lim, margin=0.0)
    guard_loose = PhysicalLimitsGuard(lim, margin=0.5)
    q_ok = {f"q{i}": 0.0 for i in range(lim.n)}
    q_ok["q0"] = 2.5
    assert guard_tight(q_ok) is False
    assert guard_loose(q_ok) is True


def test_doctests():
    assert doctest.testmod(phys_mod).failed == 0


class _Last:
    units = UnitSpec(inputs={"q": "rad"}, outputs={"q": "rad"})

    def __init__(self):
        self.got = []

    def transform_one(self, x, t=None):
        return dict(x)

    def learn_one(self, x, y=None, t=None):
        self.got.append((x, y))


class _NoT:
    def __init__(self):
        self.got = []

    def learn_one(self, x, y=None):
        self.got.append((x, y))


def _limits():
    return JointLimits(
        joint_names=("a", "b"),
        lower=np.array([-1.0, -1.0]),
        upper=np.array([1.0, 1.0]),
        velocity=np.array([1.0, 1.0]),
        effort=np.array([1.0, 1.0]),
    )


def test_unit_spec_of_tuple_and_other():
    class _T:
        units = ({"q": "deg"}, {"q": "rad"})

    class _O:
        units = "rad"

    assert unit_spec_of(_T()).outputs == {"q": "rad"}
    assert unit_spec_of(_O()) is None


def test_pipeline_learn_one_with_and_without_target():
    last = _Last()
    p = UnitCheckedPipeline([_Deg2Rad(), last])
    p.learn_one({"q": 180.0})
    p.learn_one({"q": 180.0}, 1.0, t=0.1)
    assert last.got[0][0]["q"] == pytest.approx(np.pi)
    assert last.got[1][1] == 1.0


def test_pipeline_learn_one_falls_back_without_t():
    last = _NoT()
    p = UnitCheckedPipeline([_Untyped(), last])
    p.learn_one({"q": 1.0}, t=0.1)
    p.learn_one({"q": 1.0}, 2.0, t=0.1)
    assert [g[1] for g in last.got] == [None, 2.0]


def test_joint_limits_as_guard():
    g = _limits().as_guard()
    assert g({"q": [0.0, 0.0]}) is False
    assert g({"q0": 2.0, "q1": 0.0}) is True


def test_physical_limits_guard_keyed_joints():
    guard = PhysicalLimitsGuard(_limits())
    assert not guard({"q0": 0.5, "q1": -0.5})
    assert guard({"q0": 1.5})


def test_limits_from_urdf_object_with_path_and_edge_cases(tmp_path):
    f = tmp_path / "r.urdf"
    f.write_text(
        '<robot name="r">'
        '<joint name="fix" type="fixed"/>'
        '<joint name="free" type="continuous"/>'
        '<joint name="j" type="revolute"><limit effort="5"/></joint>'
        "</robot>"
    )

    class _Src:
        path = f

    lim = limits_from_urdf(_Src())
    assert lim.joint_names == ("j",)
    assert lim.lower[0] == -np.inf and lim.upper[0] == np.inf
    assert lim.velocity[0] == np.inf and lim.effort[0] == 5.0


def test_limits_from_urdf_without_limits(tmp_path):
    f = tmp_path / "e.urdf"
    f.write_text('<robot name="e"><joint name="fix" type="fixed"/></robot>')
    with pytest.raises(ValueError, match="no joint"):
        limits_from_urdf(f)
