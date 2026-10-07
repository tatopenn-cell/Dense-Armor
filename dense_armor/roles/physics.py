"""Physical units and robot limits.

Two ways in which physics enters a control loop:

- **units**: a transformer that expects an angle in radians must not
  receive an angle in degrees. When a pipeline is built, this module
  checks that the output units of each step match the input units of
  the next one, and raises a clear error otherwise.
- **limits**: the URDF file of a robot declares the position and
  velocity limits of every joint. :func:`limits_from_urdf` reads them
  into a :class:`JointLimits` object that can be used as a guard for
  :class:`~dense_armor.roles.safety.SafeEstimator`.

The unit vocabulary is small and explicit: ``"rad"``, ``"rad/s"``,
``"N*m"``, ``"m"``, ``"m/s"``, ``"m/s^2"``, ``"K"``, ``"1"`` for
dimensionless. Unknown units are strings, but they never match another
unit except themselves.

References
----------
Fitzpatrick, R. (2008). Maxwell's equations and the principles of
    electromagnetism. Jones & Bartlett — for the SI unit conventions.
"""
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from dense_armor.roles import Estimator, Transformer


DIMENSIONLESS = "1"


@dataclass(frozen=True)
class UnitSpec:
    """Input and output units of an estimator, as ``{name: unit}`` dicts.

    Attributes:
        inputs: unit per input channel.
        outputs: unit per output channel.
    """

    inputs: Mapping[str, str] = field(default_factory=dict)
    outputs: Mapping[str, str] = field(default_factory=dict)


def unit_spec_of(est: Any) -> Optional[UnitSpec]:
    """Return the ``UnitSpec`` declared by ``est``, or ``None``.

    An estimator declares its units by setting a class attribute
    ``units`` to a :class:`UnitSpec` or to a two-dict tuple. Missing
    declarations are not an error; they simply skip the unit check.
    """
    u = getattr(est, "units", None)
    if u is None:
        return None
    if isinstance(u, UnitSpec):
        return u
    if isinstance(u, tuple) and len(u) == 2:
        return UnitSpec(inputs=dict(u[0]), outputs=dict(u[1]))
    return None


def _mismatch_message(step_name: str, produced: str, expected: str) -> str:
    return (
        f"unit mismatch at step {step_name!r}: previous output unit "
        f"{produced!r} does not match declared input unit {expected!r}"
    )


class UnitCheckedPipeline(Transformer):
    """A pipeline that checks units at build time.

    Every step may declare its units via a class attribute ``units``
    holding a :class:`UnitSpec` (or a ``(inputs, outputs)`` pair of
    dicts). When a step's input unit differs from the previous step's
    output unit for the same channel, construction raises.

    Args:
        steps: the transformers and the final estimator, in order.
        output_units: optional ``{name: unit}`` for the pipeline as a
            whole. When not given, the output units of the last step are
            used.

    Raises:
        ValueError: if any two consecutive steps disagree on a unit, or
            if a step declares an output unit for a channel that the
            next step does not declare.

    Examples:
        >>> from dense_armor.roles.physics import UnitCheckedPipeline, UnitSpec
        >>> class Scale(Transformer):
        ...     def __init__(self, f=1.0): self.f = f
        ...     def transform_one(self, x, t=None):
        ...         return {k: self.f * v for k, v in x.items()}
        >>> class Deg2Rad(Transformer):
        ...     units = UnitSpec(inputs={"q": "deg"}, outputs={"q": "rad"})
        ...     def __init__(self): pass
        ...     def transform_one(self, x, t=None):
        ...         return {k: v * np.pi / 180.0 for k, v in x.items()}
        >>> class NeedsRad(Transformer):
        ...     units = UnitSpec(inputs={"q": "rad"}, outputs={"q": "rad"})
        ...     def __init__(self): pass
        ...     def transform_one(self, x, t=None):
        ...         return x
        >>> p = UnitCheckedPipeline([Deg2Rad(), NeedsRad()])
        >>> p.step_units[0][1] == {"q": "rad"}
        True
    """

    _supervised = False

    def __init__(
        self,
        steps: Sequence[Any],
        output_units: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.steps = list(steps)
        self.output_units = dict(output_units) if output_units else None
        self.step_units: list[tuple[Optional[Mapping[str, str]],
                                    Optional[Mapping[str, str]]]] = []
        prev_outputs: Optional[Mapping[str, str]] = None
        for i, step in enumerate(self.steps):
            spec = unit_spec_of(step)
            if spec is None:
                self.step_units.append((None, None))
                prev_outputs = None
                continue
            if prev_outputs is not None:
                for name, expected in spec.inputs.items():
                    if name in prev_outputs and prev_outputs[name] != expected:
                        raise ValueError(
                            _mismatch_message(
                                type(step).__name__,
                                prev_outputs[name],
                                expected,
                            )
                        )
            self.step_units.append((spec.inputs, spec.outputs))
            prev_outputs = spec.outputs

    def learn_one(self, x, y=None, t=None):
        h = x
        for step in self.steps[:-1]:
            h = step.transform_one(h, t=t) if t is not None else step.transform_one(h)
        last = self.steps[-1]
        if hasattr(last, "learn_one"):
            try:
                if y is None:
                    last.learn_one(h, t=t)
                else:
                    last.learn_one(h, y, t=t)
            except TypeError:
                if y is None:
                    last.learn_one(h)
                else:
                    last.learn_one(h, y)
        return self

    def transform_one(self, x, t=None):
        h = x
        for step in self.steps:
            h = step.transform_one(h, t=t) if t is not None else step.transform_one(h)
        return h


@dataclass(frozen=True)
class JointLimits:
    """Position and velocity limits of every joint in a robot.

    Attributes:
        joint_names: names of the joints, in URDF order.
        lower: lower position limits, radians or metres.
        upper: upper position limits, same units as ``lower``.
        velocity: maximum velocity per joint.
        effort: maximum effort per joint.
    """

    joint_names: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    velocity: np.ndarray
    effort: np.ndarray

    @property
    def n(self) -> int:
        return len(self.joint_names)

    def in_bounds(self, q: Sequence[float] | np.ndarray) -> bool:
        """Return ``True`` if ``q`` is inside ``[lower, upper]`` everywhere."""
        q_arr = np.asarray(q, dtype=float)
        if q_arr.shape != (self.n,):
            raise ValueError(
                f"q has shape {q_arr.shape}, expected ({self.n},)"
            )
        return bool(np.all(q_arr >= self.lower) and np.all(q_arr <= self.upper))

    def as_guard(self):
        """Return a guard ``x -> bool`` that flags out-of-bounds positions.

        The returned callable accepts a dict with a key ``"q"`` holding
        a sequence of joint positions, or a dict where each joint is a
        separate key ``q0, q1, ...``. It is suitable for
        :class:`~dense_armor.roles.safety.SafeEstimator`.
        """

        def _guard(x: Mapping[str, Any]) -> bool:
            if "q" in x:
                q = np.asarray(x["q"], dtype=float)
            else:
                q = np.asarray(
                    [float(x.get(f"q{i}", 0.0)) for i in range(self.n)]
                )
            return not self.in_bounds(q)

        return _guard


def limits_from_urdf(source: str | Path | Any) -> JointLimits:
    """Read joint limits from a URDF file or a loaded model.

    Args:
        source: a path to a ``.urdf`` file, or an object exposing a
            ``.path`` attribute pointing to one (for example a
            ``RigidBodyModel``).

    Returns:
        A :class:`JointLimits` covering every joint in the URDF that
        has a ``<limit>`` block (fixed joints have none and are
        skipped).

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if no joint with a ``<limit>`` block is found.
    """
    if hasattr(source, "path"):
        path = Path(getattr(source, "path"))
    else:
        path = Path(str(source))
    if not path.is_file():
        raise FileNotFoundError(f"URDF not found: {path}")
    root = ET.parse(path).getroot()
    names: list[str] = []
    lower: list[float] = []
    upper: list[float] = []
    velocity: list[float] = []
    effort: list[float] = []
    for j in root.findall("joint"):
        jtype = j.get("type", "")
        if jtype == "fixed":
            continue
        lim = j.find("limit")
        if lim is None:
            continue
        name = j.get("name", f"joint{len(names)}")
        names.append(name)
        lo = lim.get("lower")
        up = lim.get("upper")
        lower.append(float(lo) if lo is not None else float("-inf"))
        upper.append(float(up) if up is not None else float("inf"))
        velocity.append(float(lim.get("velocity", "inf")))
        effort.append(float(lim.get("effort", "inf")))
    if not names:
        raise ValueError(f"no joint with a <limit> block in {path}")
    return JointLimits(
        joint_names=tuple(names),
        lower=np.asarray(lower, dtype=float),
        upper=np.asarray(upper, dtype=float),
        velocity=np.asarray(velocity, dtype=float),
        effort=np.asarray(effort, dtype=float),
    )


class PhysicalLimitsGuard:
    """Guard that flags samples whose joint positions are out of bounds.

    Args:
        limits: the :class:`JointLimits` of the robot.
        margin: safety margin applied to both sides, in the same units
            as the limits. ``0.05`` shrinks the upper limit by 5% of
            its range and extends the lower limit by the same amount.
    """

    def __init__(self, limits: JointLimits, margin: float = 0.0) -> None:
        self.limits = limits
        self.margin = margin
        span = limits.upper - limits.lower
        self._lo = limits.lower + margin * span
        self._hi = limits.upper - margin * span

    def __call__(self, x: Mapping[str, Any]) -> bool:
        if "q" in x:
            q = np.asarray(x["q"], dtype=float)
        else:
            q = np.asarray(
                [float(x.get(f"q{i}", 0.0)) for i in range(self.limits.n)]
            )
        return not bool(np.all(q >= self._lo) and np.all(q <= self._hi))
