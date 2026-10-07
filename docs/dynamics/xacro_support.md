# Loading robots from xacro macros

Real robot descriptions are rarely shipped as a single flat URDF. Manufacturers publish
them as `.xacro` macro files: parametrized building blocks (`xacro:macro`), math
expressions (`${-pi/2}`), conditionals (`xacro:unless`) and includes (`xacro:include`)
that get expanded into a plain URDF before anything can parse them. A robot library
that only accepts the expanded form forces the user to run the expander by hand and
keep two files in sync.

`RigidBodyModel` accepts either. Point it at a `.xacro` file and the expansion happens
inside the loader.

## The robot

The Franka Panda from `clvrai/furniture`. Its description ships as two xacro files
(`panda_arm.xacro` and `hand.xacro`) plus a top-level `panda_arm_hand.urdf.xacro` that
includes them. Loading the top-level file gives the full arm + hand.

```python
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel

model = RigidBodyModel("panda_arm_hand.urdf.xacro")
print(model.n)
```

```
8
```

`model.n` is `8` — 7 arm joints plus 1 independent gripper coordinate. The two finger
joints are counted as one real DOF because of the `<mimic>` tag on the second one
(see [coupled joints via mimic](mimic_joints.md)).

## 1. What happens under the hood

If the path ends in `.xacro`, the loader calls the real `xacro` package —
`xacro.process_file(path).toxml()` — and parses the resulting XML. If the path ends in
anything else, the loader parses it directly as URDF, exactly as before.

The `xacro` package is the same expander the ROS ecosystem uses. Nothing about macros,
math expressions, conditionals, or includes is reimplemented here: a xacro file is
expanded by xacro, and the resulting URDF is parsed by `RigidBodyModel`.

## 2. Install

```bash
pip install xacro
```

The `xacro` package is a small pure-Python dependency. It does not require a ROS
installation.

## 3. Flat URDF files are unchanged

If your URDF is already expanded (no `<xacro:...>` tags, no `.xacro` extension), the
loader reads it exactly the same as before. There is no cost and no change in
behaviour for the common case.

## 4. Common cases

**Parameters.** A manufacturer ships the same arm with different payloads, and
parametrizes the mass:

```xml
<xacro:property name="payload" value="0.5"/>
<link name="link2">
  <inertial>
    <mass value="${payload}"/>
  </inertial>
</link>
```

`xacro` substitutes `0.5` for `${payload}` before the loader sees the file. The
resulting `RigidBodyModel` reads the substituted value.

**Conditionals.** A description includes an optional sensor:

```xml
<xacro:if value="${with_sensor}">
  <link name="sensor_link">...</link>
</xacro:if>
```

The `with_sensor` flag is set at the top of the file (or passed on the command line by
the caller's tooling). Expanded, the sensor either exists or does not.

**Includes.** A description splits the arm and the hand across files:

```xml
<xacro:include filename="hand.xacro"/>
```

The include is resolved relative to the including file. This is what makes the Panda's
`panda_arm_hand.urdf.xacro` work: it includes `panda_arm.xacro` and `hand.xacro`
in the right order.

## 5. Debugging an expansion

When the expansion fails — a missing parameter, a macro recursion, an include path the
`xacro` package cannot resolve — the error comes from `xacro` and the traceback points
at the offending line in the macro file. The error is not a `RigidBodyModel` error and
should not be treated as one.

To see the expanded URDF without loading it into the model, call `xacro` directly:

```bash
python -c "import xacro; print(xacro.process_file('panda_arm_hand.urdf.xacro').toxml())" > panda_expanded.urdf
```

The result is a plain URDF. It can be inspected, diffed against the source, and — if
you want — passed to `RigidBodyModel` as a normal file.

## API reference

::: dense_armor.dynamics.urdf_dynamics.RigidBodyModel

---

## Details

**Promoted from Dense-Evolution-Discovery Experiment 66.** Expanding the Franka
Panda's own published macros (`panda_arm.xacro` + `hand.xacro`, from
`clvrai/furniture`) first produced a 7-joint model and a `KeyError: 'panda_hand'`: the
hand macro attaches with `connected_to="panda_link8"`, but the arm macro's own
`panda_link8` / `panda_joint8` block was **commented out** in the source. The
separately checked-in, pre-expanded `panda_arm_hand.urdf` in the same upstream repo
does include that link, meaning it was generated from an earlier, uncommented version
of the same macro. Restoring that block (same real values: mass 0.005, inertia
0.00003, origin `0 0 0.107`) reconnects the tree.

**New dependency**: `xacro`.

**Reproducing this**: `pytest test/test_xacro_support.py`.

**See also**: [Mimic joints](mimic_joints.md) — coupled joints (a gripper's two
fingers) also need special handling, and the two features are independent: a robot can
have both xacro files and mimic joints, or just one of them.
