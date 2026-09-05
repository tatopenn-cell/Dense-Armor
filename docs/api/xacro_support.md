# Loading robots from xacro macros

Real robot descriptions are rarely shipped as a single flat URDF. Manufacturers publish them as
`.xacro` macro files: parametrized building blocks (`xacro:macro`), math expressions
(`${-pi/2}`), conditionals (`xacro:unless`) and includes (`xacro:include`) that get expanded into
a plain URDF before anything can parse them. `RigidBodyModel` only accepted the expanded form.

## What changed

Point `RigidBodyModel` at a `.xacro` file directly:

```python
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel

model = RigidBodyModel("panda_arm_hand.urdf.xacro")
model.n   # 8 -- 7 arm joints + 1 independent gripper coordinate
```

The real `xacro` package (the same expander the ROS ecosystem itself uses, no ROS install
required) does the expansion; nothing about macros, math, or conditionals is reimplemented here.
A `.xacro` extension routes through `xacro.process_file(path).toxml()` first; any other
extension parses as a plain URDF exactly as before.

::: dense_armor.dynamics.urdf_dynamics.RigidBodyModel

---

## Details

**Promoted from Dense-Evolution-Discovery Experiment 66.** Expanding the Franka Panda's own
published macros (`panda_arm.xacro` + `hand.xacro`, from `clvrai/furniture`) first produced a
7-joint model and a `KeyError: 'panda_hand'` -- the hand macro attaches with
`connected_to="panda_link8"`, but the arm macro's own `panda_link8`/`panda_joint8` block was
commented out in the source. The separately checked-in, pre-expanded `panda_arm_hand.urdf` in
the same upstream repo does include that link, meaning it was generated from an earlier,
uncommented version of the same macro. Restoring that block (same real values: mass 0.005,
inertia 0.00003, origin `0 0 0.107`) reconnects the tree.

**New dependency**: `xacro`.

**Reproducing this**: `pytest test/test_xacro_support.py`.
