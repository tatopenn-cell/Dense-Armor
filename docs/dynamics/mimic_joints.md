# Coupled joints via `<mimic>`

A gripper's two fingers usually move together: closing one closes the other, at the
same rate, in the opposite direction. URDF expresses this with a `<mimic>` tag inside
the slaved joint:

```xml
<joint name="panda_finger_joint2" type="prismatic">
  <mimic joint="panda_finger_joint1" multiplier="1.0" offset="0.0"/>
</joint>
```

The slaved joint's own angle is always `multiplier · master_angle + offset`. It is not
a free variable. The gripper has **one** degree of freedom, not two.

A parser that ignores the `<mimic>` tag gives the slaved joint its own independent
coordinate. The model then has one extra joint, one extra column in the mass matrix,
and one extra torque channel that does not correspond to any real actuator. Every
downstream computation — the mass matrix, the gravity vector, the Jacobian, the
controllers — is silently wrong.

## The robot

The Franka Panda with the hand. Two finger joints, one master (`panda_finger_joint1`),
one mimic (`panda_finger_joint2`).

```python
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel

model = RigidBodyModel("panda_arm_hand.urdf.xacro")
print(model.n, model.mimic_map["panda_finger_joint2"])
```

```
8 (7, 1.0, 0.0)
```

`model.n` is `8` — 7 arm joints plus 1 independent gripper coordinate, not 9. The two
fingers are counted as one real DOF.

## 1. How mimic joints are represented

The model keeps a map from each mimic joint to `(master_dof_idx, multiplier, offset)`:

```
(7, 1.0, 0.0)
```

- `7` — the index of the master joint in the `q` / `qd` / `tau` vectors.
- `1.0` — the multiplier.
- `0.0` — the offset.

Wherever the model needs the mimic joint's own angle (forward kinematics, Jacobian,
mass matrix), it computes `q[master] · multiplier + offset` instead of indexing its
own coordinate. The mimic joint contributes no independent column of its own.

## 2. Why this matters for the mass matrix

Without the mimic handling the model would have 9 coordinates: a 9×9 mass matrix and a ninth
torque channel for a finger that no motor drives on its own, and the dynamics could move the two
fingers independently, which the real gripper cannot do.

With the mimic handling the model is 8×8: the finger pair is one coordinate, and its mass enters
the matrix through the master joint.

## 3. How mimic joints are handled in the Jacobian

Forward kinematics is exact: `q[master] · multiplier + offset` is substituted
wherever the mimic joint's own angle would appear. Since the rest of the kinematics
pipeline uses `jax.grad` / `jax.jvp` on this forward-kinematics function, the
gravity and Coriolis terms obtained from it are automatically correct — no separate
chain rule to write by hand.

The hand-built **geometric Jacobian** (used inside the mass-matrix construction) does
not use autodiff, so it needs its own explicit handling: a mimic joint's local Jacobian
column (computed from its own axis and origin, same as any other joint) is
**scaled by the multiplier** and **added into its master's column**, instead of getting
a column of its own.

## 4. Verifying the handling

Drive the master finger joint by a small amount and check that both fingertips move by
the same amount in opposite directions (their local closing axes point opposite ways):

```python
import jax.numpy as jnp
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel

model = RigidBodyModel("panda_arm_hand.urdf.xacro")
q = jnp.zeros(model.n)
p1, p2 = (model.link_pose(q, k)[0] for k in ("panda_leftfinger", "panda_rightfinger"))
q1, q2 = (model.link_pose(q.at[7].set(0.02), k)[0] for k in ("panda_leftfinger", "panda_rightfinger"))
print(round(float(jnp.linalg.norm(q1 - p1)), 4), round(float(jnp.linalg.norm(q2 - p2)), 4))
```

```
0.02 0.02
```

Both fingers moved by exactly the master's displacement. In the Jacobian, the mimic
column matches the finite-difference derivative of `link_pose` to under `1e-5`.

## 5. What is not supported

**One master per mimic.** A mimic joint's `joint=` attribute must name an
**independent** (non-mimic) joint. Mimicking another mimic joint is not supported.

**No transitive chains.** Even if the source file names a chain of mimic joints, only
the direct link is used. Real published URDFs do not do this; the restriction is a
scope decision, not a limitation that matters in practice.

**One mimic per master is not required.** Multiple mimics can slave off the same
master (a three-finger gripper where two fingers follow the first). Each mimic records
its own `(master_idx, multiplier, offset)`.

## API reference

::: dense_armor.dynamics.urdf_dynamics.RigidBodyModel

---

## Details

**Promoted from Dense-Evolution-Discovery Experiment 67.** Checked against a real
central finite difference of `link_pose`, not just plausibility: driving Franka
Panda's `finger_joint1` (master) by 0.02 moves both fingertips by exactly 0.02 in
opposite directions (their local closing axes point opposite ways); the hand-built
Jacobian's mimic column matches the finite-difference derivative to under `1e-5`.

**Only one master per mimic, no transitive chains.** A mimic joint's `joint=` attribute
must name an independent (non-mimic) joint; mimicking another mimic joint is not
supported — not something real published URDFs do.

**Reproducing this**: `pytest test/test_mimic_joints.py`.

**See also**: [Xacro files](xacro_support.md) — the other feature real robot
descriptions use that a naive URDF parser misses. The two are independent: a robot can
have both, or just one.
