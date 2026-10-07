# Coupled joints via `<mimic>`

A gripper's two fingers usually move together: closing one closes the other. URDF expresses
this with a `<mimic joint="..." multiplier="..." offset="..."/>` tag inside the slaved joint --
its own angle is always `multiplier * master_angle + offset`, never a free variable.
`RigidBodyModel` previously ignored that tag and gave the slaved joint its own independent
coordinate, double-counting a single real degree of freedom.

## What changed

A joint with a `<mimic>` tag no longer gets its own entry in `q`/`qd`/`tau` -- its motion is
computed from its master's coordinate wherever the model needs it:

```python
model = RigidBodyModel("panda_arm_hand.urdf.xacro")
model.n                      # 8, not 9 -- the two fingers share one real DOF
model.mimic_map["panda_finger_joint2"]   # (master_dof_idx, multiplier, offset)
```

Forward kinematics substitutes `q[master] * multiplier + offset` for the mimic joint's own
angle -- exact, not approximate, so gravity/Coriolis terms obtained through `jax.grad`/`jax.jvp`
on the resulting kinematics are automatically correct. The hand-built geometric Jacobian used
for the mass matrix needs its own explicit chain rule: a mimic joint's local Jacobian column
(computed from its own axis and origin, same as any joint) is scaled by `multiplier` and added
into its *master's* column, rather than getting a column of its own.

::: dense_armor.dynamics.urdf_dynamics.RigidBodyModel

---

## Details

**Promoted from Dense-Evolution-Discovery Experiment 67.** Checked against a real central
finite difference of `link_pose`, not just plausibility: driving Franka Panda's
`finger_joint1` (master) by 0.02 moves both fingertips by exactly 0.02 in opposite directions
(their local closing axes point opposite ways); the hand-built Jacobian's mimic column matches
the finite-difference derivative to under 1e-5.

**Only one master per mimic, no transitive chains.** A mimic joint's `joint=` attribute must
name an independent (non-mimic) joint; mimicking another mimic joint is not supported -- not
something real published URDFs do.

**Reproducing this**: `pytest test/test_mimic_joints.py`.
