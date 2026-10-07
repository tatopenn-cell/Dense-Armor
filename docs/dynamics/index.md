# Robot dynamics

The control modules on [Control](../control/index.md) work at the single-integrator
level: you send a joint velocity, they bound it. That is enough for a robot whose
driver already implements an inner torque loop.

This section is for the other case: you have the robot's **URDF file** and you want the
actual physical model behind the velocity API — mass matrix, gravity, forward dynamics,
and the kinematics of any link. From the model, the sections build two real controllers
that work at the torque level and stay away from kinematic singularities.

## The robot used everywhere

A Franka Emika Panda, described by its standard URDF file. Every page uses it unless
otherwise noted; the `urdf_dynamics` page additionally cross-checks on two Kinova Gen3
arms. Nothing here is hardcoded to a specific robot: `RigidBodyModel` reads the file and
builds whatever the file describes.

```python
import jax
jax.config.update("jax_enable_x64", True)
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel

model = RigidBodyModel("panda.urdf")
model.n
```

## Pages

- **[Rigid-body dynamics](urdf_dynamics.md)** — mass matrix, gravity, Coriolis terms,
  forward dynamics, and kinematics of any link, all from the URDF file. The module
  everything else in this section builds on.
- **[Passivity + singularity-CBF controller](passivity_cbf_controller.md)** — task-space
  position control that guarantees passivity of the tracking error and stays away from
  kinematic singularities.
- **[Full 6-DoF controller](six_dof_pbc_cbf_controller.md)** — the same idea, extended
  to position **and** orientation, using the link's full spatial Jacobian.
- **[Xacro files](xacro_support.md)** — loading robots described with `.xacro` macros
  (parameters, math expressions, conditionals, includes) directly.
- **[Mimic joints](mimic_joints.md)** — modelling coupled joints (a gripper's two
  fingers tied together) correctly, so the model counts real DOF, not raw `<joint>` tags.
