# Robot dynamics

The physical model of a real robot, read from its URDF file, and controllers built on it.

- **[Rigid-body dynamics](urdf_dynamics.md)** — mass matrix, gravity and energy of any robot from its URDF.
- **[Passivity + singularity-CBF controller](passivity_cbf_controller.md)** — task-space control that stays away from singularities.
- **[Full 6-DoF controller](six_dof_pbc_cbf_controller.md)** — the same, for position and orientation.
- **[Xacro files](xacro_support.md)** — loading robots described with xacro macros.
- **[Mimic joints](mimic_joints.md)** — coupled joints that move together.
