# Control

Commands for a robot: where it may go, how fast a command may change, the path to follow, and the velocity that follows it.

- **[Rate limiter](rate_limiter.md)** — bounds how fast a command can change.
- **[CBF filter](cbf_filter.md)** — keeps a command out of forbidden regions.
- **[Trajectory](trajectory.md)** — closed-form, minimum-jerk point-to-point paths.
- **[Kinematic controller](kinematic_controller.md)** — turns a reference path into a velocity command with guaranteed convergence.
