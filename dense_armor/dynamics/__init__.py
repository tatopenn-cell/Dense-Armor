# -*- coding: utf-8 -*-
"""
dynamics/__init__.py
=======================
Real physical robot models (rigid-body dynamics from a real URDF file) --
its own subpackage, separate from utility/, because it needs an actual
physical model file and returns real torque-level quantities, unlike the
single-integrator utility/ modules (rate_limiter, cbf_filter, trajectory,
kinematic_controller).
"""
from .urdf_dynamics import RigidBodyModel

__all__ = ["RigidBodyModel"]
