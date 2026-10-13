"""Static and parametric transformation matrices, backend-agnostic.

:class:`TensorVault` detects the active backend (JAX or NumPy) and the active
precision (float32 or float64) and synthesizes the requested matrix in that
precision.
"""

import numpy as np

try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False


class TensorVault:
    """Vault for static and parametric transformation matrices."""

    def __init__(self) -> None:
        """Detect the backend (JAX or NumPy) and the active precision.

        ``jax_enable_x64`` is read with :func:`getattr` instead of the
        deprecated ``jax.config.get`` method; it defaults to ``False`` when
        JAX is not installed.
        """
        self.xp = jnp if HAS_JAX else np
        is_x64 = getattr(jax.config, "jax_enable_x64", False) if HAS_JAX else False
        self.dtype = self.xp.float64 if (is_x64 or not HAS_JAX) else self.xp.float32

    def get_static_transform(self, name: str) -> np.ndarray:
        """Synthesize and return the requested static matrix.

        Args:
            name: one of ``invert``, ``identity``, ``edge_detector``, ``blend``.

        Returns:
            The matrix in the active backend and precision.

        Raises:
            KeyError: if ``name`` is not one of the available transforms.
        """
        xp, dt = self.xp, self.dtype
        transforms = {
            "invert": lambda: xp.array([[0.0, 1.0], [1.0, 0.0]], dtype=dt),
            "identity": lambda: xp.eye(2, dtype=dt),
            "edge_detector": lambda: xp.array([-1.0, 2.0, -1.0], dtype=dt),
            "blend": lambda: xp.array([[0.5, 0.5], [0.5, 0.5]], dtype=dt),
        }
        key = name.lower()
        if key not in transforms:
            raise KeyError(
                f"Static transform {name!r} not available. "
                f"Available: {list(transforms.keys())}"
            )
        return transforms[key]()

    def get_parametric_transform(self, name: str, p: float) -> np.ndarray:
        """Synthesize the requested parametric matrix from ``p``.

        Args:
            name: one of ``scale_project``, ``amplify``, ``bias_shift``.
            p: the parameter of the transform (angle or scale factor).

        Returns:
            The matrix in the active backend and precision.

        Raises:
            KeyError: if ``name`` is not one of the available transforms.
        """
        xp, dt = self.xp, self.dtype
        transforms = {
            "scale_project": lambda: xp.array(
                [[xp.cos(p), -xp.sin(p)], [xp.sin(p), xp.cos(p)]], dtype=dt
            ),
            "amplify": lambda: xp.array([[p, 0.0], [0.0, p]], dtype=dt),
            "bias_shift": lambda: xp.array([p, -p], dtype=dt),
        }
        key = name.lower()
        if key not in transforms:
            raise KeyError(
                f"Parametric transform {name!r} not available. "
                f"Available: {list(transforms.keys())}"
            )
        return transforms[key]()

    def get_backend_info(self) -> str:
        """Readable description of the active backend and precision.

        Returns:
            A string such as ``"JAX / GPU Accelerated (32-bit)"``.
        """
        precision = (
            "64-bit"
            if self.dtype in (np.float64, getattr(jnp, "float64", None))
            else "32-bit"
        )
        backend = "JAX / GPU Accelerated" if HAS_JAX else "NumPy / CPU Standard"
        return f"{backend} ({precision})"
