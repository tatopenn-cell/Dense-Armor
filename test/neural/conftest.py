"""Import jax before dense_armor so numpy is not reloaded under coverage."""

import jax  # noqa: F401
