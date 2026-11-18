"""Losses, with a forward and a backward.

The module of the same name, ``losses.losses``, is re-exported here, so that
``dense_armor.core.deep.neural.losses`` keeps the names it had as a module.
"""

from dense_armor.core.deep.neural.losses import losses as _impl

globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
