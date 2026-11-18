"""Online and continual learning of a network.

The module of the same name, ``online.online``, is re-exported here, so that
``dense_armor.core.deep.neural.online`` keeps the names it had as a module.
"""

from dense_armor.core.deep.neural.online import online as _impl

globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
