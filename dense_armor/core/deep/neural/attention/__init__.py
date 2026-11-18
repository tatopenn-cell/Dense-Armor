"""Attention layers, position embeddings and the sparse mixture of experts.

The module of the same name, ``attention.attention``, is re-exported here, so that
``dense_armor.core.deep.neural.attention`` keeps the names it had as a module.
"""

from dense_armor.core.deep.neural.attention import attention as _impl

globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
