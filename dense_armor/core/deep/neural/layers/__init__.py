"""Layers and their building blocks: parameters, initialisations, dense, convolution, recurrent, normalisation, activations.

The module of the same name, ``layers.layers``, is re-exported here, so that
``dense_armor.core.deep.neural.layers`` keeps the names it had as a module.
"""

from dense_armor.core.deep.neural.layers import layers as _impl

globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
