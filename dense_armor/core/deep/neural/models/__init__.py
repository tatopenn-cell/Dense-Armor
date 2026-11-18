"""Classic architectures, the autoencoder and the vector quantizer.

The module of the same name, ``models.models``, is re-exported here, so that
``dense_armor.core.deep.neural.models`` keeps the names it had as a module.
"""

from dense_armor.core.deep.neural.models import models as _impl

globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
