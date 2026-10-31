"""Online optimizers: stateful objects and pure JAX forms.

Each optimizer comes in two shapes. The object form owns its state and
offers step(weights, gradient) -> weights. The pure form is the pair
init_state(dim) / update(state, weights, gradient) -> (state, weights)
on JAX arrays, so it can be used inside jax.jit and jax.lax.scan.

The linear models in dense_armor.utility.linear use the pure form to
make a whole stream a single fused XLA computation; the tests use the
object form for step-by-step comparisons.

References:
    Orabona, F. (2019). A Modern Introduction to Online Learning.
    Kingma, D. P., Ba, J. (2014). Adam: a method for stochastic
    optimization. arXiv:1412.6980.
"""

from dense_armor.utility.optim.adagrad import AdaGrad
from dense_armor.utility.optim.adam import Adam
from dense_armor.utility.optim.adamw import AdamW
from dense_armor.utility.optim.ftrl import FTRLProximal
from dense_armor.utility.optim.newton import OnlineNewtonStep
from dense_armor.utility.optim.rmsprop import RMSProp
from dense_armor.utility.optim.sgd import SGD

__all__ = [
    "SGD",
    "AdaGrad",
    "Adam",
    "AdamW",
    "FTRLProximal",
    "OnlineNewtonStep",
    "RMSProp",
]
