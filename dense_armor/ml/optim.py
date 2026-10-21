"""Optimisers: the classes of the family that have no batch form.

References:
    Orabona, F. (2019). A modern introduction to online learning, Algorithms 2.1,
        4.1 and 7.2; Kingma, D. P., Ba, J. (2015). Adam, Algorithm 1; Hazan, E.,
        Agarwal, A., Kale, S. (2007). Logarithmic regret algorithms for online
        convex optimization, Algorithm 2: update rules applied to the gradient
        of one example at a time (not in the folder, bibliographic only).
"""

from __future__ import annotations

__all__: list[str] = []

STOPPED = {
    "SGD": "the reference defines an update on the gradient of one example; it holds no algorithm for the whole dataset",
    "AdaGrad": "the reference defines an update on the gradient of one example; it holds no algorithm for the whole dataset",
    "Adam": "the reference defines an update on the gradient of one example; it holds no algorithm for the whole dataset",
    "FTRLProximal": "the reference defines an update on the gradient of one example; it holds no algorithm for the whole dataset",
    "OnlineNewtonStep": "the reference defines an update on the gradient of one example; it holds no algorithm for the whole dataset",
}
