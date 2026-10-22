"""Rules: the classes of the family that have no batch form.

References:
    Duarte, J., Gama, J., Bifet, A. (2016). Adaptive model rules from
        high-speed data streams, ACM TKDD 10(3): the rule set grows and is
        revised as examples arrive (not in the folder, bibliographic only).
"""

from __future__ import annotations

__all__: list[str] = []

STOPPED = {
    "AMRulesRegressor": "the reference defines the rule set online, example by example: it holds no algorithm for the whole dataset",
}
