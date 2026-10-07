"""Dense-Armor estimator checks.

Each check is a standalone function; ``check_estimator(est)`` runs the
ones that apply to the estimator's role.
"""

from dense_armor.checks.check_estimator import check_estimator

__all__ = ["check_estimator"]
