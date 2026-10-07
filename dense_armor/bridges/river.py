"""River bridge.

Exposes a single import-time check: importing this module without
river raises a clear ModuleNotFoundError that names the extra.
The estimators themselves are duck-typed and interoperate without
wrapping (river pipelines accept our objects, our pipelines accept
river objects).

Needs the optional dependency: ``pip install dense-armor[river]``.
"""

from __future__ import annotations

try:
    from river import base as _river_base  # noqa: F401
    from river import checks as _river_checks  # noqa: F401
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "dense_armor.bridges.river needs river: pip install dense-armor[river]"
    ) from exc


def to_river(est):
    """Return a river-compatible view of a Dense-Armor estimator.

    Duck-typed: our estimators and river estimators share the same
    ``learn_one`` / ``predict_one`` / ``predict_proba_one`` interface,
    so the bridge is the identity function.
    """
    return est


def from_river(est):
    """Return a Dense-Armor-compatible view of a river estimator."""
    return est
