"""Online evaluation helpers.

Two protocols:

- :func:`progressive_val_score` / :func:`progressive_val_proba_score`
  for a supervised model on a labelled stream.
- :func:`evaluate_events` for a drift or anomaly detector against
  labelled event windows.

Plus :class:`OnlineModelSelection` / :func:`best_of` for model
selection on a stream.
"""
from dense_armor.utility.evaluate.events import evaluate_events
from dense_armor.utility.evaluate.model_selection import (
    OnlineModelSelection,
    best_of,
)
from dense_armor.utility.evaluate.prequential import (
    progressive_val_proba_score,
    progressive_val_score,
)

__all__ = [
    "OnlineModelSelection",
    "best_of",
    "evaluate_events",
    "progressive_val_proba_score",
    "progressive_val_score",
]
