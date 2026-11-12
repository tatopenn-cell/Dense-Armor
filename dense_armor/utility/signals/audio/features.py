"""Frame features and a runner that feeds them to the online estimators of Dense-Armor."""

from collections.abc import Iterable
from typing import Any

import numpy as np

from dense_armor.roles import Root
from dense_armor.utility.signals.audio.signal import Audio
from dense_armor.utility.signals.audio.windows import get_window


class AudioFeatures(Root):
    """The magnitude spectrum of a block, averaged in ``n_bands`` groups of bins.

    The magnitude is ``|rfft(w x)|`` of the windowed block, the short-time Fourier transform
    of Allen and Rabiner (1977). Choice stated: the bins are split in ``n_bands`` groups of
    nearly equal size with ``numpy.array_split`` and each group is replaced by the mean
    magnitude; no source defines this grouping. The keys are ``b0 .. b{n_bands-1}``, fixed, so
    the dict goes straight into ``learn_one`` of a Dense-Armor estimator.

    Args:
        n_bands: number of groups, at least 1.
        window: a window name.

    Example:
        >>> f = AudioFeatures(n_bands=2)
        >>> sorted(f.transform(Audio(np.ones(16), 16)))
        ['b0', 'b1']
    """

    budget_s: float = 1e-3
    memory_class: str = "O(1)"

    def __init__(self, n_bands: int = 8, window: str = "hann") -> None:
        if n_bands < 1:
            raise ValueError("n_bands must be >= 1")
        self.n_bands = int(n_bands)
        self.window = window

    def transform(self, block: Audio) -> dict[str, float]:
        """Features of one mono block."""
        x: np.ndarray = np.asarray(block.to_mono().x, dtype=float)
        mag = np.abs(np.fft.rfft(x * get_window(self.window, len(x))))
        parts = np.array_split(mag, self.n_bands)
        return {f"b{i}": float(p.mean()) for i, p in enumerate(parts)}


_MODES = ("drift", "score", "supervised", "transform", "learn")


def _try(fn, x, scalar):
    try:
        return fn(x), "dict"
    except (TypeError, ValueError, KeyError, AttributeError, IndexError):
        return fn(scalar), "scalar"


def run_estimators(
    blocks: Iterable[Audio],
    estimators: dict[str, Any],
    key: str = "b0",
    label: Any = None,
    n_bands: int = 8,
) -> dict[str, dict[str, Any]]:
    """Feed the features of every block to every estimator, in order.

    For each estimator the call depends on what it has: a drift detector (``update`` and
    ``drift_detected``) gets the feature ``key`` as a scalar; an estimator with ``score_one``
    scores the block and then learns it; one with ``predict_one`` needs ``label(block)`` to
    learn (else it is reported as ``"needs labels"``); one with ``transform_one`` learns and
    transforms; any other with ``learn_one`` just learns. An estimator that takes a scalar
    instead of a dict (such as ``HampelScorer``) gets the feature ``key``. An estimator that
    raises is reported with the first error and not called again.

    Args:
        blocks: audio blocks, from a file stream or a live stream.
        estimators: name to estimator.
        key: the feature given to scalar estimators.
        label: optional function from a block to a label or target.
        n_bands: bands of :class:`AudioFeatures`.

    Returns:
        For each name: ``mode``, ``n`` (blocks processed), ``last`` (the last score, flag or
        transform), ``alarms`` (blocks where a drift was detected) and ``error``.
    """
    fe = AudioFeatures(n_bands)
    rep: dict[str, dict[str, Any]] = {
        k: {
            "mode": None,
            "n": 0,
            "last": None,
            "alarms": 0,
            "error": None,
            "scalar": False,
        }
        for k in estimators
    }
    for blk in blocks:
        x = fe.transform(blk)
        s = x[key]
        y = label(blk) if label is not None else None
        for name, e in estimators.items():
            r = rep[name]
            if r["error"] is not None:
                continue
            try:
                if hasattr(e, "drift_detected") and hasattr(e, "update"):
                    r["mode"] = "drift"
                    e.update(s)
                    r["last"] = bool(e.drift_detected)
                    r["alarms"] += int(r["last"])
                elif hasattr(e, "score_one"):
                    r["mode"] = "score"
                    if r["scalar"]:
                        r["last"] = e.score_one(s)
                        e.learn_one(s)
                    else:
                        r["last"], how = _try(lambda v: e.score_one(v), x, s)
                        r["scalar"] = how == "scalar"
                        e.learn_one(s if r["scalar"] else x)
                elif hasattr(e, "predict_one") and hasattr(e, "learn_one"):
                    if y is None:
                        r["mode"] = "needs labels"
                        r["error"] = "needs labels"
                        continue
                    r["mode"] = "supervised"
                    r["last"] = e.predict_one(x if not r["scalar"] else s)
                    e.learn_one(x if not r["scalar"] else s, y)
                elif hasattr(e, "transform_one"):
                    r["mode"] = "transform"
                    if r["scalar"]:
                        e.learn_one(s)
                        r["last"] = e.transform_one(s)
                    else:
                        _, how = _try(lambda v: e.learn_one(v), x, s)
                        r["scalar"] = how == "scalar"
                        r["last"] = e.transform_one(s if r["scalar"] else x)
                elif hasattr(e, "learn_one"):
                    r["mode"] = "learn"
                    e.learn_one(x)
                else:
                    r["error"] = "no known method"
                    continue
                r["n"] += 1
            except Exception as ex:
                r["error"] = f"{type(ex).__name__}: {ex}"
    return rep
