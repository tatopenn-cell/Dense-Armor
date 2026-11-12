"""One adapter that opens any Dense-Armor primitive to sound."""

from collections.abc import Callable, Iterable
from typing import Any

from dense_armor.utility.audio.features import AudioFeatures
from dense_armor.utility.audio.signal import Audio

_DICT_ERRORS = (TypeError, ValueError, KeyError, AttributeError, IndexError)


class NeedsLabels(ValueError):
    """The primitive learns from labels and ``feed`` got none."""


def _try(fn, x, scalar):
    try:
        return fn(x), "dict"
    except _DICT_ERRORS:
        return fn(scalar), "scalar"


class AudioInput:
    """Feed blocks of sound to a primitive that already exists, in the form it takes.

    The primitive is not copied or rewritten: the adapter looks at its methods and calls them.

    * a drift detector (``update`` and ``drift_detected``) gets the feature ``key`` as a scalar;
    * one with ``score_one`` scores the block and then learns it;
    * one with ``predict_one`` and ``learn_one`` predicts and then learns with the label ``y``
      (``NeedsLabels`` if there is none);
    * one with ``transform_one`` learns and transforms;
    * any other with ``learn_one`` just learns;
    * one with only ``update_many`` (a filter, ``SDFT``) gets the samples of the block, mixed to
      one channel.

    The dict form is the one of :class:`AudioFeatures`. If the first call with the dict raises
    ``TypeError``, ``ValueError``, ``KeyError``, ``AttributeError`` or ``IndexError``, the
    primitive is called again with the feature ``key`` as a scalar, and from then on gets the
    scalar.

    Args:
        primitive: any object of Dense-Armor with the methods above.
        key: the feature given to scalar primitives.
        n_bands: bands of :class:`AudioFeatures`.
        features: an :class:`AudioFeatures` to share between adapters.

    Example:
        >>> from dense_armor.utility.signals.stats.sdft import SDFT
        >>> import numpy as np
        >>> a = AudioInput(SDFT(4))
        >>> _ = a.feed(Audio(np.array([1.0, 2.0, 3.0, 4.0]), 4))
        >>> a.mode, a.n
        ('samples', 1)
    """

    def __init__(
        self,
        primitive: Any,
        key: str = "b0",
        n_bands: int = 8,
        features: AudioFeatures | None = None,
    ) -> None:
        self.primitive = primitive
        self.key = key
        self.features = features if features is not None else AudioFeatures(n_bands)
        self.mode: str | None = None
        self.scalar = False
        self.n = 0
        self.last: Any = None
        self.alarms = 0

    def feed(
        self, block: Audio, y: Any = None, x: dict[str, float] | None = None
    ) -> Any:
        """Give one block to the primitive and return what it answers.

        ``x`` is the feature dict of the block, if the caller has it already.
        """
        e = self.primitive
        out: Any = None
        is_drift = hasattr(e, "drift_detected") and hasattr(e, "update")
        takes_features = is_drift or any(
            hasattr(e, m)
            for m in ("score_one", "predict_one", "transform_one", "learn_one")
        )
        if not takes_features and hasattr(e, "update_many"):
            self.mode = "samples"
            out = e.update_many(block.to_mono().x)
            self.n += 1
            self.last = out
            return out
        if x is None:
            x = self.features.transform(block)
        s = x[self.key]
        if is_drift:
            self.mode = "drift"
            e.update(s)
            out = bool(e.drift_detected)
            self.alarms += int(out)
        elif hasattr(e, "score_one"):
            self.mode = "score"
            if self.scalar:
                out = e.score_one(s)
                e.learn_one(s)
            else:
                out, how = _try(e.score_one, x, s)
                self.scalar = how == "scalar"
                e.learn_one(s if self.scalar else x)
        elif hasattr(e, "predict_one") and hasattr(e, "learn_one"):
            if y is None:
                self.mode = "needs labels"
                raise NeedsLabels("needs labels")
            self.mode = "supervised"
            arg = s if self.scalar else x
            out = e.predict_one(arg)
            e.learn_one(arg, y)
        elif hasattr(e, "transform_one"):
            self.mode = "transform"
            if self.scalar:
                e.learn_one(s)
                out = e.transform_one(s)
            else:
                _, how = _try(e.learn_one, x, s)
                self.scalar = how == "scalar"
                out = e.transform_one(s if self.scalar else x)
        elif hasattr(e, "learn_one"):
            self.mode = "learn"
            e.learn_one(x)
        else:
            raise ValueError("no known method")
        self.n += 1
        self.last = out
        return out

    def run(
        self, blocks: Iterable[Audio], label: Callable[[Audio], Any] | None = None
    ) -> list:
        """Feed every block, from a file stream or a live stream; return the answers."""
        return [self.feed(b, label(b) if label is not None else None) for b in blocks]


def run_estimators(
    blocks: Iterable[Audio],
    estimators: dict[str, Any],
    key: str = "b0",
    label: Callable[[Audio], Any] | None = None,
    n_bands: int = 8,
) -> dict[str, dict[str, Any]]:
    """Feed the blocks to every primitive of ``estimators`` through one :class:`AudioInput` each.

    The features of a block are computed once and shared. A primitive that raises is reported
    with the first error and not called again.

    Args:
        blocks: audio blocks, from a file stream or a live stream.
        estimators: name to primitive.
        key: the feature given to scalar primitives.
        label: optional function from a block to a label or target.
        n_bands: bands of :class:`AudioFeatures`.

    Returns:
        For each name: ``mode``, ``n`` (blocks processed), ``last`` (the last answer),
        ``alarms`` (blocks where a drift was detected), ``error`` and ``scalar`` (True when the
        primitive was switched to the scalar form).
    """
    fe = AudioFeatures(n_bands)
    adapters = {k: AudioInput(e, key, features=fe) for k, e in estimators.items()}
    errors: dict[str, str | None] = {k: None for k in estimators}
    for blk in blocks:
        x = fe.transform(blk)
        y = label(blk) if label is not None else None
        for name, a in adapters.items():
            if errors[name] is not None:
                continue
            try:
                a.feed(blk, y, x)
            except NeedsLabels:
                errors[name] = "needs labels"
            except Exception as ex:
                errors[name] = f"{type(ex).__name__}: {ex}"
    return {
        k: {
            "mode": a.mode,
            "n": a.n,
            "last": a.last,
            "alarms": a.alarms,
            "error": errors[k],
            "scalar": a.scalar,
        }
        for k, a in adapters.items()
    }
