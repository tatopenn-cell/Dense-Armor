"""One adapter that opens any Dense-Armor primitive to text."""

from typing import Any

from dense_armor.core._no_samples import NoSamples
from dense_armor.core.audio.adapter import AudioInput
from dense_armor.core.signals.preprocessing.vectorizer import TextVectorizer

N_TERMS = "n_terms"


class _Terms(dict):
    """The scores of one document. A term it does not have reads as 0.0 and is not added.

    The key ``n_terms`` reads as the number of terms the document has, so a primitive that takes
    one number (a drift detector) gets a number for every document.
    """

    def __missing__(self, key: str) -> float:
        return float(len(self)) if key == N_TERMS else 0.0


class TextInput(AudioInput):
    """Feed documents to a primitive that already exists, in the form it takes.

    The text is turned into the flat dict of numbers that the library already reads by the
    existing :class:`TextVectorizer` (its ``scores``). What the primitive is called with is
    decided by :class:`~dense_armor.core.audio.adapter.AudioInput`, which this class extends
    without copying it: a drift detector gets one number, one with ``score_one`` scores and
    learns, one with ``predict_one`` and ``learn_one`` predicts and learns with the label ``y``
    (``NeedsLabels`` if there is none), one with ``transform_one`` learns and transforms.

    A primitive with only ``update_many`` takes samples of a signal: for it ``feed`` raises
    ``ValueError``.

    Args:
        primitive: any object of Dense-Armor with the methods above.
        key: the entry given to scalar primitives: a term, or ``"n_terms"`` (the default), the
            number of terms of the document.
        vectorizer: a :class:`TextVectorizer` to share between adapters. Each document is
            learned by it and then transformed.

    Example:
        >>> from dense_armor.core.online.anomaly.mahalanobis import OnlineRobustMahalanobis
        >>> a = TextInput(OnlineRobustMahalanobis(feature_keys=["n_terms"]))
        >>> _ = a.feed("the cat sat on the mat")
        >>> a.mode, a.n
        ('score', 1)
    """

    def __init__(
        self,
        primitive: Any,
        key: str = N_TERMS,
        vectorizer: TextVectorizer | None = None,
    ) -> None:
        super().__init__(primitive, key, scaling=None)
        self.vectorizer = vectorizer if vectorizer is not None else TextVectorizer()

    def feed(self, text: Any, y: Any = None, x: dict[str, float] | None = None) -> Any:
        """Give one document to the primitive and return what it answers.

        ``x`` is the dict of scores of the document, if the caller has it already.
        """
        if x is None:
            self.vectorizer.learn_one(text)
            x = self.vectorizer.transform_one(text)["scores"]
        return super().feed(NoSamples("text"), y, _Terms(x))
