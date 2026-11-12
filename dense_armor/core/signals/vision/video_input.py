"""One adapter that opens any Dense-Armor primitive to video."""

from typing import Any

from dense_armor.core._no_samples import NoSamples
from dense_armor.core.audio.adapter import AudioInput
from dense_armor.core.signals.vision.features import FrameFeatures


class VideoInput(AudioInput):
    """Feed frames to a primitive that already exists, in the form it takes.

    A frame (a ``Frame``, a dict with ``"frame"``, or an array) is turned into the flat dict of
    numbers that the library already reads by the existing :class:`FrameFeatures`. What the
    primitive is called with is decided by
    :class:`~dense_armor.core.audio.adapter.AudioInput`, which this class extends without
    copying it: a drift detector gets one number, one with ``score_one`` scores and learns, one
    with ``predict_one`` and ``learn_one`` predicts and learns with the label ``y``
    (``NeedsLabels`` if there is none), one with ``transform_one`` learns and transforms.

    ``FrameFeatures`` keeps the previous frame to compute the optical flow, so the frames have
    to come in order, one adapter for one stream.

    A primitive with only ``update_many`` takes samples of a signal: for it ``feed`` raises
    ``ValueError``.

    Args:
        primitive: any object of Dense-Armor with the methods above.
        key: the feature given to scalar primitives (``"mean"``, the mean luminance, by
            default).
        extractor: a :class:`FrameFeatures` to use, for another grid or for colour images.

    Example:
        >>> import numpy as np
        >>> from dense_armor.core.online.anomaly.mahalanobis import OnlineRobustMahalanobis
        >>> a = VideoInput(OnlineRobustMahalanobis(feature_keys=["mean", "contrast"]))
        >>> _ = a.feed(np.zeros((6, 8), dtype=np.float32))
        >>> a.mode, a.n
        ('score', 1)
    """

    def __init__(
        self,
        primitive: Any,
        key: str = "mean",
        extractor: FrameFeatures | None = None,
    ) -> None:
        super().__init__(primitive, key, scaling=None)
        self.extractor = extractor if extractor is not None else FrameFeatures()

    def feed(self, frame: Any, y: Any = None, x: dict[str, float] | None = None) -> Any:
        """Give one frame to the primitive and return what it answers.

        ``x`` is the feature dict of the frame, if the caller has it already.
        """
        if x is None:
            x = self.extractor.transform_one(frame)
        return super().feed(NoSamples("video"), y, x)
