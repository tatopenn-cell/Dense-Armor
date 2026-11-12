"""From a sound to an alarm: block features, a causal Hampel score and a drift detector."""

from collections.abc import Iterable

from dense_armor.core.audio.features import AudioFeatures
from dense_armor.core.audio.signal import Audio
from dense_armor.core.online.anomaly.filters import HampelScorer
from dense_armor.core.online.drift.detector import CUSUMDriftDetector


def detect_events(
    blocks: Iterable[Audio], key: str = "b0", n_bands: int = 8
) -> list[tuple[float, float]]:
    """Return ``(t, score)`` for every block where the drift detector fires.

    For each block the feature ``key`` of :class:`AudioFeatures` is scored by a
    :class:`HampelScorer` (the score of a value against the median and MAD of the values
    that precede it) and the score is given to a :class:`CUSUMDriftDetector`. ``t`` is the
    start time of the block. The same function reads a file stream and a live stream.

    Args:
        blocks: audio blocks.
        key: which feature to follow.
        n_bands: bands of :class:`AudioFeatures`.

    Returns:
        The list of ``(t, score)`` at the alarms.
    """
    fe = AudioFeatures(n_bands)
    sc = HampelScorer()
    dd = CUSUMDriftDetector()
    out = []
    for b in blocks:
        s = fe.transform(b)[key]
        score = sc.score_one({key: s})
        sc.learn_one({key: s})
        dd.update(score)
        if dd.drift_detected:
            out.append((b.t0, float(score)))
    return out
