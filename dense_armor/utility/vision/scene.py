"""Scene changes and temporal anomalies.

Two transformers, both causal:

- :class:`SceneChange` turns a frame into a scalar distance from the
  previous frame (histogram distance plus mean absolute difference) and
  feeds it to a drift detector of the library
  (:class:`~dense_armor.utility.drift.cusum.CUSUMDriftDetector`,
  :class:`~dense_armor.utility.drift.page_hinkley.PageHinkley`,
  :class:`~dense_armor.utility.drift.adwin.ADWIN`). A hard cut in a video
  is a level shift in the distance stream, which is exactly what a
  detector for a scalar stream was built for. The detectors are existing
  components; the paper of each is in its own module.

- :class:`FrameAnomaly` scores every frame with the patch memory of the
  library (:class:`~dense_armor.utility.vision.patches.PatchMemory`),
  which follows the PatchCore idea: a memory bank of nominal
  patch-features, the score is the distance to the nearest neighbour. A
  frame whose patches are far from the bank is anomalous.

The parameters of the detectors are set in the constructor, so they are
chosen before the result is seen.

References:
    Reynolds, M. R. et al. (1975). The CUSUM chart with fast initial
        response. Reynolds, M. R. (1971). Approximations to the average
        run length of the CUSUM. Technometrics.
    Bifet, A., Gavalda, R. (2007). Learning from time-changing data with
        adaptive windowing. In SIAM SDM.
    Raab, C. et al. (2020). Reactive soft prototype computing for concept
        drift. Neurocomputing 416, 340-351.
    Roth, K. et al. (2022). Towards total recall in industrial anomaly
        detection. In CVPR.
"""

import numpy as np

from dense_armor.roles import Transformer
from dense_armor.utility.vision.patches import PatchFeatures, PatchMemory


def _to_gray(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 3 and a.shape[2] >= 3:
        return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    if a.ndim == 3:
        return a[..., 0]
    return a


def _histogram(img: np.ndarray, bins: int, lo: float, hi: float) -> np.ndarray:
    h, _ = np.histogram(img, bins=bins, range=(lo, hi))
    s = h.sum()
    return h / s if s > 0 else h.astype(np.float64)


class SceneChange(Transformer):
    """Scalar distance between consecutive frames, given to a drift detector.

    The distance combines two ingredients, both robust and content-free:

    - the L1 distance between the intensity histograms of the two frames,
      normalised by the number of bins, which is invariant to the
      spatial layout and reacts to exposure and content changes;
    - the mean absolute difference of the two frames after a small box
      blur, which reacts to any spatial change.

    The scalar distance is fed to a drift detector of the library; the
    detector flags a change when the distance jumps. The detector's
    ``drift_detected`` attribute is exposed as ``scene_change``.

    Args:
        detector: a drift detector from
            :mod:`dense_armor.utility.drift` (``update`` +
            ``drift_detected``). ``None`` uses
            :class:`~dense_armor.utility.drift.adwin.ADWIN`.
        bins: number of histogram bins. Default ``32``.
        blur: box filter radius on the frame difference. Default ``1``.
        downsample: integer factor applied to the image before the mean
            absolute difference. Default ``2``.

    Raises:
        ValueError: on a non-positive ``bins`` or a ``downsample`` below
            one.
        TypeError: if ``detector`` does not expose ``update``.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.vision.streams import ArrayStream
        >>> from dense_armor.utility.vision.scene import SceneChange
        >>> rng = np.random.default_rng(0)
        >>> frames = []
        >>> for i in range(60):
        ...     frames.append(rng.normal(0.5, 0.05, (16, 16)).astype(np.float32))
        >>> for i in range(60):
        ...     frames.append(np.full((16, 16), 0.95, dtype=np.float32))
        >>> sc = SceneChange()
        >>> dists = []
        >>> for fr in ArrayStream(np.stack(frames), fps=10.0):
        ...     out = sc.transform_one(fr)
        ...     _ = sc.learn_one(fr)
        ...     dists.append(out["distance"])
        >>> dists[60] > 0.5
        True
        >>> max(dists[61:]) < 0.05
        True
    """

    memory_class = "O(HW)"

    def __init__(
        self,
        detector=None,
        bins: int = 32,
        blur: int = 1,
        downsample: int = 2,
    ) -> None:
        if bins < 1:
            raise ValueError(f"bins must be >= 1, got {bins}")
        if blur < 0:
            raise ValueError(f"blur must be >= 0, got {blur}")
        if downsample < 1:
            raise ValueError(f"downsample must be >= 1, got {downsample}")
        if detector is None:
            from dense_armor.utility.drift.adwin import ADWIN
            detector = ADWIN()
        if not hasattr(detector, "update"):
            raise TypeError("detector must expose update()")
        self.detector = detector
        self.bins = int(bins)
        self.blur = int(blur)
        self.downsample = int(downsample)
        self._prev: np.ndarray | None = None
        self._prev_hist: np.ndarray | None = None

    def _reset(self) -> None:
        self._prev = None
        self._prev_hist = None

    def _unit_test_skips(self) -> set:
        return set()

    def _prep(self, img) -> np.ndarray:
        if isinstance(img, dict):
            img = img["array"]
        elif hasattr(img, "array"):
            img = img.array
        g = _to_gray(np.asarray(img))
        if self.downsample > 1:
            f = self.downsample
            h, w = g.shape[0] // f, g.shape[1] // f
            g = g[: h * f, : w * f].reshape(h, f, w, f).mean(axis=(1, 3))
        return g

    def learn_one(self, x, y=None, t=None):
        g = self._prep(x)
        self._prev = g
        self._prev_hist = _histogram(g, self.bins, 0.0, 1.0)
        return self

    def transform_one(self, x, t=None):
        g = self._prep(x)
        if self._prev is None or self._prev.shape != g.shape:
            d = 0.0
        else:
            h_cur = _histogram(g, self.bins, 0.0, 1.0)
            hist_dist = float(np.abs(h_cur - self._prev_hist).sum()) / 2.0
            diff = np.abs(g - self._prev)
            if self.blur > 0:
                k = 2 * self.blur + 1
                pad = np.pad(diff, self.blur, mode="edge")
                cs = np.cumsum(pad, axis=0)
                cs = np.vstack([np.zeros((1, cs.shape[1]), cs.dtype), cs])
                rows = (cs[k:] - cs[:-k]) / k
                cs = np.cumsum(rows, axis=1)
                cs = np.hstack([np.zeros((cs.shape[0], 1), cs.dtype), cs])
                diff = (cs[:, k:] - cs[:, :-k]) / k
            d = 0.5 * hist_dist + 0.5 * float(diff.mean())
        self.detector.update(float(d))
        return {
            "distance": float(d),
            "scene_change": int(bool(getattr(self.detector, "drift_detected", False))),
        }


class FrameAnomaly(Transformer):
    """Score a frame with the patch memory of the library.

    The frame is turned into patch descriptors by
    :class:`~dense_armor.utility.vision.patches.PatchFeatures`, and the
    descriptors are scored against a
    :class:`~dense_armor.utility.vision.patches.PatchMemory` built from
    the frames already seen. The memory is bounded in reservoir mode, or
    built from a coreset in coreset mode; see the ``PatchMemory``
    docstring.

    Args:
        patch_features: a configured :class:`PatchFeatures`, or ``None``
            for the default one.
        memory: a configured :class:`PatchMemory`, or ``None`` for a
            reservoir memory of size 128.
        learn: whether :meth:`learn_one` adds the frame to the memory.
            Default ``True``.

    Raises:
        TypeError: if ``patch_features`` is not a ``PatchFeatures`` or
            ``memory`` is not a ``PatchMemory``.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.vision.streams import ArrayStream
        >>> from dense_armor.utility.vision.scene import FrameAnomaly
        >>> rng = np.random.default_rng(0)
        >>> normal = [rng.normal(0.5, 0.02, (16, 16)).astype(np.float32)
        ...           for _ in range(40)]
        >>> fa = FrameAnomaly()
        >>> for fr in ArrayStream(np.stack(normal), fps=10.0):
        ...     _ = fa.learn_one(fr)
        >>> normal_scores = [
        ...     fa.transform_one(fr)["anomaly"]
        ...     for fr in ArrayStream(np.stack(normal), fps=10.0)
        ... ]
        >>> anomalous = []
        >>> for _ in range(10):
        ...     f = np.zeros((16, 16), dtype=np.float32)
        ...     f[::2, ::2] = 1.0
        ...     f[1::2, 1::2] = 1.0
        ...     anomalous.append(f)
        >>> anomalous_scores = [
        ...     fa.transform_one(fr)["anomaly"]
        ...     for fr in ArrayStream(np.stack(anomalous), fps=10.0)
        ... ]
        >>> round(float(np.mean(normal_scores)), 4) < round(
        ...     float(np.mean(anomalous_scores)), 4)
        True
    """

    memory_class = "O(window)"

    def __init__(self, patch_features=None, memory=None, learn: bool = True) -> None:
        if patch_features is None:
            patch_features = PatchFeatures()
        if not isinstance(patch_features, PatchFeatures):
            raise TypeError(
                f"patch_features must be a PatchFeatures, "
                f"got {type(patch_features).__name__}"
            )
        if memory is None:
            memory = PatchMemory(max_size=1024)
        if not isinstance(memory, PatchMemory):
            raise TypeError(
                f"memory must be a PatchMemory, "
                f"got {type(memory).__name__}"
            )
        self.patch_features = patch_features
        self.memory = memory
        self.learn = bool(learn)

    def _reset(self) -> None:
        if hasattr(self.patch_features, "_reset"):
            self.patch_features._reset()
        if hasattr(self.memory, "_reset"):
            self.memory._reset()

    def _unit_test_skips(self) -> set:
        return set()

    def _prepare(self, x) -> dict:
        out = self.patch_features.transform_one(x)
        return {
            "descriptors": out.get("descriptors"),
            "positions": out.get("positions"),
        }

    def transform_one(self, x, t=None):
        d = self._prepare(x)
        try:
            s = self.memory.score_one(d)
        except Exception:
            s = 0.0
        return {"anomaly": float(s)}

    def learn_one(self, x, y=None, t=None):
        _ = self.patch_features.learn_one(x)
        if self.learn:
            d = self._prepare(x)
            try:
                self.memory.learn_one(d)
            except Exception:
                pass
        return self
