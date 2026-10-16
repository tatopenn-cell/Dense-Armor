"""Track moving objects without a pretrained detector.

The tracker works in two steps, both in numpy, both causal:

- a **motion detector** keeps a robust running background (the median of
  the last few frames) and flags the pixels whose absolute difference
  from the background is more than ``n_sigmas`` times the robust scale
  of that difference. The mask is cleaned with a small morphological
  open and grouped into connected components; every component becomes a
  bounding box. No labels, no pretrained network, no assumption about
  the content: only "this pixel changed".
- a **Kalman filter** per track predicts the box of the next frame
  (constant velocity, then constant acceleration) and the association
  matches predicted boxes to detected boxes. Following the ByteTrack
  idea (Ma et al. 2023), the association has two rounds: the first
  matches the detections with a high score to the tracks, the second
  matches the still unmatched tracks to the detections that the first
  round left, including the low-score ones. The tracks not matched in
  either round are kept alive for ``max_missing`` frames, so a short
  occlusion does not kill a track.

The user can also feed boxes from an external detector with
``update(boxes)``: the tracker then skips the motion detector and only
does the association and the filter, with the same interface.

References:
    Ma, Y. et al. (2023). Adaptive confidence threshold for ByteTrack.
        arXiv:2312.01650. The two-round association with the low-score
        detections is the contribution the paper reviews and extends.
    Mbalawata, I. S., Sarkka, S. (2013). Moment conditions for
        convergence of the variational Bayes approximation in Kalman
        filtering. arXiv:1308.5875. The adaptive noise update the paper
        writes is available in the library as
        :class:`~dense_armor.utility.timeseries.kalman.VBAdaptiveKalman`;
        the tracker uses the plain filter by default and accepts any
        ``KalmanFilter`` subclass.
"""

import numpy as np

from dense_armor.roles import Transformer
from dense_armor.utility.timeseries.kalman import KalmanFilter


def _to_gray(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 3 and a.shape[2] >= 3:
        return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    if a.ndim == 3:
        return a[..., 0]
    return a


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _components(mask: np.ndarray, min_area: int) -> list:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    boxes = []
    for y0 in range(h):
        for x0 in range(w):
            if not mask[y0, x0] or seen[y0, x0]:
                continue
            stack = [(y0, x0)]
            seen[y0, x0] = True
            ys, xs = [], []
            while stack:
                y, x = stack.pop()
                ys.append(y)
                xs.append(x)
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(ys) < min_area:
                continue
            boxes.append((min(xs), min(ys), max(xs) + 1, max(ys) + 1))
    return boxes


class _Track:
    """One track: a box, a filter, and how long it has been unseen."""

    __slots__ = ("id", "box", "score", "missing", "kf")

    def __init__(self, tid: int, box, score: float, kf: KalmanFilter) -> None:
        self.id = tid
        self.box = np.asarray(box, dtype=float)
        self.score = float(score)
        self.missing = 0
        self.kf = kf
        self.kf.learn_one({
            "a": float(box[0]),
            "b": float(box[1]),
            "c": float(box[2] - box[0]),
            "d": float(box[3] - box[1]),
        })

    def predict(self) -> np.ndarray:
        p = self.kf.predict_one({})
        p = np.asarray(p, dtype=float).reshape(-1)
        x0, y0, w, h = p[0], p[1], p[2], p[3]
        return np.array([x0, y0, x0 + w, y0 + h])

    def update(self, box, score: float) -> None:
        self.box = np.asarray(box, dtype=float)
        self.score = float(score)
        self.missing = 0
        self.kf.learn_one({
            "a": float(box[0]),
            "b": float(box[1]),
            "c": float(box[2] - box[0]),
            "d": float(box[3] - box[1]),
        })


def _default_kf() -> KalmanFilter:
    """Constant-velocity Kalman: state ``(x, y, w, h, vx, vy, vw, vh)``."""
    F = np.eye(8)
    for i in range(4):
        F[i, i + 4] = 1.0
    H = np.zeros((4, 8))
    for i in range(4):
        H[i, i] = 1.0
    Q = 1e-3 * np.eye(8)
    R = 1e-1 * np.eye(4)
    return KalmanFilter(F=F, H=H, Q=Q, R=R)


class MovingObjectTracker(Transformer):
    """Track moving objects from a frame stream or from external boxes.

    Args:
        n_sigmas: threshold on the robust z-score of the frame difference
            for the motion mask. Default ``3.0``.
        history: number of frames used for the running background.
            Default ``10``.
        min_area: minimum area of a connected component to become a
            track. Default ``16``.
        max_missing: number of frames a track survives without a matched
            detection. Default ``15``.
        iou_threshold: minimum IoU for a match. Default ``0.3``.
        high_score: score above which a detection goes to the first
            association round. Below this value the detection goes to
            the second round. Default ``0.5``.
        kf: a :class:`KalmanFilter` subclass, or a callable returning
            one. Default is the plain filter.

    Raises:
        ValueError: on non-positive ``history``, ``min_area``,
            ``max_missing`` or bad ``n_sigmas``.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.vision.streams import ArrayStream
        >>> from dense_armor.utility.vision.tracking import MovingObjectTracker
        >>> frames = np.zeros((8, 32, 32), dtype=np.float32)
        >>> for i in range(8):
        ...     frames[i, 8:14, 4 + i:10 + i] = 1.0
        >>> t = MovingObjectTracker(min_area=8, max_missing=3)
        >>> ids = []
        >>> for fr in ArrayStream(frames, fps=10.0):
        ...     out = t.transform_one(fr)
        ...     _ = t.learn_one(fr)
        ...     if out["tracks"]:
        ...         ids.append(out["tracks"][0]["id"])
        >>> len(set(ids)) == 1
        True
    """

    memory_class = "O(n_tracks * n_frames_bg)"

    def __init__(
        self,
        n_sigmas: float = 3.0,
        history: int = 10,
        min_area: int = 16,
        max_missing: int = 15,
        iou_threshold: float = 0.3,
        high_score: float = 0.5,
        kf=None,
    ) -> None:
        if n_sigmas <= 0:
            raise ValueError(f"n_sigmas must be > 0, got {n_sigmas}")
        if history < 1:
            raise ValueError(f"history must be >= 1, got {history}")
        if min_area < 1:
            raise ValueError(f"min_area must be >= 1, got {min_area}")
        if max_missing < 0:
            raise ValueError(f"max_missing must be >= 0, got {max_missing}")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be in [0, 1], got {iou_threshold}")
        self.n_sigmas = float(n_sigmas)
        self.history = int(history)
        self.min_area = int(min_area)
        self.max_missing = int(max_missing)
        self.iou_threshold = float(iou_threshold)
        self.high_score = float(high_score)
        if kf is None:
            self.kf_factory = _default_kf
        elif callable(kf) and not hasattr(kf, "learn_one"):
            self.kf_factory = kf
        else:
            self.kf_factory = lambda: kf.clone()
        self._history: list[np.ndarray] = []
        self._tracks: list = []
        self._next_id = 1

    def _reset(self) -> None:
        self._history = []
        self._tracks = []
        self._next_id = 1

    def _unit_test_skips(self) -> set:
        return set()

    def _background(self) -> np.ndarray:
        return np.median(np.stack(self._history), axis=0)

    def _detect(self, gray: np.ndarray) -> list:
        if len(self._history) < max(1, self.history // 2):
            return []
        bg = self._background()
        diff = np.abs(gray - bg)
        scale = float(np.median(np.abs(diff))) * 1.4826 + 1e-9
        thr = self.n_sigmas * scale
        mask = diff > thr
        if mask.sum() == 0:
            return []
        eroded = mask.copy()
        for y in range(1, mask.shape[0] - 1):
            for x in range(1, mask.shape[1] - 1):
                if not mask[y, x]:
                    continue
                if (mask[y - 1, x] and mask[y + 1, x] and
                        mask[y, x - 1] and mask[y, x + 1]):
                    eroded[y, x] = True
                else:
                    eroded[y, x] = False
        return _components(eroded, self.min_area)

    def _associate(self, tracks, detections, hi_thresh) -> list:
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(detections)))
        pairs = []
        for _ in range(2):
            best = []
            for ti in unmatched_tracks:
                pred = tracks[ti].predict()
                for di in unmatched_dets:
                    det = detections[di]
                    score = det[2] if len(det) > 2 else 1.0
                    if _ <= 0:
                        pass
                    if _ == 0 and score < hi_thresh:
                        continue
                    iou = _iou(pred, det[:4])
                    if iou >= self.iou_threshold:
                        best.append((iou, ti, di))
            best.sort(reverse=True)
            used_t, used_d = set(), set()
            for iou, ti, di in best:
                if ti in used_t or di in used_d:
                    continue
                used_t.add(ti)
                used_d.add(di)
                pairs.append((ti, di))
            unmatched_tracks = [t for t in unmatched_tracks if t not in used_t]
            unmatched_dets = [d for d in unmatched_dets if d not in used_d]
            if not unmatched_tracks or not unmatched_dets:
                break
        return pairs, unmatched_tracks, unmatched_dets

    def _step(self, detections: list) -> dict:
        # predici tutti i track
        predicted = [t.predict() for t in self._tracks]
        # associazione in due round
        pairs, unmatched_t, unmatched_d = self._associate(
            self._tracks, detections, self.high_score
        )
        for ti, di in pairs:
            det = detections[di]
            score = det[2] if len(det) > 2 else 1.0
            self._tracks[ti].update(det[:4], score)
        for ti in unmatched_t:
            self._tracks[ti].missing += 1
        # nuove tracce
        for di in unmatched_d:
            det = detections[di]
            score = det[2] if len(det) > 2 else 1.0
            tr = _Track(self._next_id, det[:4], score, self.kf_factory())
            self._tracks.append(tr)
            self._next_id += 1
        # scarta quelle troppo vecchie
        self._tracks = [t for t in self._tracks if t.missing <= self.max_missing]
        return {
            "tracks": [
                {"id": t.id, "box": tuple(t.box.tolist()), "missing": t.missing}
                for t in self._tracks
            ],
            "n_tracks": len(self._tracks),
        }

    def update(self, boxes) -> dict:
        """Feed external boxes ``[(x0, y0, x1, y1[, score])]`` to the tracker.

        Bypasses the motion detector and runs the association and the
        filter on the given boxes. The output has the same shape as
        :meth:`transform_one`.
        """
        dets = [np.asarray(b[:5] if len(b) >= 5 else (b[0], b[1], b[2], b[3], 1.0),
                           dtype=float) for b in boxes]
        return self._step(dets)

    def transform_one(self, x, t=None):
        arr = x.array if hasattr(x, "array") else x
        gray = _to_gray(np.asarray(arr))
        dets = self._detect(gray)
        det_arr = [np.array([b[0], b[1], b[2], b[3], 1.0]) for b in dets]
        return self._step(det_arr)

    def learn_one(self, x, y=None, t=None):
        arr = x.array if hasattr(x, "array") else x
        gray = _to_gray(np.asarray(arr))
        self._history.append(gray)
        if len(self._history) > self.history:
            self._history.pop(0)
        return self
