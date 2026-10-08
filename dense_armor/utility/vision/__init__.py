"""Native online vision for Dense-Armor.

Give the robot a camera, or a folder of images, and the library learns
from it the same way it learns from a URDF: one frame at a time, in
fixed memory, with no pretrained network and no external model. Every
feature is computed by the library itself in numpy; the only optional
dependency is the camera driver (OpenCV, extra ``[vision]``), and it is
imported lazily.

The three steps of the pipeline
-------------------------------

A stream produces one :class:`~dense_armor.utility.vision.streams.Frame`
per sample, with an image and a timestamp. A transformer turns the
frame into a flat dict of numbers: the oriented gradients of the image
on a grid, the intensity moments, and the short-range optical flow
against the previous frame. Two more transformers reduce the numbers
and learn the structure: a fixed random projection that preserves the
pairwise distances (Johnson and Lindenstrauss), and an online PCA that
learns the top directions as they come in (Oja's rule). The reduced
features feed any anomaly detector of the library, for example the
robust Mahalanobis distance in
:mod:`dense_armor.utility.anomaly.mahalanobis`.

A full pipeline in one place
----------------------------

The pipeline below is on an :class:`ArrayStream` of synthetic frames:
a bright square moves, then its shape changes and a second object
appears. After the change, the anomaly score rises. Run it as a script
and it prints the score before and after the change.

    >>> import numpy as np
    >>> from dense_armor.utility.vision.streams import ArrayStream
    >>> from dense_armor.utility.vision.features import FrameFeatures
    >>> from dense_armor.utility.vision.reduce import IncrementalPCA
    >>> from dense_armor.utility.anomaly.mahalanobis import (
    ...     OnlineRobustMahalanobis,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> frames = []
    >>> for i in range(80):
    ...     f = np.zeros((32, 32), dtype=np.float32)
    ...     x = min(24, 4 + i // 4)
    ...     f[8:16, x:x + 8] = 1.0
    ...     frames.append(f)
    >>> for i in range(20):
    ...     f = np.zeros((32, 32), dtype=np.float32)
    ...     f[8:20, 8:20] = 1.0
    ...     f[24:30, 24:30] = 0.8
    ...     frames.append(f)
    >>> stream = ArrayStream(np.stack(frames), fps=10.0)
    >>> feats = FrameFeatures()
    >>> pca = IncrementalPCA(k=4)
    >>> det = OnlineRobustMahalanobis(
    ...     feature_keys=[f"pc{j}" for j in range(4)]
    ... )
    >>> scores = []
    >>> for frame in stream:
    ...     f = feats.transform_one(frame)
    ...     _ = feats.learn_one(frame)
    ...     p = pca.transform_one(f)
    ...     _ = pca.learn_one(f)
    ...     s = det.score_one(p)
    ...     _ = det.learn_one(p)
    ...     scores.append(s)
    >>> round(float(np.mean(scores[:80])), 3) < round(
    ...     float(np.mean(scores[80:])), 3)
    True
"""

from dense_armor.utility.vision.features import FrameFeatures
from dense_armor.utility.vision.patches import (
    PatchFeatures,
    PatchMemory,
)
from dense_armor.utility.vision.reduce import (
    IncrementalPCA,
    RandomProjection,
)
from dense_armor.utility.vision.streams import (
    ArrayStream,
    CameraStream,
    Frame,
    ImageFolderStream,
)

__all__ = [
    "ArrayStream",
    "CameraStream",
    "Frame",
    "FrameFeatures",
    "ImageFolderStream",
    "IncrementalPCA",
    "PatchFeatures",
    "PatchMemory",
    "RandomProjection",
]
