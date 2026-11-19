# BagOfVisualWords: a visual vocabulary

`BagOfVisualWords` cuts a frame into patches, describes each patch by the directions of its edges, learns a vocabulary of words with an internal `OnlineKMeans`, and turns a frame into a histogram of words. Its code is in `core/signals/vision`; it is on this page because it is the k-means applied to images. The overview of the section is on [Clustering](index.md).

## A visual vocabulary of image patches

An image is cut into small patches; each patch is described by the directions of its edges;
online k-means groups the patch descriptions into "visual words"; a frame becomes a histogram
of the words it contains.

```python
import numpy as np
from dense_armor.core.signals.vision.vocabulary import BagOfVisualWords

vert = np.zeros((32, 32), dtype=np.float32)
vert[:, ::4] = 1.0
horz = vert.T.copy()
bow = BagOfVisualWords(n_words=4, patch_size=8, n_bins=4)
for _ in range(10):
    bow.learn_one({"frame": vert})
    bow.learn_one({"frame": horz})
print(bow.transform_one({"frame": vert})["hist"], bow.transform_one({"frame": horz})["hist"])
```

```
[0.0, 0.25, 0.0, 0.75] [0.25, 0.0, 0.75, 0.0]
```

Vertical stripes and horizontal stripes use different words, so their histograms do not overlap:
two textures, two signatures. The vocabulary is learned online with `OnlineKMeans` on the
descriptors of the 8×8 patches, so the same estimator can be fed patches from any number of
frames without ever storing them; a frame becomes a fixed-length histogram, a signature ready
for a classifier.

## The parameters

| Parameter | Meaning | Default |
|---|---|---|
| `n_words` | size of the visual vocabulary | 8 |
| `patch_size` | side of a square patch in pixels | 8 |
| `n_bins` | number of gradient orientation bins inside a patch | 8 |
| `seed` | passed to the internal `OnlineKMeans` | 0 |

A frame that is empty, has a wrong shape or contains a value that is not finite is skipped and counted in `n_missing`.

## API reference

::: dense_armor.core.signals.vision.vocabulary

---

## Details

- Sources: Huang, C., Huang, J. (2017), "A fast HOG descriptor using lookup table and integral image", arXiv:1703.06256.
