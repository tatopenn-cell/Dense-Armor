# Where is the anomaly: patch memory

Section 7 of the Vision page ends with a camera that notices a change: one number per frame.
A robot inspecting an object needs more: which part of the image is wrong. This page cuts each
frame into small patches, keeps a memory of the patches seen on normal images, and scores a new
image by its most unusual patch. The method is PatchCore (Roth et al. 2021); the patch
descriptors are computed by the library in numpy, with no pretrained network.

## 1. A frame becomes a grid of patches

A patch is a small square of the image. Each patch gets a short list of numbers that describes
the directions of its edges, its mean brightness and its contrast.

```python
import numpy as np
from dense_armor.utility.vision import PatchFeatures

img = np.zeros((64, 64), dtype=np.float32)
img[:, ::8] = 1.0
pf = PatchFeatures(patch=16, stride=8, n_bins=9, neighbourhood=3)
out = pf.transform_one(img)
print(out["descriptors"].shape, out["positions"].shape)
print(out["positions"][:3].tolist(), out["positions"][-1].tolist())
```

```
(49, 11) (49, 2)
[[0, 0], [0, 1], [0, 2]] [6, 6]
```

Patches of 16 × 16 pixels every 8 pixels give a 7 × 7 grid on a 64 × 64 frame, 49 patches. Each
descriptor has 11 numbers: a 9-bin histogram of edge directions (weighted by the edge strength,
L2-normalised), the mean and the standard deviation of the patch. `positions` gives the (row,
column) of each patch in the grid. Each descriptor is then averaged with its 3 × 3 neighbours, so
it also knows a little about its surroundings; the neighbourhood of PatchCore eq. 1:

$$\mathcal{N}_p^{(h,w)} = \{(a,b) \mid a \in [h-\lfloor p/2\rfloor, h+\lfloor p/2\rfloor],\ b \in [w-\lfloor p/2\rfloor, w+\lfloor p/2\rfloor]\}$$

with $p = 3$, the value the paper chooses from its Figure 4.

## 2. Learn what normal looks like, then score a scratch

The memory stores the patches of normal images; a new image is scored by the patch that is
farthest from everything in memory.

```python
import numpy as np
from dense_armor.utility.vision import PatchFeatures, PatchMemory

rng = np.random.default_rng(0)
pf = PatchFeatures(patch=16, stride=8)
mem = PatchMemory(max_size=5000)
for _ in range(20):
    img = np.zeros((64, 64), dtype=np.float32)
    img[:, ::8] = 1.0
    mem.learn_one(pf.transform_one(img + rng.normal(0, 0.05, img.shape)))
good = np.zeros((64, 64), dtype=np.float32)
good[:, ::8] = 1.0
bad = good.copy()
bad[40:44, 20:40] = 1.0
s_good = mem.score_one(pf.transform_one(good + rng.normal(0, 0.05, good.shape)))
s_bad = mem.score_one(pf.transform_one(bad + rng.normal(0, 0.05, bad.shape)))
print(round(s_good, 3), round(s_bad, 3))
```

```
0.032 0.424
```

Twenty noisy images of vertical stripes are the normal part. A new striped image scores 0.032;
the same image with a horizontal scratch scores 0.424, about thirteen times more. The image
score is PatchCore eq. 6: for each test patch, the distance to its nearest neighbour in memory;
the score is the largest of these distances:

$$s^* = \max_{m^{\text{test}} \in \mathcal{P}(x^{\text{test}})}\ \min_{m \in \mathcal{M}} \lVert m^{\text{test}} - m \rVert_2$$

## 3. Where the anomaly is

The per-patch distances form a map of the image; the largest one points at the scratch.

```python
import numpy as np
from dense_armor.utility.vision import PatchFeatures, PatchMemory

rng = np.random.default_rng(0)
pf = PatchFeatures(patch=16, stride=8)
mem = PatchMemory(max_size=5000)
for _ in range(20):
    img = np.zeros((64, 64), dtype=np.float32)
    img[:, ::8] = 1.0
    mem.learn_one(pf.transform_one(img + rng.normal(0, 0.05, img.shape)))
bad = np.zeros((64, 64), dtype=np.float32)
bad[:, ::8] = 1.0
bad[40:44, 20:40] = 1.0
scores, pos = mem.patch_scores(pf.transform_one(bad + rng.normal(0, 0.05, bad.shape)))
grid = np.zeros((7, 7))
grid[pos[:, 0], pos[:, 1]] = scores
print(pos[int(np.argmax(scores))].tolist())
print(np.round(grid, 2))
```

```
[4, 3]
[[0.01 0.01 0.01 0.01 0.02 0.02 0.03]
 [0.01 0.01 0.01 0.01 0.01 0.02 0.01]
 [0.02 0.03 0.06 0.06 0.04 0.02 0.02]
 [0.04 0.11 0.23 0.25 0.16 0.05 0.01]
 [0.07 0.18 0.38 0.42 0.27 0.08 0.01]
 [0.06 0.16 0.34 0.37 0.24 0.08 0.01]
 [0.04 0.11 0.23 0.26 0.17 0.05 0.02]]
```

The scratch covers rows 40 to 43 and columns 20 to 39 of the image. Patch (4, 3) starts at pixel
row 32 and column 24, so it contains the scratch: it has the largest distance, 0.42. Patches far
from the scratch stay near 0.01; the neighbours of the scratch are in between, since each
descriptor is averaged over its 3 × 3 neighbourhood.

## 4. A memory that never grows: reservoir mode

A robot sees images all day; the default memory keeps at most `max_size` patches, and each new
patch replaces a random stored one with probability `max_size / n_seen`, so the memory stays a
uniform sample of everything seen (reservoir sampling).

```python
import numpy as np
from dense_armor.utility.vision import PatchMemory

rng = np.random.default_rng(0)
mem = PatchMemory(max_size=1000, mode="reservoir")
for _ in range(50):
    d = rng.normal(size=(49, 11))
    mem.learn_one({"descriptors": d, "positions": None})
print(mem.n_seen, mem.bank_size)
```

```
2450 1000
```

Fifty images of 49 patches each, 2450 patches seen; the memory holds 1000.

## 5. Covering the normal patches better: coreset mode

When training images are collected first and inspection starts afterwards, the memory can keep
every training patch and, once at the end, choose the `max_size` patches that cover all of them
best.

```python
import numpy as np
from dense_armor.utility.vision import PatchMemory

rng = np.random.default_rng(0)
data = rng.normal(size=(500, 6))
data[::7] += 20.0
mem = PatchMemory(max_size=50, mode="coreset")
mem.learn_one({"descriptors": data, "positions": None})
print(mem.n_seen, mem.bank_size)
mem.finalize()
print(mem.n_seen, mem.bank_size)
```

```
500 0
500 50
```

Before `finalize()` the bank is not built yet (size 0, and `score_one` raises an error); after
it, 50 patches are kept out of 500. The choice is the greedy approximation of PatchCore eq. 5:
at each step add the patch farthest from those already chosen, so that every training patch has a
chosen one close to it:

$$\mathcal{M}_C^* = \arg\min_{\mathcal{M}_C \subset \mathcal{M}}\ \max_{m \in \mathcal{M}}\ \min_{n \in \mathcal{M}_C} \lVert m - n \rVert_2$$

On these 500 points the farthest point from its nearest kept patch is at distance 2.678 with the
coreset, 3.631 with 50 points chosen at random.

## 6. Real robot images: the RAD benchmark

RAD (Chang et al. 2024) has 13 objects photographed by a robot arm from many poses; the score is
the image-level AUROC (1 = every anomalous image ranked above every normal one, 0.5 = chance).
The script `benchmarks/vision_rad_patches.py` runs the method on every category with the same
parameters (`patch=16`, `stride=8`, `n_bins=9`, `neighbourhood=3`, `max_size=20000`).

| mode | image size | mean AUROC | p50 ms | p99 ms |
|---|---|---|---|---|
| reservoir | 160×120 | 0.593 | 73.2 | 81.2 |
| coreset | 160×120 | 0.585 | 84.3 | 127.5 |
| reservoir | 320×240 | 0.711 | 399.6 | 596.6 |
| coreset | 320×240 | 0.748 | 386.5 | 410.2 |

Reference points: global frame features of the Vision page (incremental PCA + robust
Mahalanobis), 0.497; PatchCore as reported in the RAD paper, 0.833.

Per category, coreset at 320×240:

| category | AUROC | | category | AUROC |
|---|---|---|---|---|
| binderclip | 0.658 | | phonecase | 0.782 |
| bowl | 0.702 | | rubberduck | 0.860 |
| box | 0.611 | | spoon | 0.799 |
| can | 0.851 | | spraybottle | 0.826 |
| charger | 0.593 | | tennisball | 0.975 |
| cup1 | 0.674 | | | |
| cup2 | 0.724 | | | |
| gluebottle | 0.663 | | | |

Patches beat the global features in all four runs, and the larger image helps: 0.711 and 0.748
against 0.593 and 0.585.

At 160×120 each image gives 266 patches. The 9 categories with 59 training images have 15694
patches, fewer than `max_size`, so the two modes keep the same patches and give the same AUROC on
those 9.

At 320×240 each image gives 1131 patches, so between 70 % (59 training images) and 90 % (cup2, 177
images) of the patches must be dropped. There the coreset is ahead, 0.748 against 0.711.

The latency is per image (patches + scoring) on a Colab CPU.

## API reference

::: dense_armor.utility.vision.patches

---

## Details

- Image score: $s^*$ of PatchCore eq. 6, without the reweighting $w$ the paper applies after it.
- Patch descriptor: central-difference gradients, as in `FrameFeatures`; a histogram of the
  gradient direction over `n_bins` bins in $[0, 2\pi)$, weighted by the gradient magnitude and
  L2-normalised; then the mean and the standard deviation of the patch. Averaging over the
  neighbourhood (eq. 1) uses the plain mean; the paper uses adaptive average pooling (eq. 2).
- Coreset: the first patch is the one closest to the mean, so the result is deterministic
  (Algorithm 1 of the paper starts from an empty set); the patches are used as they are,
  without the random projection $\psi$ of Algorithm 1, since they have only `n_bins + 2`
  numbers. Cost of `finalize()`: $O(n \cdot \texttt{max\_size} \cdot d)$; on RAD at 320×240 it
  took up to 143 s for the largest category (cup2), against 22 s for the reservoir.
- Nearest-neighbour distances are computed in numpy with
  $\lVert a-b\rVert^2 = \lVert a\rVert^2 + \lVert b\rVert^2 - 2\,a\cdot b$, in chunks of 512
  queries.
- Time budgets: `PatchFeatures.transform_one` measured 80.7 ms on a 320 × 240 frame
  (`budget_s = 0.15`); `PatchMemory.score_one` measured 267.8 ms (p50) against a 20000-row bank
  (`budget_s = 0.5`); both on a Colab CPU.
- Sources: Roth, K., Pemula, L., Zepeda, J., Schölkopf, B., Brox, T., Gehler, P. (2021),
  "Towards total recall in industrial anomaly detection", arXiv:2106.08265 (eq. 1, 5, 6,
  Algorithm 1, Figure 4). Chang, X., Ye, Z., et al. (2024), "RAD: A Realistic Multi-View
  Benchmark for Pose-Agnostic Anomaly Detection", arXiv:2410.00713 (image-level AUROC table,
  PatchCore 0.833 with the standard Anomalib configuration). Huang, C., Huang, J. (2017), "A fast
  HOG descriptor using lookup table and integral image", arXiv:1703.06256.
