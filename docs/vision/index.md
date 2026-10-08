# Vision

A robot with a camera sees thousands of numbers per frame: one per pixel,
and with three colour channels, three per pixel. A modern camera at 160 by
120 pixels delivers nineteen thousand two hundred numbers, thirty times a
second. Most of them are noise for the task at hand. What matters is how
bright the scene is, where the edges point, how the light distribution sits
in the frame, and how things moved since the previous frame. This page shows
how Dense-Armor turns a stream of frames into a few dozen numbers that
describe all of that, and how it learns from them one frame at a time,
inside the library: no pretrained network, no external model, no files to
download.

The interface is the same as everywhere else in the library. A source of
frames is a stream of samples, each with a timestamp. A transformer reads
one sample and returns a smaller dict of numbers. The transformers compose
with `|`, and every step of the chain learns as the frames go by. The only
optional dependency is the camera driver itself, and only for real cameras.

## The ideas, and where they come from

Every technique on this page has been on someone's desk for decades, and
each one answers a question that is easy to ask and surprisingly hard to
answer well.

The **histogram of oriented gradients** was invented in 2005 by Dalal and
Triggs, who wanted to find people walking in photographs. Their insight was
that a person is recognisable not by the brightness of their clothes but by
the directions of the edges on their body. It remains a standard way to
describe an image cheaply.

The **optical flow** came from Lucas and Kanade in 1981. They wanted to know
how much a patch of pixels moved from one frame to the next, and they
realised that a single pixel is not enough to tell direction, but a small
neighbourhood is. The global alternative,
Horn and Schunck in the same year, adds a smoothness term over the whole
image and solves one large system; Dense-Armor uses Lucas–Kanade because it
is local and its cost per cell does not grow with the frame.

The **Johnson and Lindenstrauss lemma** is from 1984 and it is a statement
about a question a pure mathematician would ask: how many dimensions do you
really need to keep the distances between a set of points? The answer is
surprisingly few, and a random projection is enough to keep them. This page
uses that.

**Principal component analysis** is even older, from Pearson in 1901, and
the incremental version used here is a form of Oja's rule from 1982 that
learns the answer as data streams in. Krasulina, in 1969, arrived at almost
the same rule from a different direction, and the modern analysis in
Balsubramani, Dasgupta and Freund (2013) proves how quickly it converges.
What all these ideas share is a willingness to throw information away: a
frame becomes a hundred numbers, a hundred numbers become four, four become
one anomaly score. The trick is doing it without losing what matters, and
four decades of mathematics went into making that possible.

## 1. Frames with a timestamp

Every source gives the same object, a `Frame` with the image in `.array`
(values between 0 and 1) and the time in seconds in `.t`.

```python
import numpy as np
from dense_armor.utility.vision import ArrayStream

frames = np.zeros((5, 120, 160), dtype=np.float32)
s = ArrayStream(frames, fps=30.0)
for f in s:
    print(f.array.shape, round(f.t, 4))
```

```
(120, 160) 0.0
(120, 160) 0.0333
(120, 160) 0.0667
(120, 160) 0.1
(120, 160) 0.1333
```

Five frames, 120 rows by 160 columns, one every 1/30 of a second: the
timestamps grow by 0.0333 s at a time. The image never leaves the array,
it is just a numpy view with a timestamp attached.

The same object comes from three sources. `ArrayStream` reads frames
already in memory, which is what tests use. `CameraStream(device=0, fps=30,
size=(160, 120))` reads a real camera through OpenCV; OpenCV is installed
with `pip install dense-armor[vision]`, and if it is missing the constructor
raises a clear `ImportError` with that exact message. A real camera
delivers frames in OpenCV's native BGR order, which is turned into RGB
before the grey conversion, so a red square stays a red square in the
features. `ImageFolderStream(path, fps=...)` reads a folder of images in
name order, using Pillow, which is already a dependency of matplotlib;
if no image file is found the constructor raises a `ValueError`.

The `Frame` object itself is deliberately thin: a numpy array and a
timestamp, nothing more. Every source — a USB camera, a folder of PNGs, a
numpy array built on the fly — produces frames in different ways and at
different rates, and the rest of the library should not care which one it
is reading from. The stream is the adapter, the frame is the common
currency. When a robot gets a new camera, only the stream changes.

## 2. Edges and their direction

The gradient of an image says how fast the brightness changes and in which
direction. The image is cut into cells, and each cell counts how much edge
it has in each direction, one histogram per cell. This is the idea behind
the histogram of oriented gradients.

```python
import numpy as np
from dense_armor.utility.vision import FrameFeatures

img = np.zeros((16, 16), dtype=np.float32)
img[8:, :] = 1.0
ff = FrameFeatures(n_bins=4, cells=(1, 1))
out = ff.transform_one(img)
print([round(out[f"hog_0_0_{j}"], 3) for j in range(4)])
```

```
[0.0, 1.0, 0.0, 0.0]
```

The image is dark on top and bright at the bottom, so the edge between the
two halves is horizontal. All the gradient mass lands in one direction bin;
the other three bins are zero. The library saw an edge of a certain
direction, and nothing else.

The gradient at each pixel is a central difference of the neighbouring
brightness values,

$$I_x = \tfrac12\,(I_{y,x+1} - I_{y,x-1}), \quad I_y = \tfrac12\,(I_{y+1,x} - I_{y-1,x}), \quad
m = \sqrt{I_x^2 + I_y^2}, \quad \theta = \operatorname{atan2}(I_y, I_x)$$

Every pixel carries a magnitude `m` (how strong the edge is) and an angle
`θ` (which way it points). Inside a cell, the pixel votes into the bin of
its angle, with a weight equal to `m`. The cell histogram is then divided
by its own length, so a brighter scene gives the same numbers as a dimmer
one with the same shapes. The construction and the choice of nine bins per
cell are the ones described in Huang and Huang (2017).

Why does this work? Because edges are robust in a way that brightness is
not. A person walking past a camera changes the brightness of a thousand
pixels slightly, because the light in the room shifted, but the direction
of every edge on the person stays the same. The oriented histogram sees
the edges and ignores the brightness. A descriptor that survives a change
in the room lights is what a robot wants.

## 3. Where the light is

Three global numbers summarise the intensity of the frame: the mean, the
contrast, and the centre of mass of the brightness.

```python
import numpy as np
from dense_armor.utility.vision import FrameFeatures

img = np.zeros((20, 20), dtype=np.float32)
img[2:6, 14:18] = 1.0
out = FrameFeatures().transform_one(img)
print(round(out["mean"], 3), round(out["centroid_x"], 3), round(out["centroid_y"], 3))
```

```
0.04 0.816 0.184
```

A small bright square near the top right. The mean is low because most of
the frame is black. The centroid sits at `x = 0.816` (towards the right)
and `y = 0.184` (towards the top), where the light is. A zero on either
axis means the left or the top edge, a one means the right or the bottom.

Three numbers, and between them they tell a surprisingly large part of
the story. A bright object entering from the left moves the centroid to
the left; a shadow passing over the scene lowers the mean; a person walking
close to the camera raises the contrast. None of these three numbers is
interesting on its own, but together they catch a whole family of changes
that the edges alone would miss — a light switched off, a lens partially
covered, a foggy window. All three are computed with a single pass over
the frame and no stored state.

## 4. What moved: optical flow

Comparing a frame with the previous one says how far things moved in the
time between the two frames. The library solves a small least-squares
problem in every cell of a coarse grid; this is the Lucas–Kanade method.

```python
import numpy as np
from dense_armor.utility.vision import FrameFeatures

a = np.zeros((16, 16), dtype=np.float32)
a[6:10, 6:10] = 1.0
b = np.roll(a, 1, axis=1)
ff = FrameFeatures(flow_cells=(2, 2))
ff.learn_one(a)
out = ff.transform_one(b)
print(round(float(np.mean([out[f"flow_u_{i}_{j}"] for i in range(2) for j in range(2)])), 3))
```

```
1.0
```

The square moved one pixel to the right. The mean horizontal flow, averaged
over the four cells of the flow grid, is exactly 1: the library saw one
pixel of motion to the right, and nothing vertically.

Inside one cell, every pixel provides one equation `I_x u + I_y v = -I_t`,
where `I_x` and `I_y` are the spatial gradients and `I_t` is the difference
between the current and the previous frame. A single equation has two
unknowns (the aperture problem). Stacking all the equations of the cell gives an over-determined
system,

$$A\,w = b, \qquad w = (A^\top A)^{-1} A^\top b, \qquad w = (u, v)$$

with `A` the matrix of the stacked gradients, `b` the vector of the negative
temporal differences, and `w` the flow vector. This is equations (4) and
(5) of that paper. When the cell has little texture, `A^T A` is close to
singular; the library uses a least-squares solver that returns the
minimum-norm solution, which is zero flow when the cell has no texture at
all.

The aperture problem is the reason a single pixel is not enough. Staring
at one pixel through a small hole, you cannot tell whether the pattern
moved left or up and to the side; the equation constrains only the
component of motion perpendicular to the edge. Lucas and Kanade solved
this in 1981 by looking at a whole neighbourhood at once, and that is what
the code above does: every cell stacks its equations and solves them
together. The global alternative from the same year, Horn and Schunck,
adds a smoothness term over the whole image and solves one large system;
Dense-Armor stays local because a robot needs the answer in a few
milliseconds, and a small least-squares per cell delivers that.

## 5. Fewer numbers, same distances: random projection

Multiplying by a fixed random matrix shrinks a vector from 256 numbers to
64 while keeping the distances between frames almost the same. This is the
Johnson–Lindenstrauss lemma.

```python
import numpy as np
from dense_armor.utility.vision import RandomProjection

rng = np.random.default_rng(0)
d, n, k = 256, 100, 64
X = rng.standard_normal((n, d))
rp = RandomProjection(k=k, seed=0)
keys = [f"f{i:03d}" for i in range(d)]
Y = np.array([list(rp.transform_one(dict(zip(keys, row))).values()) for row in X])
r = [np.sum((Y[i] - Y[j]) ** 2) / np.sum((X[i] - X[j]) ** 2) for i in range(n) for j in range(i + 1, n)]
print(round(float(np.median(r)), 3), round(float(np.mean((np.array(r) < 0.5) | (np.array(r) > 1.5))), 4))
```

```
0.975 0.0024
```

Two hundred fifty-six numbers per frame become sixty-four, a factor of
four fewer. The median ratio of the squared distances between two frames
before and after the projection is 0.975, close to 1: a pair of frames that
was far apart in the original space stays far apart, and a pair that was
close stays close. The fraction of pairs off by more than fifty per cent
is 0.0024, well below the theoretical limit.

The projection matrix is a dense Gaussian matrix with variance `1 / k`, the
one used in the proof of the lemma. The bound for a single pair of points
is equation (21) of Ghojogh et al. (2021):

$$P\big((1-\epsilon)\|x_i - x_j\|^2 \le \|f(x_i) - f(x_j)\|^2 \le (1+\epsilon)\|x_i - x_j\|^2\big)
\ge 1 - \delta, \qquad \delta = 2\,e^{-(\epsilon^2 - \epsilon^3)\,k/4}$$

With `k = 64` and `ε = 0.5`, a single pair of points is off by more than
fifty per cent with probability at most `δ = 0.27` in the worst case. The
experiment above is on `n = 100` independent points and the fraction of bad
pairs is well below that, because `δ` is a worst-case bound, not a tight
estimate.

The lemma is counter-intuitive on purpose. It says that if all you care
about is the distances between points, you can throw away most of the
coordinates, as long as you throw them away in a random direction. The
proof in Ghojogh et al. (2021) follows the classical steps: one pair of
points first, then a union bound over all the pairs.
The consequence for the library is practical: a single matrix drawn once
with a seed keeps the geometry of the data good enough, and never needs to
be retrained.

## 6. Learning the main directions: incremental PCA

Principal component analysis finds the directions along which the data
varies most. The incremental version learns them one sample at a time,
without ever storing the whole dataset.

```python
import numpy as np
from dense_armor.utility.vision import IncrementalPCA

rng = np.random.default_rng(0)
pca = IncrementalPCA(k=1)
for _ in range(400):
    pca.learn_one({"a": float(rng.normal(0, 0.1)), "b": float(rng.normal(0, 1))})
v = pca.components_[:, 0]
print(round(float(np.degrees(np.arccos(abs(v[1]) / np.linalg.norm(v)))), 2))
```

```
1.08
```

The stream has two channels: `b` varies ten times more than `a`. After 400
samples the learned direction is within 1.08 degrees of `b`, the true top
direction. The library found the right axis without ever keeping more than
two floats of state for the component.

The update is a stochastic power iteration with orthonormalisation, the
same rule as the one-component Oja update in Balsubramani, Dasgupta,
Freund (2013), equation (1):

$$v_n = v_{n-1} + \gamma_n\big(X_n X_n^\top - (v_{n-1}^\top X_n X_n^\top v_{n-1})\, I\big)\, v_{n-1},
\qquad \gamma_n = c / n$$

The running mean is removed from each sample before the update, and the
component matrix is re-orthonormalised after every sample, so the `k`
components stay orthogonal to each other. The paper analyses the rate of
convergence: the potential `1 - (v · v*)^2 / |v|^2` goes to zero as
`O(1 / n)` when the constant `c` is at least
`1 / (2 (λ_1 - λ_2))`, where `λ_1` and `λ_2` are the two top eigenvalues
of the covariance.

The word "incremental" is the whole point. Classical PCA, from Pearson in
1901, needs the whole dataset in memory and does an eigen-decomposition of
the covariance matrix; for the 256 features of this page that would be a
256-by-256 matrix filled by a hundred frames. The incremental version keeps
a fixed `d × k` matrix and updates it in place. It has seen a hundred
thousand frames, or a million, and its state is still the same size: a
matrix and a mean. On a robot, that is the difference between a component
that runs for a shift and a component that has to be restarted every hour.

## 7. The whole chain: a camera that notices a change

Three steps joined with `|`. Every step learns as the frames go by; the
anomaly score is read before the frame is fed to the chain.

```python
import numpy as np
from dense_armor.utility.anomaly.mahalanobis import OnlineRobustMahalanobis
from dense_armor.utility.vision import ArrayStream, FrameFeatures, IncrementalPCA

frames = []
for i in range(100):
    f = np.zeros((32, 32), dtype=np.float32)
    if i < 80:
        f[8:16, min(24, 4 + i // 4):min(24, 4 + i // 4) + 8] = 1.0
    else:
        f[8:20, 8:20] = 1.0
        f[24:30, 24:30] = 0.8
    frames.append(f)
pipe = FrameFeatures() | IncrementalPCA(k=4) | OnlineRobustMahalanobis(feature_keys=[f"pc{j}" for j in range(4)])
scores = []
for fr in ArrayStream(np.stack(frames), fps=10.0):
    scores.append(pipe.steps[-1][1].score_one(pipe.steps[1][1].transform_one(pipe.steps[0][1].transform_one(fr))))
    pipe.learn_one(fr, t=fr.t)
print(round(float(np.mean(scores[20:80])), 3), round(float(np.mean(scores[80:])), 3))
```

```
2.998 4.439
```

For the first 80 frames a small bright square slides across the scene. From
frame 80 the scene changes: the square grows and a second object appears.
The mean anomaly score rises from 2.998 to 4.439. The library has learned
what "normal" looks like from the streaming frames alone, without a
pretrained model and without storing a single frame after processing it.

The way to read this chain is the same way as the rest of the library.
Each `|` passes the output of one step to the input of the next. The first
step turns a frame into a hundred and thirty numbers. The second step
compresses those into four. The third step measures how far the four
numbers sit from the running centre of mass of the previous ones; a large
distance means the frame does not look like the ones the library has seen,
and the robot has a signal to react to. The third step is the online
robust Mahalanobis distance from Guillot et al. (2026), which uses the
geometric median and the median covariation matrix of the projected
features; Cardot, Cénac and Zitt (2013) and Cardot and Godichon-Baggioni
(2017) prove the averaged stochastic gradient updates that keep both up
to date.

Nothing in the chain was trained in advance. No labelled data, no
pretrained weights, no external service was called. The library saw a
square move for eighty frames and then noticed, on its own, that the
scene had changed.

## 8. Fast enough for a control loop

Measured on the development machine, 160×120 frames, median and 99th
percentile over 95 frames:

| step | p50 (ms) | p99 (ms) |
|---|---|---|
| `FrameFeatures` | 1.76 | 2.05 |
| `FrameFeatures` + `IncrementalPCA(k=8)` + Mahalanobis | 2.05 | 3.02 |

A camera at 30 frames per second leaves 33 milliseconds per frame. Even at
the 99th percentile, the full chain is well inside that budget: there is
room left for the rest of the control loop.

The reason the numbers are this small is that everything runs in numpy on
the CPU. There is no GPU, no separate process, no network call to a
pretrained model. On the development machine the whole chain uses less
than a tenth of the 33 ms frame budget at 30 frames per second.

## API reference

::: dense_armor.utility.vision.streams

::: dense_armor.utility.vision.features

::: dense_armor.utility.vision.reduce

---

## Details

- Everything on this page is computed by the library in numpy; the camera
  driver (OpenCV) is the only optional dependency, extra `[vision]`,
  imported only by `CameraStream`.
- `FrameFeatures` keys: `hog_{cy}_{cx}_{b}` (default 3×4 cells, 9 bins),
  `mean`, `contrast`, `centroid_x`, `centroid_y`, `flow_u_*`, `flow_v_*`,
  `flow_mag_*` (default 2×3 flow cells). The first frame has zero flow; an
  empty or non-finite frame is skipped and counted in `n_missing`.
- `RandomProjection` uses a dense Gaussian matrix with variance `1 / k`
  (the matrix of equation 21). Sparse projections (Achlioptas 2003, Li et
  al. 2006) are named in the same survey but not defined there; they are
  not implemented yet.
- `IncrementalPCA` with several components updates all of them with the
  same step and re-orthonormalises with a QR decomposition after each
  sample (a stochastic subspace iteration); the paper analyses the
  one-component rule.
- Pipelines built with `|` train every step: each intermediate step first
  transforms the sample, then learns from it, so `FrameFeatures` keeps the
  previous frame for the flow.
- Sources:
  - Huang, C., Huang, J. (2017). A fast HOG descriptor using lookup table
    and integral image. arXiv:1703.06256. The HOG construction used in
    section 2.
  - Dalal, N., Triggs, B. (2005). Histograms of oriented gradients for
    human detection. In *IEEE Conference on Computer Vision and Pattern
    Recognition*. The original HOG paper, cited in the opening.
  - Ziani, H. (2025). Investigating optical flow computation: from local
    methods to a multiresolution Horn–Schunck implementation.
    arXiv:2511.16535. The Lucas–Kanade least squares of section 4.
  - Lucas, B. D., Kanade, T. (1981). An iterative image registration
    technique with an application to stereo vision. In *Proceedings of the
    7th International Joint Conference on Artificial Intelligence*.
    The original Lucas–Kanade algorithm.
  - Horn, B. K., Schunck, B. G. (1981). Determining optical flow.
    *Artificial Intelligence* 17(1-3), 185-203. The global method
    mentioned at the end of section 4.
  - Johnson, W. B., Lindenstrauss, J. (1984). Extensions of Lipschitz
    mappings into a Hilbert space. *Contemporary Mathematics* 26. The
    distance-preservation lemma of section 5.
  - Achlioptas, D. (2003). Database-friendly random projections:
    Johnson–Lindenstrauss with binary coins. *Journal of Computer and
    System Sciences* 66(4), 671-687. The sparse projection matrix named
    in the Details.
  - Li, P., Hastie, T. J., Church, K. W. (2006). Very sparse random
    projections. In *Proceedings of the 12th ACM SIGKDD International
    Conference on Knowledge Discovery and Data Mining*. The sparser
    alternative named in the Details.
  - Ghojogh, B., Ghodsi, A., Karray, F., Crowley, M. (2021).
    Johnson–Lindenstrauss lemma, linear and nonlinear random projections,
    random Fourier features, and random kitchen sinks: tutorial and
    survey. arXiv:2108.04172. The bound of section 5, equation 21.
  - Pearson, K. (1901). On lines and planes of closest fit to systems of
    points in space. *Philosophical Magazine* 2(11), 559-572. The origin
    of principal component analysis.
  - Krasulina, T. P. (1969). A method of stochastic approximation for the
    determination of the least eigenvalue of a symmetrical matrix. *USSR
    Computational Mathematics and Mathematical Physics* 9(6), 189-195.
    One of the two classical one-component rules.
  - Oja, E. (1982). Simplified neuron model as a principal component
    analyzer. *Journal of Mathematical Biology* 15(3), 267-273. The
    other classical one-component rule.
  - Balsubramani, A., Dasgupta, S., Freund, Y. (2013). The fast
    convergence of incremental PCA. In *Advances in Neural Information
    Processing Systems* (NIPS), arXiv:1501.03796. The convergence rate
    of section 6, equation 1.
  - Guillot, A. et al. (2026). Online robust covariance and outlier
    detection. arXiv:2601.03957. The robust Mahalanobis detector used in
    the last step of section 7.
  - Cardot, H., Cénac, P., Zitt, P.-A. (2013). Efficient and fast
    estimation of the geometric median in Hilbert spaces with an averaged
    stochastic gradient algorithm. *Bernoulli* 19(1), 18-43. The
    geometric median update of section 7.
  - Cardot, H., Godichon-Baggioni, A. (2017). Fast estimation of the
    median covariation matrix with application to online robust principal
    components analysis. *TEST* 26(3), 461-480. The median covariation
    matrix update of section 7.
