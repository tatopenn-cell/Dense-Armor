# Motion: how far things moved between two frames

Comparing one frame with the previous one tells how far a pattern moved
in the time between them. On a robot this is how a moving arm knows the
world is moving past it, and how a visual servoing loop knows whether
its target has shifted.

## 1. Lucas–Kanade in a cell

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

The square moved one pixel to the right. The mean horizontal flow,
averaged over the four cells of the flow grid, is exactly 1: the library
saw one pixel of motion to the right, and nothing vertically.

Inside one cell, every pixel provides one equation `I_x u + I_y v = -I_t`,
where `I_x` and `I_y` are the spatial gradients and `I_t` is the
difference between the current and the previous frame. A single equation
has two unknowns (the aperture problem). Stacking all the equations of
the cell gives an over-determined system,

$$A\,w = b, \qquad w = (A^\top A)^{-1} A^\top b, \qquad w = (u, v)$$

with `A` the matrix of the stacked gradients, `b` the vector of the
negative temporal differences, and `w` the flow vector. This is equations
(4) and (5) of Ziani (2025). When the cell has little texture, `A^T A` is
close to singular; the library uses a least-squares solver that returns
the minimum-norm solution, which is zero flow when the cell has no
texture at all.

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

## 2. Bigger motions: the pyramid

Lucas–Kanade assumes that what moved between two frames moved by a small
amount, one or two pixels at most. The assumption is buried in the
first-order Taylor expansion of the brightness constancy equation, and it
is not optional: when the motion is bigger, the linearisation is wrong and
the least-squares lands on the wrong answer. On a robot moving at 30
frames per second this rarely matters for slow motions, but a fast arm
sweeping across the field of view, or a hand waving close to the camera,
produce jumps of ten or twenty pixels per frame. For those, the library
builds a small pyramid of images and works from the top down.

```python
import numpy as np
from dense_armor.utility.vision import FrameFeatures

a = np.zeros((64, 64), dtype=np.float32)
a[24:40, 20:36] = 1.0
b = np.roll(a, 6, axis=1)
for levels in (1, 3):
    ff = FrameFeatures(flow_cells=(1, 1), flow_levels=levels)
    ff.learn_one(a)
    print(levels, round(ff.transform_one(b)["flow_u_0_0"], 2))
```

```
1 1.0
3 6.0
```

The square jumped 6 pixels to the right. With a single level the library
sees a flow of 1.0, six times too small: the equation was linearised at
the original position and the gradient of the square's edge only supports
a small correction. With three levels the answer is 6.0, exactly right.
The pyramid did not make the code smarter; it made the motion look smaller
at first.

![Lucas–Kanade on a smooth scene: small motion, large motion with one level, large motion with three levels](../assets/vision/flow/lucas_kanade.png)

The scheme has three steps, all of them classical:

- **Build the pyramid.** Each level is the level below, averaged over 2×2
  blocks, so the image at level `l+1` has half the height and half the
  width of level `l`. The square's jump of 6 pixels at the finest level is
  a jump of 3 at the second level and a jump of 1.5 at the third; at the
  coarsest, Lucas–Kanade is back inside its comfort zone.
- **Solve at the coarsest level.** The least-squares of Section 1 is run
  on the smallest image, with zero initial flow. It gives an approximate
  motion, small in the coarse pixels but already pointing the right way.
- **Go down one level and refine.** The flow is doubled (the pixels are
  twice as big at the level above, so a motion in those pixels is a motion
  twice as large in the pixels below), the previous image is shifted by it
  and compared to the current one, and Lucas–Kanade solves for what is
  left. The refinement is added to the coarse flow, and the process
  repeats until the finest level.

The shift of the previous image is called *warping*, and it is where the
bilinear interpolation comes in. The flow is fractional, so the shifted
image is sampled at non-integer coordinates; the value at a point `(x, y)`
between pixels `(i, j)` and `(i+1, j+1)` is a weighted average of the four
surrounding pixels, with weights that depend on how far the point sits
from each. Writing `a = x - i` and `b = y - j` for the fractional parts,
the interpolation of Ziani (2025), equation 9, is

$$f(x, y) \approx (1-a)(1-b)\,f(i,j) + a(1-b)\,f(i+1,j) + (1-a)\,b\,f(i,j+1) + ab\,f(i+1,j+1)$$

If `(x, y)` lands exactly on a pixel, three of the four weights are zero
and the formula returns that pixel. If it lands in the middle of a 2×2
block, all four weights are 0.25 and the value is the average. Everything
else is between the two.

```python
import numpy as np
from dense_armor.utility.vision import FrameFeatures

a = np.zeros((64, 64), dtype=np.float32)
a[24:40, 20:36] = 1.0
b = np.roll(a, 1, axis=1)
for levels in (1, 3):
    ff = FrameFeatures(flow_cells=(1, 1), flow_levels=levels)
    ff.learn_one(a)
    print(levels, round(ff.transform_one(b)["flow_u_0_0"], 2))
```

```
1 1.0
3 1.0
```

The same experiment with a one-pixel motion: both levels give 1.0. The
pyramid does not lose precision on small motions, it only adds robustness
on large ones. On a robot, this means the same detector works whether the
arm is moving slowly or fast, and the parameter can be set once and
forgotten.

## 3. Dense flow and a mask of what moves

The cell flow of Sections 1 and 2 gives one vector per cell. `DenseFlow`
estimates a vector at every pixel, with the Horn–Schunck iteration of
Ziani (2025), equations 6-8, and `MotionMask` turns the field into a
boolean mask of the pixels that move faster than the rest.

```python
import numpy as np
from dense_armor.utility.vision.dense import DenseFlow, MotionMask

a = np.zeros((64, 64), dtype=np.float32)
a[20:40, 20:40] = 1.0
b = np.zeros((64, 64), dtype=np.float32)
b[23:43, 23:43] = 1.0
df = DenseFlow(alpha=0.1, n_iter=100, use_jax=False)
df.learn_one({"array": a})
out = df.transform_one({"array": b})
mm = MotionMask(n_sigmas=2.0, dilation=1)
mm.learn_one({"u": out["u"], "v": out["v"]})
mask = mm.transform_one({"u": out["u"], "v": out["v"]})["mask"]
print(int(mask.sum()))
```

```
1655
```

On a 20 by 20 square that moved by (3, 3) pixels, the mask lights up
1655 pixels, the moving square plus the dilation. The endpoint error of
the flow field is high on this kind of scene (a hard boundary), which
the figure shows.

![Flow vectors and the motion mask on a synthetic scene; the IoU is reported in the caption](../assets/vision/dense_flow/flow_and_mask.png)

`DenseFlow` is the only module of the page whose JAX path is on by
default (`use_jax=True`), because the Horn–Schunck iteration runs over
the whole image. The cost of a single call is in the benchmark page.

## Details

### References

- Lucas, B. D., Kanade, T. (1981). An iterative image registration
  technique with an application to stereo vision. In IJCAI 1981, 674-679.
  Section 3 for the least-squares in a cell, the aperture problem.
- Horn, B. K. P., Schunck, B. G. (1981). Determining optical flow.
  Artificial Intelligence 17(1-3), 185-203. The global smoothness method
  used by `DenseFlow`.
- Ziani, A. (2025). Optical flow: Lucas–Kanade and Horn–Schunck.
  arXiv:2511.16535. Section 3.2 for the coarse-to-fine scheme, equation 9
  for the bilinear interpolation, equations 6-8 for the Horn–Schunck
  iteration.

### Choices stated explicitly

- The library solves the least-squares per cell, not the global
  Horn–Schunck system, for the cell flow. `DenseFlow` is the module that
  runs the global iteration, and it is opt-in.
- The pyramid is built by 2×2 block averages. The paper does not
  specify a filter; a block average is the cheapest that does not lose
  energy.
- `Flow` is measured in pixels per frame, not per second: the time
  between the two frames is the caller's, and the library does not
  divide by it.

### Measured cost

`DenseFlow` at 640 by 360 is 16.8 ms per frame (median) on the
reference machine, in `benchmarks/video_quality.json` under
`DenseFlow`. `MotionMask` costs 5.4 ms. `FrameFeatures` with a small
flow grid is in the same table.

### Limits

- `DenseFlow` underestimates on low-intensity images and on hard
  boundaries (a square on a dark background). The endpoint error of the
  figure is high, and the page states it.
- `MotionMask` uses a robust z-score of the magnitude; on a scene where
  everything moves at the same speed (a camera panning in a room), the
  MAD is zero and the mask is empty. This is intended: the mask
  reports what stands out, not what moves.
- The IoU on a synthetic square is around 0.2 with the reference
  parameters, not 0.96. The 0.96 in an earlier version of the docs is
  not reproducible with the code as it is, and the page does not quote
  it.

### The numbers in this page

- `1.0` of Section 1 is the mean horizontal flow over a 2 by 2 grid of
  cells, on a 16 by 16 image with a bright square at rows 6 to 10 and
  columns 6 to 10, rolled by 1 pixel to the right.
- `1 1.0 / 3 6.0` of Section 2 is `FrameFeatures(flow_levels=lv,
  flow_cells=(1, 1))` on a 64 by 64 image with a bright square at rows
  24 to 40 and columns 20 to 36, rolled by 6 pixels to the right.
- `1 1.0 / 3 1.0` of Section 2 is the same image rolled by 1 pixel.
- `1655` of Section 3 is `MotionMask(n_sigmas=2.0).transform_one` on
  the flow of `DenseFlow(alpha=0.1, n_iter=100)` between two 64 by 64
  images with a 20 by 20 square shifted by 3 pixels in each axis.
- The figure of Section 2 is `fig_05_flow` in
  `docs/assets/vision/make_figures.py`. The figure of Section 3 is
  `fig_16_dense_flow` in the same file. The IoU in the caption of the
  second figure comes from the same script (`fig16_iou` in
  `figures.json`), not from a hand measurement.
