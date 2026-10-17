# Vision

Dense-Armor sees a video the way it reads a robot's joints: one frame
at a time, in fixed memory, with no pretrained model.

The pages, in order:

1. [Frames](frames.md): every source gives the same `Frame` with the image and the timestamp.
2. [Features](features.md): the oriented gradients, the intensity moments, the colour.
3. [Flow](flow.md): Lucas–Kanade in a cell, the pyramid, dense flow, the motion mask.
4. [Reduce](reduce.md): the random projection, the incremental PCA, the whole chain.
5. [Motion](motion.md): the motion energy, the global shift, the scene changes, the sliding window.
6. [Tracking](tracking.md): the moving object tracker, the video to signal adapters, the joints to video fusion.
7. [Kinematics](kinematics.md): landmarks, the cleaner, the joint angles, the registration.
8. [Quality](quality.md): the frame quality, the event frame, the false colour, the overlay, the benchmark.
9. [Patches](patches.md): patch memory and the RAD benchmark.

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
- `flow_levels`: the pyramid is built by 2×2 averaging; Ziani (2025,
  section 3.2) builds Gaussian pyramids and uses the coarse-to-fine scheme
  for Horn–Schunck; the library applies the same scheme to Lucas–Kanade.
  `flow_levels=1` gives exactly the single-level flow.
- `gray=False`: keys `mean_R`, `mean_G`, `mean_B`, `contrast_R`,
  `contrast_G`, `contrast_B` are always present; for a grey input they
  repeat the luminance moments.
- Real robot images, RAD (Chang et al. 2024; MIT licence): 13 objects photographed by a Franka
  arm from many viewpoints, normal and defective. `benchmarks/vision_rad.py` learns
  `FrameFeatures | IncrementalPCA(k=8) | OnlineRobustMahalanobis` on the normal training views
  and scores each test image (160×120, grey, same parameters for every category, no tuning).
  Image-level AUROC: binderclip 0.531, bowl 0.461, box 0.428, can 0.599, charger 0.508,
  cup1 0.443, cup2 0.488, gluebottle 0.492, phonecase 0.458, rubberduck 0.552, spoon 0.569,
  spraybottle 0.464, tennisball 0.473; mean **0.497**, chance level. The paper reports 0.833
  for PatchCore and 0.803 for EfficientAD (p. 10), both built on pretrained networks. Global
  features (gradients, moments) notice changes of the scene; they do not separate a small
  defect from a change of viewpoint. Each category has only 9–27 normal test images, so the
  per-category values are uncertain; the mean is the number to read. Latency 4.5 ms p50 per
  image (score only).
- Sources:
  - Chang, X. et al. (2024). RAD: A Realistic Multi-View Benchmark for
    Pose-Agnostic Anomaly Detection. arXiv:2410.00713. The robot images of the Details.
  - Huang, C., Huang, J. (2017). A fast HOG descriptor using lookup table
    and integral image. arXiv:1703.06256. The HOG construction used in
    section 2.
  - Dalal, N., Triggs, B. (2005). Histograms of oriented gradients for
    human detection. In *IEEE Conference on Computer Vision and Pattern
    Recognition*. The original HOG paper, cited in the opening.
  - Ziani, H. (2025). Investigating optical flow computation: from local
    methods to a multiresolution Horn–Schunck implementation.
    arXiv:2511.16535. The Lucas–Kanade least squares of section 4, the
    bilinear interpolation of section 4b, and the coarse-to-fine scheme.
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
