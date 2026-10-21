# Vision: the algorithms

Vision here works on frames and on tables of features. DenseFlow gives the motion of every pixel between two frames, MotionMask keeps the pixels that move most, GlobalMotion gives the shift of the whole scene, RandomProjection and PCA reduce a table of features to fewer columns.

Every class reads the whole dataset with `fit` and leaves its results in attributes with a trailing underscore; `transform` or a query method applies them. The data are `synthetic` where a block makes them up.

## 1. `DenseFlow`

Horn-Schunck optical flow between two frames.

~~~python
import numpy as np
from dense_armor.ml.vision import DenseFlow
a = np.tile(np.sin(2 * np.pi * np.arange(16) / 16), (16, 1))
m = DenseFlow(alpha=0.1, n_iter=100).fit(a, np.roll(a, 1, axis=1))
print(round(float(m.u_[8, 0]), 2), float(abs(m.v_).max()))
~~~

~~~
1.0 0.0
~~~

## 2. `GlobalMotion`

Global shift between two frames by phase correlation.

~~~python
import numpy as np
from dense_armor.ml.vision import GlobalMotion
rng = np.random.default_rng(0)
a = rng.normal(size=(32, 32))
m = GlobalMotion().fit(a, np.roll(a, (3, -2), axis=(0, 1)))
print(round(m.dx_, 3), round(m.dy_, 3))
~~~

~~~
-2.0 3.0
~~~

## 3. `MotionMask`

Mask of the pixels whose flow magnitude is large, from the two components of a flow field.

~~~python
import numpy as np
from dense_armor.ml.vision import MotionMask
u = np.tile([0.1, 0.0], (8, 4)); u[3:5, 3:5] = 2.0
print(MotionMask(dilation=0).fit(u, np.zeros((8, 8))).mask_.sum())
~~~

~~~
4
~~~

## 4. `PCA`

Principal components of a table: the top `k` eigenvectors of its covariance.

~~~python
import numpy as np
from dense_armor.ml.vision import PCA
X = [[1.0, 0.0], [-1.0, 0.0], [2.0, 0.1], [-2.0, -0.1]]
m = PCA(k=1).fit(X)
print(m.components_.round(2).tolist())
~~~

~~~
[[1.0, 0.04]]
~~~

## 5. `RandomProjection`

Gaussian random projection to `k` dimensions.

~~~python
import numpy as np
from dense_armor.ml.vision import RandomProjection
X = np.arange(12.0).reshape(2, 6)
print(RandomProjection(k=3, seed=0).fit(X).transform(X).shape)
~~~

~~~
(2, 3)
~~~

## Details

### References

Horn, B. K., Schunck, B. G. (1981). Determining optical flow. Artificial
        Intelligence 17(1-3), 185-203: the energy and its Euler-Lagrange
        equations (not in the folder, bibliographic only).
    Ziani (2025). Methods to a Multiresolution Horn-Schunck. arXiv:2511.16535,
        equations 6, 7, 8: the Jacobi iteration (not in the folder,
        bibliographic only).
    Johnson, W. B., Lindenstrauss, J. (1984) and Ghojogh et al. (2021), section
        2.2: the Gaussian matrix with variance 1/k (not in the folder,
        bibliographic only).
    Balsubramani, A., Dasgupta, S., Freund, Y. (2013). The fast convergence of
        incremental PCA, NIPS: the target of the rule is the top eigenvector of
        the covariance (not in the folder, bibliographic only).

### Choices stated explicitly

The class of a sketch is the exact value the sketch approximates, and the class of a statistic is the textbook formula on the whole array: both are `our choice`, because the references name no other batch algorithm.

### Development diary

The classes are new and independent of the online classes of dense_armor.utility.

Classes without a form:
- `TrackToSignal`: it computes one record (a frame, a box, a window of recent frames) from the state kept between calls, and the docstring writes no step that reads the whole dataset
- `FeatureToSignal`: it computes one record (a frame, a box, a window of recent frames) from the state kept between calls, and the docstring writes no step that reads the whole dataset
- `FrameFeatures`: it computes one record (a frame, a box, a window of recent frames) from the state kept between calls, and the docstring writes no step that reads the whole dataset
- `FrameJointFusion`: the docstring cites no reference and the algorithm is not a textbook definition: it is stopped and reported
- `LandmarkCleaner`: it is a causal Hampel filter with a gap rule on a landmark stream, defined on the values seen before
- `Kinematics`: it computes one record (a frame, a box, a window of recent frames) from the state kept between calls, and the docstring writes no step that reads the whole dataset
- `MotionEnergy`: it computes one record (a frame, a box, a window of recent frames) from the state kept between calls, and the docstring writes no step that reads the whole dataset
- `PatchFeatures`: it computes one record (a frame, a box, a window of recent frames) from the state kept between calls, and the docstring writes no step that reads the whole dataset
- `PatchMemory`: the reservoir mode bounds the bank on a stream with Algorithm R; the coreset reduction of PatchCore is in a paper that is not in the folder
- `FrameQuality`: the docstring cites no reference and the algorithm is not a textbook definition: it is stopped and reported
- `Sync`: the docstring cites no reference and the algorithm is not a textbook definition: it is stopped and reported
- `EventFrame`: the docstring cites no reference and the algorithm is not a textbook definition: it is stopped and reported
- `Registration`: it computes one record (a frame, a box, a window of recent frames) from the state kept between calls, and the docstring writes no step that reads the whole dataset
- `Stabilizer`: it computes one record (a frame, a box, a window of recent frames) from the state kept between calls, and the docstring writes no step that reads the whole dataset
- `DifferenceImage`: the reference is an exponential update of the frames that came before
- `FrameNormalizer`: the docstring cites no reference and the algorithm is not a textbook definition: it is stopped and reported
- `SceneChange`: it is a CUSUM or ADWIN test on a stream of frame statistics
- `FrameAnomaly`: it is a CUSUM or ADWIN test on a stream of frame statistics
- `MovingObjectTracker`: the association and the Kalman prediction use the tracks of the previous frames
- `FalseColor`: the docstring cites no reference and the algorithm is not a textbook definition: it is stopped and reported
- `Overlay`: the docstring cites no reference and the algorithm is not a textbook definition: it is stopped and reported
- `BagOfVisualWords`: it computes one record (a frame, a box, a window of recent frames) from the state kept between calls, and the docstring writes no step that reads the whole dataset
- `FrameWindow`: it computes one record (a frame, a box, a window of recent frames) from the state kept between calls, and the docstring writes no step that reads the whole dataset
