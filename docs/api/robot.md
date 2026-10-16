# API — Robot

A map of the robot arm used in the examples, showing which parts of the library work with each joint. A joint is one moving part of the arm: a motor, a hinge or a finger.

The Dense-Armor APIs, read from the physical side. The robot is the real
Panda arm described in `test/fixtures/urdf/panda.urdf`, and every joint is
mapped to the modules that work with it.

A **joint** is one moving axis of the arm: a motor, a hinge, or a finger.
This robot has nine of them.

Two maps show the same picture. A **flat interactive map** shows where each
API touches the body: click a part and read which modules use it. A **3D
map** shows the physical volume of the robot, so the shape is recognisable
from any angle. Read them together: the flat map answers *which modules*,
the 3D map answers *where in space*.

## Interactive map

Click a body part to read what it does and which APIs concern it. The side
panel does not overlap the image.

<iframe src="../../assets/api/robot_interactive.html"
        style="width:100%;height:640px;border:0"
        loading="lazy"></iframe>

## Robot map

Hover a joint to see which modules use it. Rotate with the mouse.

<iframe src="../../assets/api/robot_3d.html"
        style="width:100%;height:640px;border:0"
        loading="lazy"></iframe>

To interact with the 3D model:

- Drag with the mouse to rotate.
- Use the arrow keys (left, right, up, down) to rotate.
- Use `+` and `-` to zoom in and out.
- Auto-rotation is on until you interact.

Ignore the colour on the 3D model and look at the silhouette: the separation
between the two arms, the curvature of the shoulders, the shape of the
pelvis. The connections that appear in the flat map as lines are, in the 3D
model, physical extensions of the body.

## APIs by joint

For each joint, the modules that work with it, grouped by purpose.

### panda_finger_joint1

**Sensors**

- [`Signal`](../api/robot_roles.md)

**Single-channel anomaly**

- [`StreamingDeviationDetector`](../anomaly/streaming.md)

**Control**

- [`KinematicController`](../control/kinematic_controller.md)

**Protection**

- [`SafeEstimator`](../api/robot_roles.md)
- [`PhysicalLimitsGuard`](../api/robot_roles.md)

### panda_finger_joint2

**Sensors**

- [`Signal`](../api/robot_roles.md)

**Single-channel anomaly**

- [`StreamingDeviationDetector`](../anomaly/streaming.md)

**Control**

- [`KinematicController`](../control/kinematic_controller.md)

**Protection**

- [`SafeEstimator`](../api/robot_roles.md)
- [`PhysicalLimitsGuard`](../api/robot_roles.md)

### panda_joint1

**Sensors**

- [`Signal`](../api/robot_roles.md)
- [`JointDerivatives`](../api/robot_roles.md)
- [`JointPower`](../api/robot_roles.md)

**Statistics**

- [`RunningMoments`](../stats/moments.md)
- [`RobustMoments`](../stats/robust.md)
- [`TDigest`](../stats/streaming.md)
- [`DDSketch`](../stats/streaming.md)

**Single-channel anomaly**

- [`ChauvenetScorer`](../anomaly/robust_filters.md)
- [`HampelScorer`](../anomaly/robust_filters.md)
- [`SigmaClipScorer`](../anomaly/robust_filters.md)
- [`StreamingDeviationDetector`](../anomaly/streaming.md)

**Drift**

- [`CUSUMDriftDetector`](../drift/cusum.md)
- [`ADWIN`](../drift/detectors.md)
- [`KSWIN`](../drift/detectors.md)
- [`PageHinkley`](../drift/detectors.md)

**Dynamics**

- [`RigidBodyModel`](../dynamics/urdf_dynamics.md)
- [`RecursiveLeastSquares`](../learn/online_dynamics.md)
- [`ResidualDynamicsLearner`](../learn/online_dynamics.md)

**Protection**

- [`SafeEstimator`](../api/robot_roles.md)
- [`PhysicalLimitsGuard`](../api/robot_roles.md)
- [`HealthMonitor`](../api/robot_roles.md)

### panda_joint2

**Sensors**

- [`JointDerivatives`](../api/robot_roles.md)
- [`JointPower`](../api/robot_roles.md)

**Statistics**

- [`RunningMoments`](../stats/moments.md)
- [`RunningCovariance`](../stats/dependence.md)

**Single-channel anomaly**

- [`ChauvenetScorer`](../anomaly/robust_filters.md)
- [`StreamingDeviationDetector`](../anomaly/streaming.md)

**Drift**

- [`CUSUMDriftDetector`](../drift/cusum.md)
- [`ADWIN`](../drift/detectors.md)

**Dynamics**

- [`RigidBodyModel`](../dynamics/urdf_dynamics.md)
- [`RecursiveLeastSquares`](../learn/online_dynamics.md)

**Control**

- [`RateLimiter`](../control/rate_limiter.md)
- [`CBFFilter`](../control/cbf_filter.md)

### panda_joint3

**Statistics**

- [`RunningMoments`](../stats/moments.md)
- [`RobustMoments`](../stats/robust.md)

**Single-channel anomaly**

- [`HampelScorer`](../anomaly/robust_filters.md)
- [`TukeyScorer`](../anomaly/robust_filters.md)

**Drift**

- [`ADWIN`](../drift/detectors.md)
- [`KSWIN`](../drift/detectors.md)

**Dynamics**

- [`ResidualDynamicsLearner`](../learn/online_dynamics.md)

**Control**

- [`PassivityCBFController`](../dynamics/passivity_cbf_controller.md)

**Protection**

- [`SafeEstimator`](../api/robot_roles.md)

### panda_joint4

**Statistics**

- [`RunningMoments`](../stats/moments.md)

**Single-channel anomaly**

- [`TukeyScorer`](../anomaly/robust_filters.md)
- [`SigmaClipScorer`](../anomaly/robust_filters.md)

**Drift**

- [`PageHinkley`](../drift/detectors.md)

**Control**

- [`CBFFilter`](../control/cbf_filter.md)
- [`KinematicController`](../control/kinematic_controller.md)

**Protection**

- [`SafeEstimator`](../api/robot_roles.md)
- [`PhysicalLimitsGuard`](../api/robot_roles.md)

### panda_joint5

**Statistics**

- [`RunningMoments`](../stats/moments.md)
- [`TDigest`](../stats/streaming.md)

**Single-channel anomaly**

- [`ChauvenetScorer`](../anomaly/robust_filters.md)
- [`HampelScorer`](../anomaly/robust_filters.md)

**Drift**

- [`CUSUMDriftDetector`](../drift/cusum.md)
- [`ADWIN`](../drift/detectors.md)
- [`PageHinkley`](../drift/detectors.md)

**Protection**

- [`HealthMonitor`](../api/robot_roles.md)
- [`StreamingHealing`](../protect/arbiter.md)

### panda_joint6

**Statistics**

- [`RunningMoments`](../stats/moments.md)

**Single-channel anomaly**

- [`ChauvenetScorer`](../anomaly/robust_filters.md)

**Multi-joint anomaly**

- [`OnlineRobustMahalanobis`](../anomaly/mahalanobis.md)
- [`HalfSpaceTrees`](../anomaly/hst.md)
- [`LODA`](../anomaly/loda.md)
- [`OneClassSGD`](../anomaly/ocsvm.md)

**Drift**

- [`ADWIN`](../drift/detectors.md)

**Protection**

- [`SafeEstimator`](../api/robot_roles.md)

### panda_joint7

**Statistics**

- [`RunningMoments`](../stats/moments.md)

**Single-channel anomaly**

- [`SigmaClipScorer`](../anomaly/robust_filters.md)

**Multi-joint anomaly**

- [`OneClassSGD`](../anomaly/ocsvm.md)
- [`LODA`](../anomaly/loda.md)
- [`OnlineIsolationForest`](../anomaly/oiforest.md)

**Drift**

- [`KSWIN`](../drift/detectors.md)

**Protection**

- [`SafeEstimator`](../api/robot_roles.md)

## Details

### How the 3D model was made

The geometry was reconstructed from a single 1024×1024 image by Meshy AI.
The mesh has roughly 59 700 vertices and 119 300 triangles, about half the
density of the head model on the AI page. The reconstruction is monochrome:
the source image was not projected onto the surface. This is intentional.
The coloured flat map is where the API hotspots live; the 3D model is where
the volume lives.

### Joint-to-API mapping

The mapping from each joint to the modules that use it is the author's
curation, not a fact extracted from the code: it says which modules talk
about that joint, not which modules run on it.
