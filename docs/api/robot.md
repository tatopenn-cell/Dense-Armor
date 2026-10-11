# API — Robot

The Dense-Armor APIs, read from the physical side: the real Panda
skeleton from `test/fixtures/urdf/panda.urdf`, with every joint
mapped to the modules that work with it.

## Interactive map

Clicca una parte del corpo per leggere cosa fa e quali API la riguardano. Il pannello a destra non copre l'immagine.

<iframe src="../assets/api/robot_interactive.html" style="width:100%;height:640px;border:0" loading="lazy"></iframe>


## Robot map

Hover a joint to see which modules use it. Rotate with the mouse.

<iframe src="../assets/api/robot_3d.html" style="width:100%;height:640px;border:0" loading="lazy"></iframe>


**What this model shows.** The solid below is a reconstruction of the same
concept robot that opens this page, generated from the 1024x1024 source image.
Every pixel of the source has been lifted into a height and the surface has
been closed into a volume. Unlike the flat coloured map at the top, this
model can be rotated and inspected from the back: the two arms, the
shoulders, and the ground rays are all visible as three-dimensional shapes.

**How it was made.** The geometry was reconstructed from a single image by
Meshy AI. The mesh has roughly 59 700 vertices and 119 300 triangles, about
half the density of the head model. The reconstruction is monochrome: the
source image was not projected onto the surface. This is intentional. The
coloured map at the top of this page is where the API hotspots live; the
3D model is where the volume lives.

**How to read it.** Ignore the colour. Look at the silhouette: the separation
between the two arms, the curvature of the shoulders, the shape of the
pelvis, the length of the rays that connect the body to the ground. The
connections that appear in the flat map as lines are, in this model,
physical extensions of the body.

**How to interact.**
- Drag with the mouse to rotate.
- Use the arrow keys (left, right, up, down) to rotate.
- Use + and - to zoom in and out.
- Auto-rotation is on until you interact.

*The joint-to-API mapping is the author's curation, not a fact
extracted from the code: it says which modules use or talk about
that joint.*

## APIs by joint

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
