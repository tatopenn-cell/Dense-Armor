# API — Robot

The Dense-Armor APIs, read from the physical side: the real Panda
skeleton from `test/fixtures/urdf/panda.urdf`, with every joint
mapped to the modules that work with it.

## Robot map

Hover a joint to see which modules use it. Rotate with the mouse.

<iframe src="../assets/api/robot_3d.html" style="width:100%;height:700px;border:0" loading="lazy"></iframe>

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
