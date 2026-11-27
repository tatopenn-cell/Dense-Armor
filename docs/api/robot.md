# API — Robot

A map of the robot arm used in the examples, showing which parts of the library work with each joint. A joint is one moving part of the arm: a motor, a hinge or a finger.

The Dense-Armor APIs, read from the physical side. The robot is the real
Panda arm described in `test/fixtures/urdf/panda.urdf`, and every joint is
mapped to the modules that work with it.

A **joint** is one moving axis of the arm: a motor, a hinge, or a finger.
This robot has nine of them.

## The map of the robot pages

The pages of the Robot group, with the modules of each. Pass over a page to light it, click to open it.

<object type="image/svg+xml" data="../../assets/maps/robot.svg" style="width:100%"></object>

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

- [`RunningMoments`](../stats/moments/index.md)
- [`RobustMoments`](../stats/robust.md)
- [`TDigest`](../stats/streaming.md)
- [`DDSketch`](../stats/streaming.md)

**Single-channel anomaly**

- [`ChauvenetScorer`](../anomaly/robust_filters.md)
- [`HampelScorer`](../anomaly/robust_filters.md)
- [`SigmaClipScorer`](../anomaly/robust_filters.md)
- [`StreamingDeviationDetector`](../anomaly/streaming.md)

**Drift**

- [`CUSUMDriftDetector`](../drift/cusum/index.md)
- [`ADWIN`](../drift/detectors/index.md)
- [`KSWIN`](../drift/detectors/index.md)
- [`PageHinkley`](../drift/detectors/index.md)

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

- [`RunningMoments`](../stats/moments/index.md)
- [`RunningCovariance`](../stats/dependence.md)

**Single-channel anomaly**

- [`ChauvenetScorer`](../anomaly/robust_filters.md)
- [`StreamingDeviationDetector`](../anomaly/streaming.md)

**Drift**

- [`CUSUMDriftDetector`](../drift/cusum/index.md)
- [`ADWIN`](../drift/detectors/index.md)

**Dynamics**

- [`RigidBodyModel`](../dynamics/urdf_dynamics.md)
- [`RecursiveLeastSquares`](../learn/online_dynamics.md)

**Control**

- [`RateLimiter`](../control/rate_limiter.md)
- [`CBFFilter`](../control/cbf_filter.md)

### panda_joint3

**Statistics**

- [`RunningMoments`](../stats/moments/index.md)
- [`RobustMoments`](../stats/robust.md)

**Single-channel anomaly**

- [`HampelScorer`](../anomaly/robust_filters.md)
- [`TukeyScorer`](../anomaly/robust_filters.md)

**Drift**

- [`ADWIN`](../drift/detectors/index.md)
- [`KSWIN`](../drift/detectors/index.md)

**Dynamics**

- [`ResidualDynamicsLearner`](../learn/online_dynamics.md)

**Control**

- [`PassivityCBFController`](../dynamics/passivity_cbf_controller.md)

**Protection**

- [`SafeEstimator`](../api/robot_roles.md)

### panda_joint4

**Statistics**

- [`RunningMoments`](../stats/moments/index.md)

**Single-channel anomaly**

- [`TukeyScorer`](../anomaly/robust_filters.md)
- [`SigmaClipScorer`](../anomaly/robust_filters.md)

**Drift**

- [`PageHinkley`](../drift/detectors/index.md)

**Control**

- [`CBFFilter`](../control/cbf_filter.md)
- [`KinematicController`](../control/kinematic_controller.md)

**Protection**

- [`SafeEstimator`](../api/robot_roles.md)
- [`PhysicalLimitsGuard`](../api/robot_roles.md)

### panda_joint5

**Statistics**

- [`RunningMoments`](../stats/moments/index.md)
- [`TDigest`](../stats/streaming.md)

**Single-channel anomaly**

- [`ChauvenetScorer`](../anomaly/robust_filters.md)
- [`HampelScorer`](../anomaly/robust_filters.md)

**Drift**

- [`CUSUMDriftDetector`](../drift/cusum/index.md)
- [`ADWIN`](../drift/detectors/index.md)
- [`PageHinkley`](../drift/detectors/index.md)

**Protection**

- [`HealthMonitor`](../api/robot_roles.md)
- [`StreamingHealing`](../protect/arbiter.md)

### panda_joint6

**Statistics**

- [`RunningMoments`](../stats/moments/index.md)

**Single-channel anomaly**

- [`ChauvenetScorer`](../anomaly/robust_filters.md)

**Multi-joint anomaly**

- [`OnlineRobustMahalanobis`](../anomaly/mahalanobis.md)
- [`HalfSpaceTrees`](../anomaly/hst.md)
- [`LODA`](../anomaly/loda.md)
- [`OneClassSGD`](../anomaly/ocsvm.md)

**Drift**

- [`ADWIN`](../drift/detectors/index.md)

**Protection**

- [`SafeEstimator`](../api/robot_roles.md)

### panda_joint7

**Statistics**

- [`RunningMoments`](../stats/moments/index.md)

**Single-channel anomaly**

- [`SigmaClipScorer`](../anomaly/robust_filters.md)

**Multi-joint anomaly**

- [`OneClassSGD`](../anomaly/ocsvm.md)
- [`LODA`](../anomaly/loda.md)
- [`OnlineIsolationForest`](../anomaly/oiforest.md)

**Drift**

- [`KSWIN`](../drift/detectors/index.md)

**Protection**

- [`SafeEstimator`](../api/robot_roles.md)

## Details

### Joint-to-API mapping

The mapping from each joint to the modules that use it is the author's
curation, not a fact extracted from the code: it says which modules talk
about that joint, not which modules run on it.
