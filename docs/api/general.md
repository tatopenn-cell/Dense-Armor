# API

Complete map of every public class in Dense-Armor: description,
base classes, public methods. When a class has a guide page,
the link leads there for the full explanation and examples.

**164 classes**, 1530 public methods, 111 public modules.

## Classes by subsystem

**Jump to:** [Roles](#roles) &nbsp;·&nbsp; [Metrics](#metrics) &nbsp;·&nbsp; [Core](#core) &nbsp;·&nbsp; [Mcp_server](#mcp_server) &nbsp;·&nbsp; [Vision](#vision) &nbsp;·&nbsp; [Datasets](#datasets) &nbsp;·&nbsp; [Protect](#protect) &nbsp;·&nbsp; [Misc](#misc) &nbsp;·&nbsp; [Learn](#learn) &nbsp;·&nbsp; [Anomaly](#anomaly) &nbsp;·&nbsp; [Armatura](#armatura) &nbsp;·&nbsp; [Dynamics](#dynamics)

---

## Roles

### ADWIN


*Module:* `dense_armor.utility.drift.adwin`
*Bases:* DriftDetector
*Guide:* [drift/detectors.md](../drift/detectors.md)

Adaptive Windowing drift detector.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `update(x: float, t=None)`
- `update_many(X, t=None)`
- `window_samples(seconds: float)`

### AdaptiveConformalRegressor


*Module:* `dense_armor.roles.uncertainty`
*Bases:* Regressor

Conformal interval with adaptive coverage, wrapping a regressor.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: signal.Signal | dict, y: Any, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `predict_interval(x: signal.Signal | dict, t=None)`
- `predict_many(X, t=None, return_std=False)`
- `predict_one(x: signal.Signal | dict, t=None, return_std=False, return_estimate=False)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_one(x: signal.Signal | dict, t=None)`
- `window_samples(seconds: float)`

### AnomalyDetector


*Module:* `dense_armor.roles.anomaly_detector`
*Bases:* Estimator

Unsupervised anomaly detector; ``score_one`` high = anomalous.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: dict, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `score_many(X, t=None)`
- `score_one(x: dict, t=None)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### AnomalyGate


*Module:* `dense_armor.roles.anomaly_detector`
*Bases:* Estimator

Detector + classification, with optional protection of the detector.

Public methods:

- `classify(score: float)`
- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_one(x: dict, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `score_one(x: dict, t=None)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### Autocorrelation


*Module:* `dense_armor.utility.stats.dependence`
*Bases:* Transformer

Running autocorrelation of a single channel at a fixed lag.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: dict, y=None, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: dict, t=None)`
- `update(x: dict, y=None, t=None)`
- `window_samples(seconds: float)`

### BagOfVisualWords


*Module:* `dense_armor.utility.vision.vocabulary`
*Bases:* Transformer
*Guide:* [cluster/index.md](../cluster/index.md)

Visual vocabulary over image patches and per-frame histogram.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: Any, y=None, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: Any, t=None)`
- `window_samples(seconds: float)`

### BagOfWords


*Module:* `dense_armor.utility.preprocessing.text`
*Bases:* Transformer
*Guide:* [preprocessing/index.md](../preprocessing/index.md)

Term counts of one document, from a token list.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: Any, y=None, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: Any, t=None)`
- `window_samples(seconds: float)`

### BloomFilter


*Module:* `dense_armor.utility.stats.sketches`
*Bases:* Transformer

Bloom membership filter (Bloom 1970).

Public methods:

- `check(item: Any)`
- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: dict, y=None, t=None)`
- `load_state_dict(state: dict)`
- `merge(other: BloomFilter)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: dict, t=None)`
- `update(x: dict, y=None, t=None)`
- `window_samples(seconds: float)`

### CUSUMDriftDetector


*Module:* `dense_armor.utility.drift.detector`
*Bases:* DriftDetector
*Guide:* [drift/detectors.md](../drift/detectors.md)

CUSUM drift detector matching the batch `cusum_detector` behaviour.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `expected_detection_delay(shift: float)`
- `expected_false_alarm_run()`
- `get_params(deep=True)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `update(x: float, t=None)`
- `update_many(X, t=None)`
- `window_samples(seconds: float)`

### ChauvenetScorer


*Module:* `dense_armor.utility.anomaly.filters`
*Bases:* AnomalyDetector
*Guide:* [anomaly/streaming.md](../anomaly/streaming.md)

Chauvenet outlier scorer on the causal window.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `is_outlier(x)`
- `learn_many(X, t=None)`
- `learn_one(x)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `score_many(X, t=None)`
- `score_one(x)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### Classifier


*Module:* `dense_armor.roles.classifier`
*Bases:* Estimator

Supervised classifier.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: dict, y: Any, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `predict_many(X, t=None)`
- `predict_one(x: dict, t=None)`
- `predict_proba_many(X, t=None)`
- `predict_proba_one(x: dict, t=None)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### CountMinSketch


*Module:* `dense_armor.utility.stats.sketches`
*Bases:* Transformer

Count-Min frequency sketch (Cormode & Muthukrishnan 2005).

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `estimate(item: Any)`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: dict, y=None, t=None)`
- `load_state_dict(state: dict)`
- `merge(other: CountMinSketch)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: dict, t=None)`
- `update(x: dict, y=None, t=None)`
- `window_samples(seconds: float)`

### DDSketch


*Module:* `dense_armor.utility.stats.quantiles`
*Bases:* Transformer

Relative-error quantile sketch (Masson, Rim, Lee 2019).

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: dict, y=None, t=None)`
- `load_state_dict(state: dict)`
- `merge(other: DDSketch)`
- `mutate(new_attrs: dict)`
- `quantile(q: float)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: dict, t=None)`
- `update(x: dict, y=None, t=None)`
- `window_samples(seconds: float)`

### DenStream


*Module:* `dense_armor.utility.cluster.denstream`
*Bases:* Transformer
*Guide:* [cluster/index.md](../cluster/index.md)

Density-based stream clustering with a damped window.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: Any, y=None, t=None)`
- `load_state_dict(state: dict)`
- `macro_clusters()`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: Any, t=None)`
- `window_samples(seconds: float)`

### Discard


*Module:* `dense_armor.utility.compose.select`
*Bases:* Transformer

Drop the named channels.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: Any, y=None, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: Any, t=None)`
- `window_samples(seconds: float)`

### DriftAdaptiveClassifier


*Module:* `dense_armor.utility.learn.online_classifiers`
*Bases:* Classifier
*Guide:* [learn/online_classifiers.md](../learn/online_classifiers.md)

Classifier with a background copy swapped in on drift detection.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, y, t=None)`
- `learn_one(x, y)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `predict_many(X, t=None)`
- `predict_one(x: dict, t=None)`
- `predict_proba_many(X, t=None)`
- `predict_proba_one(x)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### DriftDetector


*Module:* `dense_armor.roles.drift_detector`
*Bases:* Estimator

Unsupervised drift detector over a scalar stream.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `update(x: float, t=None)`
- `update_many(X, t=None)`
- `window_samples(seconds: float)`

### EWScaler


*Module:* `dense_armor.utility.preprocessing.scale`
*Bases:* Transformer
*Guide:* [preprocessing/index.md](../preprocessing/index.md)

Per-feature standardization with exponentially weighted stats.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: Any, y=None, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: Any, t=None)`
- `window_samples(seconds: float)`

### EWStats


*Module:* `dense_armor.utility.stats.moments`
*Bases:* Transformer

Exponentially weighted mean and variance, one sample at a time.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: dict, y=None, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: dict, t=None)`
- `update(x: dict, y=None, t=None)`
- `window_samples(seconds: float)`

### Estimate


*Module:* `dense_armor.roles.uncertainty`

A prediction with its variance.

### Estimator


*Module:* `dense_armor.roles.estimator`
*Bases:* Root

Root class for every Dense-Armor estimator.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### FeatureHasher


*Module:* `dense_armor.utility.preprocessing.text`
*Bases:* Transformer
*Guide:* [preprocessing/index.md](../preprocessing/index.md)

Signed feature hashing (Weinberger et al. 2009, Theorem 3).

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: Any, y=None, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: Any, t=None)`
- `window_samples(seconds: float)`

### FrameFeatures


*Module:* `dense_armor.utility.vision.features`
*Bases:* Transformer
*Guide:* [vision/index.md](../vision/index.md)

Turn one frame into a fixed-length dict of float features.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: Any, y=None, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: Any, t=None)`
- `window_samples(seconds: float)`

### FuncTransformer


*Module:* `dense_armor.utility.compose.func`
*Bases:* Transformer

Apply ``fn`` to channels, or to the whole dict.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: Any, y=None, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: Any, t=None)`
- `window_samples(seconds: float)`

### HalfSpaceTrees


*Module:* `dense_armor.utility.anomaly.hst`
*Bases:* AnomalyDetector

Half-Space Trees for streaming anomaly detection.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `is_outlier(x: dict, threshold=None)`
- `learn_many(X, t=None)`
- `learn_one(x: dict, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `score_many(X, t=None)`
- `score_one(x: dict, t=None)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### HampelFilter


*Module:* `dense_armor.utility.anomaly.filters`
*Bases:* Transformer
*Guide:* [anomaly/streaming.md](../anomaly/streaming.md)

Streaming Hampel filter: replaces outliers with the causal-window median.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x)`
- `window_samples(seconds: float)`

### HampelScorer


*Module:* `dense_armor.utility.anomaly.filters`
*Bases:* AnomalyDetector
*Guide:* [anomaly/streaming.md](../anomaly/streaming.md)

Hampel outlier scorer on the causal window.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `is_outlier(x)`
- `learn_many(X, t=None)`
- `learn_one(x)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `score_many(X, t=None)`
- `score_one(x)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### Health


*Module:* `dense_armor.roles.safety`
*Bases:* str, Enum

Health states of an estimator in a control loop.

### HealthMonitor


*Module:* `dense_armor.roles.safety`
*Bases:* Root

Running residual tracker that maps to a :class:`Health` state.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `update(residual: float)`
- `window_samples(seconds: float)`

### HoeffdingAdaptiveTreeClassifier


*Module:* `dense_armor.utility.tree.adaptive`
*Bases:* HoeffdingTreeClassifier
*Guide:* [tree/index.md](../tree/index.md)

Hoeffding Adaptive Tree classifier.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `explain_one(x: Any, t=None)`
- `get_params(deep=True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: Any, y: Any, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `predict_many(X, t=None)`
- `predict_one(x: Any, t=None)`
- `predict_proba_many(X, t=None)`
- `predict_proba_one(x: Any, t=None)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### HoeffdingAnytimeTreeClassifier


*Module:* `dense_armor.utility.tree.efdt`
*Bases:* HoeffdingTreeClassifier
*Guide:* [tree/index.md](../tree/index.md)

Hoeffding Anytime Tree classifier.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `explain_one(x: Any, t=None)`
- `get_params(deep=True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: Any, y: Any, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `predict_many(X, t=None)`
- `predict_one(x: Any, t=None)`
- `predict_proba_many(X, t=None)`
- `predict_proba_one(x: Any, t=None)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### HoeffdingTreeClassifier


*Module:* `dense_armor.utility.tree.hoeffding`
*Bases:* Classifier
*Guide:* [tree/index.md](../tree/index.md)

Hoeffding tree for incremental classification.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `explain_one(x: Any, t=None)`
- `get_params(deep=True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: Any, y: Any, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `predict_many(X, t=None)`
- `predict_one(x: Any, t=None)`
- `predict_proba_many(X, t=None)`
- `predict_proba_one(x: Any, t=None)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### HyperLogLog


*Module:* `dense_armor.utility.stats.sketches`
*Bases:* Transformer

HyperLogLog distinct-count sketch (Flajolet et al. 2007).

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `count()`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: dict, y=None, t=None)`
- `load_state_dict(state: dict)`
- `merge(other: HyperLogLog)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: dict, t=None)`
- `update(x: dict, y=None, t=None)`
- `window_samples(seconds: float)`

### InconsistentVersionWarning


*Module:* `dense_armor.roles.root`
*Bases:* UserWarning

Unpickled a Dense-Armor estimator from a different library version.

### IncrementalPCA


*Module:* `dense_armor.utility.vision.reduce`
*Bases:* Transformer
*Guide:* [vision/index.md](../vision/index.md)

Online PCA: Oja's rule for one component, stochastic subspace iteration with QR for several.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: dict, y=None, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: dict, t=None)`
- `window_samples(seconds: float)`

### JointDerivatives


*Module:* `dense_armor.utility.preprocessing.joints`
*Bases:* Transformer
*Guide:* [preprocessing/index.md](../preprocessing/index.md)

Velocity, acceleration and jerk from joint positions.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: Any, y=None, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: Any, t=None)`
- `window_samples(seconds: float)`

### JointLimits


*Module:* `dense_armor.roles.physics`

Position and velocity limits of every joint in a robot.

Public methods:

- `as_guard()`
- `in_bounds(q: Union)`

### JointPower


*Module:* `dense_armor.utility.preprocessing.joints`
*Bases:* Transformer
*Guide:* [preprocessing/index.md](../preprocessing/index.md)

Mechanical power ``tau * qd`` per joint and its total.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: Any, y=None, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: Any, t=None)`
- `window_samples(seconds: float)`

### KSWIN


*Module:* `dense_armor.utility.drift.kswin`
*Bases:* DriftDetector
*Guide:* [drift/detectors.md](../drift/detectors.md)

Kolmogorov-Smirnov Windowing drift detector.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `update(x: float, t=None)`
- `update_many(X, t=None)`
- `window_samples(seconds: float)`

### LEGO


*Module:* `dense_armor.utility.learn.metric_learning`
*Bases:* MetricLearner
*Guide:* [learn/metric_learning.md](../learn/metric_learning.md)

Online metric learning via LogDet regularization and exact gradient.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `distance(x: dict, x2: dict)`
- `get_params(deep=True)`
- `learn_pair(x: dict, x2: dict, y: float)`
- `learn_triplet(x: dict, x_pos: dict, x_neg: dict)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `percentile_targets(X, y, low=5.0, high=95.0)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### LLMEmbeddingClassifier


*Module:* `dense_armor.bridges.llm`
*Bases:* Estimator

Classifier whose input is text and whose features come from ``embed``.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_json(payload: str)`
- `learn_one(text: str, y: Any, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `predict_json(payload: str)`
- `predict_one(text: str, t=None)`
- `predict_proba_json(payload: str)`
- `predict_proba_one(text: str, t=None)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### LODA


*Module:* `dense_armor.utility.anomaly.loda`
*Bases:* AnomalyDetector

Lightweight Online Detector of Anomalies.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `is_outlier(x: dict, threshold=None)`
- `learn_many(X, t=None)`
- `learn_one(x: dict, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `score_many(X, t=None)`
- `score_one(x: dict, t=None)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### MetricKNNClassifier


*Module:* `dense_armor.utility.learn.metric_learning`
*Bases:* Classifier
*Guide:* [learn/metric_learning.md](../learn/metric_learning.md)

K-nearest-neighbours classifier that learns its metric online.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: dict, y)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `predict_many(X, t=None)`
- `predict_one(x: dict, t=None)`
- `predict_proba_many(X, t=None)`
- `predict_proba_one(x: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### MetricLearner


*Module:* `dense_armor.utility.learn.metric_learning`
*Bases:* Root
*Guide:* [learn/metric_learning.md](../learn/metric_learning.md)

Root class for online metric learners.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `distance(x: dict, x2: dict)`
- `get_params(deep=True)`
- `learn_pair(x: dict, x2: dict, y)`
- `learn_triplet(x: dict, x_pos: dict, x_neg: dict)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### MinMaxScaler


*Module:* `dense_armor.utility.preprocessing.scale`
*Bases:* Transformer
*Guide:* [preprocessing/index.md](../preprocessing/index.md)

Rescale each feature to ``[0, 1]`` with running min and max.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: Any, y=None, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: Any, t=None)`
- `window_samples(seconds: float)`

### ModelWrapper


*Module:* `dense_armor.roles.wrapper`
*Bases:* Estimator

Delegates ``_supervised`` and ``_multiclass`` to the wrapped estimator. estimator.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### MondrianForestClassifier


*Module:* `dense_armor.utility.tree.mondrian`
*Bases:* Classifier, _MondrianBase
*Guide:* [tree/index.md](../tree/index.md)

Online Mondrian forest for classification.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: Any, y: Any, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `predict_many(X, t=None)`
- `predict_one(x: Any, t=None)`
- `predict_proba_many(X, t=None)`
- `predict_proba_one(x: Any, t=None)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### MondrianForestRegressor


*Module:* `dense_armor.utility.tree.mondrian`
*Bases:* Regressor, _MondrianBase
*Guide:* [tree/index.md](../tree/index.md)

Online Mondrian forest for regression.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: Any, y: Any, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `predict_many(X, t=None, return_std=False)`
- `predict_one(x: Any, t=None, return_std=False, return_estimate=False)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### OASIS


*Module:* `dense_armor.utility.learn.metric_learning`
*Bases:* MetricLearner
*Guide:* [learn/metric_learning.md](../learn/metric_learning.md)

Online metric learning via passive-aggressive bilinear similarity.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `distance(x: dict, x2: dict)`
- `get_params(deep=True)`
- `learn_pair(x: dict, x2: dict, y)`
- `learn_triplet(x: dict, x_pos: dict, x_neg: dict)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### OneClassSGD


*Module:* `dense_armor.utility.anomaly.ocsvm`
*Bases:* AnomalyDetector

One-Class SVM solved by SGD on random Fourier features.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `is_outlier(x: dict, threshold=None)`
- `learn_many(X, t=None)`
- `learn_one(x: dict, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `score_many(X, t=None)`
- `score_one(x: dict, t=None)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### OnlineGaussianNB


*Module:* `dense_armor.utility.learn.online_classifiers`
*Bases:* Classifier
*Guide:* [learn/online_classifiers.md](../learn/online_classifiers.md)

Gaussian naive Bayes updated one sample at a time.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, y, t=None)`
- `learn_one(x, y)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `predict_many(X, t=None)`
- `predict_one(x: dict, t=None)`
- `predict_proba_many(X, t=None)`
- `predict_proba_one(x)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### OnlineIsolationForest


*Module:* `dense_armor.utility.anomaly.oiforest`
*Bases:* AnomalyDetector

Online Isolation Forest.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `is_outlier(x: dict, threshold=None)`
- `learn_many(X, t=None)`
- `learn_one(x: dict, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `score_many(X, t=None)`
- `score_one(x: dict, t=None)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### OnlineKMeans


*Module:* `dense_armor.utility.cluster.kmeans`
*Bases:* Transformer
*Guide:* [cluster/index.md](../cluster/index.md)

Sequential k-means, one sample at a time.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: Any, y=None, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `predict_one(x: Any, t=None)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: Any, t=None)`
- `window_samples(seconds: float)`

### OnlinePlattScaling


*Module:* `dense_armor.utility.learn.calibration`
*Bases:* ModelWrapper, Classifier
*Guide:* [learn/calibration.md](../learn/calibration.md)

Online Platt scaling.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, y, t=None)`
- `learn_one(x, y, kwargs)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `predict_many(X, t=None)`
- `predict_one(x: dict, t=None)`
- `predict_proba_many(X, t=None)`
- `predict_proba_one(x, kwargs)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### OnlineRobustMahalanobis


*Module:* `dense_armor.utility.anomaly.mahalanobis`
*Bases:* AnomalyDetector
*Guide:* [anomaly/streaming.md](../anomaly/streaming.md)

Online robust Mahalanobis scorer for multichannel signals.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `is_outlier(x: dict, threshold=None)`
- `learn_many(X, t=None)`
- `learn_one(x: dict, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `score_many(X, t=None)`
- `score_one(x: dict, t=None)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### OnlineSoftmaxRegression


*Module:* `dense_armor.utility.learn.online_classifiers`
*Bases:* Classifier
*Guide:* [learn/online_classifiers.md](../learn/online_classifiers.md)

Multinomial logistic regression with AdaGrad.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, y, t=None)`
- `learn_one(x, y)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `predict_many(X, t=None)`
- `predict_one(x: dict, t=None)`
- `predict_proba_many(X, t=None)`
- `predict_proba_one(x)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### POLA


*Module:* `dense_armor.utility.learn.metric_learning`
*Bases:* MetricLearner
*Guide:* [learn/metric_learning.md](../learn/metric_learning.md)

Online pseudo-metric learning with PSD projection.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `distance(x: dict, x2: dict)`
- `get_params(deep=True)`
- `learn_pair(x: dict, x2: dict, y: int)`
- `learn_triplet(x: dict, x_pos: dict, x_neg: dict)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### PageHinkley


*Module:* `dense_armor.utility.drift.page_hinkley`
*Bases:* DriftDetector
*Guide:* [drift/detectors.md](../drift/detectors.md)

Page-Hinkley test for a change in the mean of a scalar stream.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `update(x: float, t=None)`
- `update_many(X, t=None)`
- `window_samples(seconds: float)`

### PatchFeatures


*Module:* `dense_armor.utility.vision.patches`
*Bases:* Transformer
*Guide:* [vision/patches.md](../vision/patches.md)

Cut a frame into patches and describe each one locally.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: Any, y=None, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: Any, t=None)`
- `window_samples(seconds: float)`

### PatchMemory


*Module:* `dense_armor.utility.vision.patches`
*Bases:* AnomalyDetector
*Guide:* [vision/patches.md](../vision/patches.md)

Memory bank of normal patch descriptors with a bounded size.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `finalize()`
- `get_params(deep=True)`
- `is_outlier(x: Any, threshold=1.0)`
- `learn_many(X, t=None)`
- `learn_one(x: Any, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `patch_scores(x: Any)`
- `score_many(X, t=None)`
- `score_one(x: Any, t=None)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### PhysicalLimitsGuard


*Module:* `dense_armor.roles.physics`

Guard that flags samples whose joint positions are out of bounds.

### ProfileResult


*Module:* `dense_armor.roles.realtime`

Latency and memory profile of one estimator over a stream.

### Protected


*Module:* `dense_armor.roles.protection`
*Bases:* Estimator

Score first; on flag, skip learning and return the fallback.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_one(x: dict, y=None, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `predict_one(x: dict, t=None)`
- `score_one(x: dict, t=None)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### PureEW


*Module:* `dense_armor.roles.realtime`
*Bases:* Root

Exponentially weighted mean in pure form, ready for ``lax.scan``.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `init_state()`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `scan(stream: Array)`
- `set_params(params: Any)`
- `state_dict()`
- `step(state: tuple, v: Array)`
- `window_samples(seconds: float)`

### QueueResampler


*Module:* `dense_armor.utility.preprocessing.imbalance`
*Bases:* Classifier
*Guide:* [preprocessing/index.md](../preprocessing/index.md)

Resampling wrapper for imbalanced online streams.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: Any, y: Any, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `predict_many(X, t=None)`
- `predict_one(x: Any, t=None)`
- `predict_proba_many(X, t=None)`
- `predict_proba_one(x: Any, t=None)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### RandomProjection


*Module:* `dense_armor.utility.vision.reduce`
*Bases:* Transformer
*Guide:* [vision/index.md](../vision/index.md)

Fixed Gaussian random projection to ``k`` dimensions.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: dict, y=None, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: dict, t=None)`
- `window_samples(seconds: float)`

### RealtimePipeline


*Module:* `dense_armor.roles.realtime`

A list of estimators that must fit inside a control-loop period.

Public methods:

- `learn_one(sig: Signal)`

### RecursiveLeastSquares


*Module:* `dense_armor.utility.learn.online_dynamics`
*Bases:* Regressor
*Guide:* [learn/online_dynamics.md](../learn/online_dynamics.md)

Recursive least squares with exponential forgetting.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, y, t=None)`
- `learn_one(x, y)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `predict_many(X, t=None, return_std=False)`
- `predict_one(x)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### Regressor


*Module:* `dense_armor.roles.regressor`
*Bases:* Estimator

Supervised regressor; scalar or vector target.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: dict, y: Any, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `predict_many(X, t=None, return_std=False)`
- `predict_one(x: dict, t=None, return_std=False)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### RobustScaler


*Module:* `dense_armor.utility.preprocessing.scale`
*Bases:* Transformer
*Guide:* [preprocessing/index.md](../preprocessing/index.md)

Robust per-feature standardization with rolling median and MAD.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: Any, y=None, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: Any, t=None)`
- `window_samples(seconds: float)`

### RollingCorrelation


*Module:* `dense_armor.utility.stats.dependence`
*Bases:* Transformer

Pearson correlation of two channels over a bounded window.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: dict, y=None, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: dict, t=None)`
- `update(x: dict, y=None, t=None)`
- `window_samples(seconds: float)`

### RollingCovariance


*Module:* `dense_armor.utility.stats.dependence`
*Bases:* Transformer

Sample covariance of two channels over a bounded window.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: dict, y=None, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: dict, t=None)`
- `update(x: dict, y=None, t=None)`
- `window_samples(seconds: float)`

### RollingIQR


*Module:* `dense_armor.utility.stats.robust`
*Bases:* _RollingWindow

Interquartile range of the last ``window`` samples.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: dict, y=None, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: dict, t=None)`
- `update(x: dict, y=None, t=None)`
- `window_samples(seconds: float)`

### RollingMAD


*Module:* `dense_armor.utility.stats.robust`
*Bases:* _RollingWindow

Median absolute deviation of the last ``window`` samples, scaled.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: dict, y=None, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: dict, t=None)`
- `update(x: dict, y=None, t=None)`
- `window_samples(seconds: float)`

### RollingMedian


*Module:* `dense_armor.utility.stats.robust`
*Bases:* _RollingWindow

Median of the last ``window`` samples.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: dict, y=None, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: dict, t=None)`
- `update(x: dict, y=None, t=None)`
- `window_samples(seconds: float)`

### RollingMedianVector


*Module:* `dense_armor.utility.stats.robust`
*Bases:* Transformer

One :class:`RollingMedian` per channel, one update call.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: dict, y=None, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: dict, t=None)`
- `update(x: dict, y=None, t=None)`
- `window_samples(seconds: float)`

### RollingQuantile


*Module:* `dense_armor.utility.stats.robust`
*Bases:* _RollingWindow

Arbitrary quantile of the last ``window`` samples.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: dict, y=None, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: dict, t=None)`
- `update(x: dict, y=None, t=None)`
- `window_samples(seconds: float)`

### Root


*Module:* `dense_armor.roles.root`

Root of the Dense-Armor online-learning stack.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### RunningCorrelation


*Module:* `dense_armor.utility.stats.dependence`
*Bases:* Transformer

Incremental Pearson correlation between two channels.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: dict, y=None, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: dict, t=None)`
- `update(x: dict, y=None, t=None)`
- `window_samples(seconds: float)`

### RunningCovariance


*Module:* `dense_armor.utility.stats.dependence`
*Bases:* Transformer

Incremental covariance of two channels.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: dict, y=None, t=None)`
- `load_state_dict(state: dict)`
- `merge(other: RunningCovariance)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: dict, t=None)`
- `update(x: dict, y=None, t=None)`
- `window_samples(seconds: float)`

### RunningMoments


*Module:* `dense_armor.utility.stats.moments`
*Bases:* Transformer

Incremental mean, variance, skewness, kurtosis for one feature.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: dict, y=None, t=None)`
- `load_state_dict(state: dict)`
- `merge(other: RunningMoments)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: dict, t=None)`
- `update(x: dict, y=None, t=None)`
- `window_samples(seconds: float)`

### RunningMomentsVector


*Module:* `dense_armor.utility.stats.moments`
*Bases:* Transformer

One ``RunningMoments`` per channel, updated in a single call.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: dict, y=None, t=None)`
- `load_state_dict(state: dict)`
- `merge(other: RunningMomentsVector)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: dict, t=None)`
- `update(x: dict, y=None, t=None)`
- `window_samples(seconds: float)`

### SGTClassifier


*Module:* `dense_armor.utility.tree.sgt`
*Bases:* Classifier, _SGTBase
*Guide:* [tree/index.md](../tree/index.md)

Stochastic gradient tree for binary classification.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `explain_one(x: Any, t=None)`
- `get_params(deep=True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: Any, y: Any, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `predict_many(X, t=None)`
- `predict_one(x: Any, t=None)`
- `predict_proba_many(X, t=None)`
- `predict_proba_one(x: Any, t=None)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### SGTRegressor


*Module:* `dense_armor.utility.tree.sgt`
*Bases:* Regressor, _SGTBase
*Guide:* [tree/index.md](../tree/index.md)

Stochastic gradient tree for regression.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `explain_one(x: Any, t=None)`
- `get_params(deep=True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: Any, y: Any, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `predict_many(X, t=None, return_std=False)`
- `predict_one(x: Any, t=None, return_std=False)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### SafeEstimator


*Module:* `dense_armor.roles.safety`
*Bases:* Estimator

Wrap an estimator with a guard and a fallback.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `is_flagged(x: Any)`
- `learn_one(x: Any, y=None, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `predict_one(x: Any, t=None, return_std=False, return_estimate=False)`
- `restore(path: str | pathlib._local.Path)`
- `save(path: str | pathlib._local.Path)`
- `score_one(x: Any, t=None)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### Select


*Module:* `dense_armor.utility.compose.select`
*Bases:* Transformer

Keep only the named channels.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: Any, y=None, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: Any, t=None)`
- `window_samples(seconds: float)`

### SelectKBest


*Module:* `dense_armor.utility.preprocessing.select`
*Bases:* Transformer
*Guide:* [preprocessing/index.md](../preprocessing/index.md)

Keep the ``k`` features with the highest absolute running correlation.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: Any, y=None, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: Any, t=None)`
- `window_samples(seconds: float)`

### SigmaClipScorer


*Module:* `dense_armor.utility.anomaly.filters`
*Bases:* AnomalyDetector
*Guide:* [anomaly/streaming.md](../anomaly/streaming.md)

Iterative sigma-clipping outlier scorer on the causal window.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `is_outlier(x)`
- `learn_many(X, t=None)`
- `learn_one(x)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `score_many(X, t=None)`
- `score_one(x)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### Signal


*Module:* `dense_armor.roles.signal`
*Bases:* dict

Dict-like one-sample reading, with vector form and metadata.

Public methods:

- `from_dict(x: dict, units=None, t=None, names=None)`
- `stacked(others: Iterable)`
- `to_dict()`

### SpaceSaving


*Module:* `dense_armor.utility.stats.sketches`
*Bases:* Transformer

Space-Saving heavy hitters (Metwally et al. 2005).

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `estimate(item: Any)`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: dict, y=None, t=None)`
- `load_state_dict(state: dict)`
- `merge(other: SpaceSaving)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `top(n=None)`
- `transform_many(X, t=None)`
- `transform_one(x: dict, t=None)`
- `update(x: dict, y=None, t=None)`
- `window_samples(seconds: float)`

### StandardScaler


*Module:* `dense_armor.utility.preprocessing.scale`
*Bases:* Transformer
*Guide:* [preprocessing/index.md](../preprocessing/index.md)

Per-feature standardization with running mean and std.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: Any, y=None, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: Any, t=None)`
- `window_samples(seconds: float)`

### StreamingDeviationScorer


*Module:* `dense_armor.utility.anomaly.deviation`
*Bases:* AnomalyDetector
*Guide:* [anomaly/streaming.md](../anomaly/streaming.md)

`StreamingDeviationDetector` as a river anomaly detector.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `score_many(X, t=None)`
- `score_one(x)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### TDigest


*Module:* `dense_armor.utility.stats.quantiles`
*Bases:* Transformer

Clustered sketch with small clusters near the tails (Dunning 2019).

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: dict, y=None, t=None)`
- `load_state_dict(state: dict)`
- `merge(other: TDigest)`
- `mutate(new_attrs: dict)`
- `quantile(q: float)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: dict, t=None)`
- `update(x: dict, y=None, t=None)`
- `window_samples(seconds: float)`

### TFIDF


*Module:* `dense_armor.utility.preprocessing.text`
*Bases:* Transformer
*Guide:* [preprocessing/index.md](../preprocessing/index.md)

Online TF-IDF following Silajev 2026, section 1.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: Any, y=None, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: Any, t=None)`
- `window_samples(seconds: float)`

### Tokenizer


*Module:* `dense_armor.utility.preprocessing.text`
*Bases:* Transformer
*Guide:* [preprocessing/index.md](../preprocessing/index.md)

Lower-case word tokens, with optional n-grams.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: Any, y=None, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: Any, t=None)`
- `window_samples(seconds: float)`

### Transformer


*Module:* `dense_armor.roles.transformer`
*Bases:* Estimator

Unsupervised transformer: ``transform_one(x) -> dict``.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: dict, y=None, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: dict, t=None)`
- `window_samples(seconds: float)`

### TransformerSupervised


*Module:* `dense_armor.roles.transformer`
*Bases:* Transformer

Supervised transformer: ``learn_one(x, y)``.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: dict, y=None, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: dict, t=None)`
- `window_samples(seconds: float)`

### TukeyScorer


*Module:* `dense_armor.utility.anomaly.filters`
*Bases:* AnomalyDetector
*Guide:* [anomaly/streaming.md](../anomaly/streaming.md)

Tukey-fences outlier scorer on the causal window.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `is_outlier(x)`
- `learn_many(X, t=None)`
- `learn_one(x)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `score_many(X, t=None)`
- `score_one(x)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### UnitCheckedPipeline


*Module:* `dense_armor.roles.physics`
*Bases:* Transformer

A pipeline that checks units at build time.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x, y=None, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x, t=None)`
- `window_samples(seconds: float)`

### UnitSpec


*Module:* `dense_armor.roles.physics`

Input and output units of an estimator, as ``{name: unit}`` dicts.

### VarianceThreshold


*Module:* `dense_armor.utility.preprocessing.select`
*Bases:* Transformer
*Guide:* [preprocessing/index.md](../preprocessing/index.md)

Drop features whose running variance is at or below ``threshold``.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `get_params(deep=True)`
- `learn_many(X, t=None)`
- `learn_one(x: Any, y=None, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: Any, t=None)`
- `window_samples(seconds: float)`

## Metrics

### Accuracy


*Module:* `dense_armor.utility.metrics.classification`
*Bases:* Metric

Fraction of samples where ``y_pred == y_true``.

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### BalancedAccuracy


*Module:* `dense_armor.utility.metrics.classification`
*Bases:* Metric

Mean of the per-class recall.

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### BrierScore


*Module:* `dense_armor.utility.metrics.classification`
*Bases:* Metric

Mean squared error of predicted probabilities.

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### CohenKappa


*Module:* `dense_armor.utility.metrics.classification`
*Bases:* Metric

Cohen's kappa: agreement corrected for chance.

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### ConfusionMatrix


*Module:* `dense_armor.utility.metrics.classification`
*Bases:* Metric

Multiclass confusion matrix.

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### EventMetrics


*Module:* `dense_armor.utility.metrics.events`

Collect the event metrics for a stream of alarms.

Public methods:

- `add_alarm(t: float)`
- `detection_delays_s()`
- `false_alarms()`
- `false_alarms_per_hour()`
- `missed_events()`
- `nab_score()`
- `range_f1()`
- `range_precision()`
- `range_recall()`
- `report()`

### EventWindow


*Module:* `dense_armor.utility.metrics.events`

A labelled event interval.

Public methods:

- `contains(t: float)`
- `n_points(dt: float)`

### F1


*Module:* `dense_armor.utility.metrics.classification`
*Bases:* FBeta

F1 score: the harmonic mean of precision and recall.

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### FBeta


*Module:* `dense_armor.utility.metrics.classification`
*Bases:* _PRFBase

Weighted harmonic mean of precision and recall.

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### GaussianNLL


*Module:* `dense_armor.utility.metrics.regression`
*Bases:* Metric

Gaussian negative log-likelihood.

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### IntervalCoverage


*Module:* `dense_armor.utility.metrics.regression`
*Bases:* Metric

Fraction of true labels inside a prediction interval.

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### LogLoss


*Module:* `dense_armor.utility.metrics.classification`
*Bases:* Metric

Binary log-loss (cross-entropy).

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### MeanAbsoluteError


*Module:* `dense_armor.utility.metrics.regression`
*Bases:* Metric

Mean absolute error, one sample at a time.

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### MeanIntervalWidth


*Module:* `dense_armor.utility.metrics.regression`
*Bases:* Metric

Mean width of a prediction interval.

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### MeanSquaredError


*Module:* `dense_armor.utility.metrics.regression`
*Bases:* Metric

Mean squared error, one sample at a time.

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### Metric


*Module:* `dense_armor.utility.metrics.base`

Base class for online metrics.

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### OnlineModelSelection


*Module:* `dense_armor.utility.evaluate.model_selection`

Track several models on a stream and expose the current best.

Public methods:

- `best_model()`
- `current_best()`
- `run(stream: Iterable)`
- `scores()`

### PointAdjustedF1


*Module:* `dense_armor.utility.metrics.events`
*Bases:* Metric

Point-adjusted F1 (Xu et al. 2018).

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### Precision


*Module:* `dense_armor.utility.metrics.classification`
*Bases:* _PRFBase

Fraction of predicted positives that are correct.

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### R2Score


*Module:* `dense_armor.utility.metrics.regression`
*Bases:* Metric

Coefficient of determination, Welford running mean.

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### Recall


*Module:* `dense_armor.utility.metrics.classification`
*Bases:* _PRFBase

Fraction of true positives that are detected.

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### Rolling


*Module:* `dense_armor.utility.metrics.rolling`
*Bases:* Metric

Rolling window over any metric with ``revert``.

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### RollingAUC


*Module:* `dense_armor.utility.metrics.roc_auc`
*Bases:* Metric

Rolling ROC-AUC, exact on the last ``window`` samples.

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### RootMeanSquaredError


*Module:* `dense_armor.utility.metrics.regression`
*Bases:* MeanSquaredError

Square root of the mean squared error.

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

## Core

### AIEngineVisualizer


*Module:* `dense_armor.core.visualizer`
*Guide:* [toolkit/toolkit.md](../toolkit/toolkit.md)

Export tools for scientific traceability.

Public methods:

- `export_provenance_archive(run_history: list, filename='ai_provenance_archive.json')`
- `export_trend_report_text(raw_signal: ndarray, filtered_signal: ndarray, filename='ai_trend_report.txt')`

### AIHardwareProfiler


*Module:* `dense_armor.core.noise`
*Guide:* [toolkit/toolkit.md](../toolkit/toolkit.md)

Profiles the host architecture (CPU/GPU/JAX) to set the thresholds.

Public methods:

- `get_profile_summary()`

### AdaptiveSignalStabilizer


*Module:* `dense_armor.core.engine`
*Guide:* [shields/engine.md](../shields/engine.md)

Adaptive anti-panic filter for chaotic AI signals.

Public methods:

- `calibrate_macro_context(raw_batch: ndarray)`
- `calibrate_macro_context_independent(raw_batch: ndarray)`
- `filter_batch_scenarios(raw_batch: ndarray, hard_clamp_mask=None)`
- `filter_batch_scenarios_independent(raw_batch: ndarray, hard_clamp_mask=None)`
- `filter_data_stream(raw_data: ndarray)`

### BitwisePermutationEngine


*Module:* `dense_armor.core.vector`
*Guide:* [toolkit/toolkit.md](../toolkit/toolkit.md)

Bit-mask manipulation of combinatorial vectors (2^n space).

Public methods:

- `apply_bitwise_swap(data: ndarray, target_bit: int, control_bit: int)`

### CompactJsonFormatter


*Module:* `dense_armor.core.logger`
*Bases:* Formatter
*Guide:* [toolkit/toolkit.md](../toolkit/toolkit.md)

Formatter instances are used to convert a LogRecord to text.

Public methods:

- `format(record: LogRecord)`
- `formatException(ei)`
- `formatMessage(record)`
- `formatStack(stack_info)`
- `formatTime(record, datefmt=None)`
- `usesTime()`

### DynamicAICodegen


*Module:* `dense_armor.core.compiler`
*Guide:* [toolkit/toolkit.md](../toolkit/toolkit.md)

Compiles textual recipes into JAX instruction matrices and runs the resulting pipeline.

Public methods:

- `compile_pipeline(text_instructions: list)`
- `compute_gradients(input_data: ndarray, compiled_ops: ndarray)`
- `load_compiled_pipeline(filename='compiled_recipe.npy')`
- `run_dynamic_pipeline(input_data: ndarray, compiled_ops: ndarray)`
- `run_pipeline_with_chunking(input_data: ndarray, compiled_ops: ndarray, chunk_size=500)`
- `save_compiled_pipeline(compiled_ops: ndarray, filename='compiled_recipe.npy')`

### ImageChunker


*Module:* `dense_armor.core.chunk`
*Guide:* [toolkit/toolkit.md](../toolkit/toolkit.md)

Splits and rejoins data batches and instruction lists into fixed-size chunks.

Public methods:

- `execute_pipeline_chunked(codegen_engine, input_vector: ndarray, compiled_ops: list)`
- `merge_chunks(chunks_list: list)`
- `patch_and_scan_parameters(template_ops: Array, dynamic_parameters: Array)`
- `split_array(array: ndarray)`

### MemoryPressureError


*Module:* `dense_armor.core.memory`
*Bases:* Exception
*Guide:* [toolkit/toolkit.md](../toolkit/toolkit.md)

Raised when system memory (RAM/VRAM) is insufficient.

### MinimalConsoleFormatter


*Module:* `dense_armor.core.logger`
*Bases:* Formatter
*Guide:* [toolkit/toolkit.md](../toolkit/toolkit.md)

Formatter instances are used to convert a LogRecord to text.

Public methods:

- `format(record: LogRecord)`
- `formatException(ei)`
- `formatMessage(record)`
- `formatStack(stack_info)`
- `formatTime(record, datefmt=None)`
- `usesTime()`

### ParametricScenarioSimulator


*Module:* `dense_armor.core.vector`
*Guide:* [toolkit/toolkit.md](../toolkit/toolkit.md)

Massive parallel Monte Carlo simulations and decision collapse.

Public methods:

- `collapse_decision(distribution_vector: ndarray, target_idx: int)`
- `run_parallel_scenarios(base_state: float, parameters_batch: ndarray)`

### PipelineProfiler


*Module:* `dense_armor.core.profiler`
*Guide:* [toolkit/toolkit.md](../toolkit/toolkit.md)

Profile pipeline and filter performance in microseconds.

Public methods:

- `measure_microseconds(codegen_instance, input_data: ndarray, compiled_ops: ndarray, repetitions=100)`
- `measure_stabilizer_microseconds(stabilizer_instance, raw_batch: ndarray, repetitions=100)`

### StochasticAdversarialNoise


*Module:* `dense_armor.core.noise`
*Guide:* [toolkit/toolkit.md](../toolkit/toolkit.md)

Injects probabilistic noise or adversarial perturbations into AI tensors.

Public methods:

- `inject_noise(data_vector: ndarray, noise_type: str, intensity: float, seed=42)`

### TensorVault


*Module:* `dense_armor.core.tensor`
*Guide:* [toolkit/toolkit.md](../toolkit/toolkit.md)

Compact, optimized vault for static and parametric matrices.

Public methods:

- `get_backend_info()`
- `get_parametric_transform(name: str, p: float)`
- `get_static_transform(name: str)`

### UniversalMemoryGuard


*Module:* `dense_armor.core.memory`
*Guide:* [toolkit/toolkit.md](../toolkit/toolkit.md)

Pre-flight RAM and VRAM check before every heavy allocation.

Public methods:

- `calculate_optimal_chunks(total_items: int, item_size_bytes: int)`
- `check_memory_safety()`

## Mcp_server

### CleanSignalInput


*Module:* `dense_armor.mcp_server.models`
*Bases:* BaseModel

!!! abstract "Usage Documentation"

*Inherits the full pydantic* ``BaseModel`` *API* (``model_validate``, ``model_dump``, ``model_json_schema``, ...).

### DetectAnomaliesInput


*Module:* `dense_armor.mcp_server.models`
*Bases:* BaseModel

!!! abstract "Usage Documentation"

*Inherits the full pydantic* ``BaseModel`` *API* (``model_validate``, ``model_dump``, ``model_json_schema``, ...).

### HealSeriesInput


*Module:* `dense_armor.mcp_server.models`
*Bases:* BaseModel

!!! abstract "Usage Documentation"

*Inherits the full pydantic* ``BaseModel`` *API* (``model_validate``, ``model_dump``, ``model_json_schema``, ...).

### RobustFilterInput


*Module:* `dense_armor.mcp_server.models`
*Bases:* BaseModel

!!! abstract "Usage Documentation"

*Inherits the full pydantic* ``BaseModel`` *API* (``model_validate``, ``model_dump``, ``model_json_schema``, ...).

### StreamEndInput


*Module:* `dense_armor.mcp_server.models`
*Bases:* BaseModel

!!! abstract "Usage Documentation"

*Inherits the full pydantic* ``BaseModel`` *API* (``model_validate``, ``model_dump``, ``model_json_schema``, ...).

### StreamStartInput


*Module:* `dense_armor.mcp_server.models`
*Bases:* BaseModel

!!! abstract "Usage Documentation"

*Inherits the full pydantic* ``BaseModel`` *API* (``model_validate``, ``model_dump``, ``model_json_schema``, ...).

### StreamUpdateInput


*Module:* `dense_armor.mcp_server.models`
*Bases:* BaseModel

!!! abstract "Usage Documentation"

*Inherits the full pydantic* ``BaseModel`` *API* (``model_validate``, ``model_dump``, ``model_json_schema``, ...).

## Vision

### ArrayStream


*Module:* `dense_armor.utility.vision.streams`
*Guide:* [vision/index.md](../vision/index.md)

Iterate over a numpy array of frames.

### CameraStream


*Module:* `dense_armor.utility.vision.streams`
*Guide:* [vision/index.md](../vision/index.md)

Iterate over camera frames through OpenCV.

### Frame


*Module:* `dense_armor.utility.vision.streams`
*Guide:* [vision/index.md](../vision/index.md)

One image frame with its timestamp.

Public methods:

- `to_dict()`

### ImageFolderStream


*Module:* `dense_armor.utility.vision.streams`
*Guide:* [vision/index.md](../vision/index.md)

Iterate over image files in a folder.

## Datasets

### Casper


*Module:* `dense_armor.utility.datasets.casper`

CASPER UR3e joint stream.

Public methods:

- `n_rows()`
- `stream(columns: list, start_h=None, stop_h=None)`

### DriftStream


*Module:* `dense_armor.utility.datasets.drift_stream`

Synthetic binary stream with concept drift.

Public methods:

- `data()`
- `stream()`

### SyntheticArm


*Module:* `dense_armor.utility.datasets.synthetic_arm`

Periodic joint stream from a URDF with an optional fault.

Public methods:

- `nominal_torque(q: ndarray, qd: ndarray, qdd: ndarray)`
- `payload_torque(q: ndarray)`
- `stream()`
- `trajectory(t: float)`

## Protect

### Orca


*Module:* `dense_armor.utility.protect.orca`
*Guide:* [shields/orca.md](../shields/orca.md)

Public methods:

- `protect_and_forward(ai_model_callable: Optional, x_corrupted: ndarray, x_reference=None, use_input_shield=True, use_model_injection=True, use_output_shield=True, use_arbiter=False)`
- `tipi_corruzione_visti(slice_shape: tuple)`

### StreamingArbiter


*Module:* `dense_armor.utility.protect.streaming_arbiter`
*Guide:* [protect/arbiter.md](../protect/arbiter.md)

Bounded-delay streaming version of `arbiter.classify_segments`.

Public methods:

- `flush()`
- `update(x: float)`

### StreamingHealing


*Module:* `dense_armor.utility.protect.streaming_arbiter`
*Guide:* [protect/arbiter.md](../protect/arbiter.md)

One-sample-at-a-time version of `healing.healing_filter`.

Public methods:

- `flush()`
- `update(x: float)`

## Misc

### ABCollatz


*Module:* `dense_armor.utility.misc.collatz`

Discrete-axis functor for binarizing noise.

### Metro


*Module:* `dense_armor.utility.misc.metro`

Public methods:

- `dec(v_new: float, fact: float)`
- `enc(raw_v: float)`

## Learn

### DriftAwareResidualDynamicsLearner


*Module:* `dense_armor.utility.learn.online_dynamics`
*Bases:* ResidualDynamicsLearner
*Guide:* [learn/online_dynamics.md](../learn/online_dynamics.md)

``ResidualDynamicsLearner`` with a per-joint drift trigger on the residual. residual.

Public methods:

- `learn_one(q, qd, qdd, tau_measured)`
- `predict_torque(q, qd, qdd)`

### ResidualDynamicsLearner


*Module:* `dense_armor.utility.learn.online_dynamics`
*Guide:* [learn/online_dynamics.md](../learn/online_dynamics.md)

Online residual torque learner over a nominal ``RigidBodyModel``.

Public methods:

- `learn_one(q, qd, qdd, tau_measured)`
- `predict_torque(q, qd, qdd)`

## Anomaly

### MultiChannelStreamingDeviationDetector


*Module:* `dense_armor.utility.anomaly.streaming`
*Guide:* [anomaly/streaming.md](../anomaly/streaming.md)

N independent ``StreamingDeviationDetector`` instances, one per channel.

Public methods:

- `update(x_vec)`

### StreamingDeviationDetector


*Module:* `dense_armor.utility.anomaly.streaming`
*Guide:* [anomaly/streaming.md](../anomaly/streaming.md)

Zero-latency causal deviation flag, one point at a time.

Public methods:

- `update(x: float)`

## Armatura

### Armatura


*Module:* `dense_armor.armatura`
*Guide:* [shields/armatura.md](../shields/armatura.md)

Wearable shield with universal DYNAMIC CLIP.

Public methods:

- `analizza(serie: ndarray, riferimento=None)`
- `deriva(serie: ndarray)`
- `livello_da_entropia(entropia: float, vocab_size: int)`
- `referto(serie: ndarray, nome='serie', riferimento=None)`
- `referto_json(serie: ndarray, nome='serie', riferimento=None)`
- `set_livello(livello: float)`

## Dynamics

### RigidBodyModel


*Module:* `dense_armor.dynamics.urdf_dynamics`

Real Euler-Lagrange rigid-body dynamics, parsed from a real URDF file.

Public methods:

- `bias_forces(q, qd)`
- `com_positions(q)`
- `forward_dynamics(q, qd, tau)`
- `forward_kinematics(q)`
- `gravity_forces(q)`
- `kinetic_energy(q, qd)`
- `link_jacobian(q, link_name)`
- `link_pose(q, link_name)`
- `link_position(q, link_name)`
- `link_spatial_jacobian(q, link_name)`
- `mass_matrix(q)`
- `potential_energy(q)`
- `total_energy(q, qd)`

