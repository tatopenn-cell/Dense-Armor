# API

A complete list of everything the library offers: every class, what it does and its methods. When a class has a guide page, the link takes you to it.

Complete map of every public class in Dense-Armor: description,
base classes, public methods. When a class has a guide page,
the link leads there for the full explanation and examples.

**392 classes**, 3478 public methods, 204 public modules.

## Classes by subsystem

**Jump to:** [Roles](#roles) &nbsp;·&nbsp; [Metrics](#metrics) &nbsp;·&nbsp; [Evaluate](#evaluate) &nbsp;·&nbsp; [Core](#core) &nbsp;·&nbsp; [Mcp_server](#mcp_server) &nbsp;·&nbsp; [Anomaly](#anomaly) &nbsp;·&nbsp; [Drift](#drift) &nbsp;·&nbsp; [Trees](#trees) &nbsp;·&nbsp; [Naive Bayes](#naive-bayes) &nbsp;·&nbsp; [Distributions](#distributions) &nbsp;·&nbsp; [Nearest neighbours](#nearest-neighbours) &nbsp;·&nbsp; [Multiclass](#multiclass) &nbsp;·&nbsp; [Multi-output](#multi-output) &nbsp;·&nbsp; [Forests](#forests) &nbsp;·&nbsp; [Ensembles](#ensembles) &nbsp;·&nbsp; [Bandits](#bandits) &nbsp;·&nbsp; [Active learning](#active-learning) &nbsp;·&nbsp; [Factorization machines](#factorization-machines) &nbsp;·&nbsp; [Recommenders](#recommenders) &nbsp;·&nbsp; [Rules](#rules) &nbsp;·&nbsp; [Clustering](#clustering) &nbsp;·&nbsp; [Stats](#stats) &nbsp;·&nbsp; [Preprocessing](#preprocessing) &nbsp;·&nbsp; [Compose](#compose) &nbsp;·&nbsp; [Vision](#vision) &nbsp;·&nbsp; [Learn](#learn) &nbsp;·&nbsp; [Datasets](#datasets) &nbsp;·&nbsp; [Protect](#protect) &nbsp;·&nbsp; [Bridges](#bridges) &nbsp;·&nbsp; [Armor](#armor) &nbsp;·&nbsp; [Dynamics](#dynamics) &nbsp;·&nbsp; [Deep learning](#deep-learning) &nbsp;·&nbsp; [Infrastructure](#infrastructure) &nbsp;·&nbsp; [Linear](#linear) &nbsp;·&nbsp; [ML](#ml) &nbsp;·&nbsp; [Optimizers](#optimizers) &nbsp;·&nbsp; [Time series](#time-series)

---

## Roles

### AdaptiveConformalRegressor


*Module:* `dense_armor.roles.uncertainty`
*Bases:* Regressor

Conformal interval with adaptive coverage, wrapping a regressor.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep=True)`
- `learn_one(x: dict, t=None)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `score_one(x: dict, t=None)`
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
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### DriftDetector


*Module:* `dense_armor.roles.drift_detector`
*Bases:* Estimator

Unsupervised drift detector over a scalar stream.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep=True)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `update(x: float, t=None)`
- `update_many(X, t=None)`
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
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep=True)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
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
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep=True)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `update(residual: float)`
- `window_samples(seconds: float)`

### InconsistentVersionWarning


*Module:* `dense_armor.roles.root`
*Bases:* UserWarning

Unpickled a Dense-Armor estimator from a different library version.

### JointLimits


*Module:* `dense_armor.roles.physics`

Position and velocity limits of every joint in a robot.

Public methods:

- `as_guard()`
- `in_bounds(q: Union)`

### ModelWrapper


*Module:* `dense_armor.roles.wrapper`
*Bases:* Estimator

Delegates ``_supervised`` and ``_multiclass`` to the wrapped estimator.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep=True)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
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
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep=True)`
- `init_state()`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `scan(stream: Array)`
- `set_params(params: Any)`
- `state_dict()`
- `step(state: tuple, v: Array)`
- `window_samples(seconds: float)`

### RealtimePipeline


*Module:* `dense_armor.roles.realtime`

A list of estimators that must fit inside a control-loop period.

Public methods:

- `learn_one(sig: Signal)`

### Regressor


*Module:* `dense_armor.roles.regressor`
*Bases:* Estimator

Supervised regressor; scalar or vector target.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### Root


*Module:* `dense_armor.roles.root`

Root of the Dense-Armor online-learning stack.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep=True)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
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
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### Signal


*Module:* `dense_armor.roles.signal`
*Bases:* dict

Dict-like one-sample reading, with vector form and metadata.

Public methods:

- `from_dict(x: dict, units=None, t=None, names=None)`
- `stacked(others: Iterable)`
- `to_dict()`

### SignalBatch


*Module:* `dense_armor.roles.signal`

``n`` samples of ``d`` channels, for the code that works on vectors.

Public methods:

- `row(i: int)`

### Transformer


*Module:* `dense_armor.roles.transformer`
*Bases:* Estimator

Unsupervised transformer: ``transform_one(x) -> dict``.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### UnitCheckedPipeline


*Module:* `dense_armor.roles.physics`
*Bases:* Transformer

A pipeline that checks units at build time.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

## Metrics

### Accuracy


*Module:* `dense_armor.utility.evaluation.metrics.classification`
*Bases:* Metric

Fraction of samples where ``y_pred == y_true``.

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### BalancedAccuracy


*Module:* `dense_armor.utility.evaluation.metrics.classification`
*Bases:* Metric

Mean of the per-class recall.

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### BrierScore


*Module:* `dense_armor.utility.evaluation.metrics.classification`
*Bases:* Metric

Mean squared error of predicted probabilities.

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### CohenKappa


*Module:* `dense_armor.utility.evaluation.metrics.classification`
*Bases:* Metric

Cohen's kappa: agreement corrected for chance.

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### ConfusionMatrix


*Module:* `dense_armor.utility.evaluation.metrics.classification`
*Bases:* Metric

Multiclass confusion matrix.

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### EventMetrics


*Module:* `dense_armor.utility.evaluation.metrics.events`

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


*Module:* `dense_armor.utility.evaluation.metrics.events`

A labelled event interval.

Public methods:

- `contains(t: float)`
- `n_points(dt: float)`

### F1


*Module:* `dense_armor.utility.evaluation.metrics.classification`
*Bases:* FBeta

F1 score: the harmonic mean of precision and recall.

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### FBeta


*Module:* `dense_armor.utility.evaluation.metrics.classification`
*Bases:* _PRFBase

Weighted harmonic mean of precision and recall.

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### GaussianNLL


*Module:* `dense_armor.utility.evaluation.metrics.regression`
*Bases:* Metric

Gaussian negative log-likelihood.

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### IntervalCoverage


*Module:* `dense_armor.utility.evaluation.metrics.regression`
*Bases:* Metric

Fraction of true labels inside a prediction interval.

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### LogLoss


*Module:* `dense_armor.utility.evaluation.metrics.classification`
*Bases:* Metric

Binary log-loss (cross-entropy).

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### MeanAbsoluteError


*Module:* `dense_armor.utility.evaluation.metrics.regression`
*Bases:* Metric

Mean absolute error, one sample at a time.

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### MeanIntervalWidth


*Module:* `dense_armor.utility.evaluation.metrics.regression`
*Bases:* Metric

Mean width of a prediction interval.

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### MeanSquaredError


*Module:* `dense_armor.utility.evaluation.metrics.regression`
*Bases:* Metric

Mean squared error, one sample at a time.

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### Metric


*Module:* `dense_armor.utility.evaluation.metrics.base`

Base class for online metrics.

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### PointAdjustedF1


*Module:* `dense_armor.utility.evaluation.metrics.events`
*Bases:* Metric

Point-adjusted F1 (Xu et al. 2018).

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### Precision


*Module:* `dense_armor.utility.evaluation.metrics.classification`
*Bases:* _PRFBase

Fraction of predicted positives that are correct.

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### R2Score


*Module:* `dense_armor.utility.evaluation.metrics.regression`
*Bases:* Metric

Coefficient of determination, Welford running mean.

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### Recall


*Module:* `dense_armor.utility.evaluation.metrics.classification`
*Bases:* _PRFBase

Fraction of true positives that are detected.

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### Rolling


*Module:* `dense_armor.utility.evaluation.metrics.rolling`
*Bases:* Metric

Rolling window over any metric with ``revert``.

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### RollingAUC


*Module:* `dense_armor.utility.evaluation.metrics.roc_auc`
*Bases:* Metric

Rolling ROC-AUC, exact on the last ``window`` samples.

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

### RootMeanSquaredError


*Module:* `dense_armor.utility.evaluation.metrics.regression`
*Bases:* MeanSquaredError

Square root of the mean squared error.

Public methods:

- `get()`
- `merge(other: Metric)`
- `revert(y_true: Any, y_pred: Any, w=1.0)`
- `update(y_true: Any, y_pred: Any, t=None, w=1.0)`
- `works_with(model: Any)`

## Evaluate

### OnlineModelSelection


*Module:* `dense_armor.utility.evaluation.evaluate.model_selection`

Track several models on a stream and expose the current best.

Public methods:

- `best_model()`
- `current_best()`
- `run(stream: Iterable)`
- `scores()`

## Core

### AIHardwareProfiler


*Module:* `dense_armor.core.noise`
*Guide:* [toolkit/toolkit.md](../toolkit/toolkit.md)

Profiles the host architecture (CPU/GPU/JAX) to set the damping thresholds.

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

### MemoryPressureError


*Module:* `dense_armor.core.memory`
*Bases:* Exception
*Guide:* [toolkit/toolkit.md](../toolkit/toolkit.md)

Raised when system memory (RAM/VRAM) is insufficient.

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

## Anomaly

### ChauvenetScorer


*Module:* `dense_armor.utility.online.anomaly.filters`
*Bases:* AnomalyDetector
*Guide:* [anomaly/streaming.md](../anomaly/streaming.md)

Chauvenet outlier scorer on the causal window.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### HalfSpaceTrees


*Module:* `dense_armor.utility.online.anomaly.hst`
*Bases:* AnomalyDetector

Half-Space Trees for streaming anomaly detection.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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


*Module:* `dense_armor.utility.online.anomaly.filters`
*Bases:* Transformer
*Guide:* [anomaly/streaming.md](../anomaly/streaming.md)

Streaming Hampel filter: replaces outliers with the causal-window median.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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


*Module:* `dense_armor.utility.online.anomaly.filters`
*Bases:* AnomalyDetector
*Guide:* [anomaly/streaming.md](../anomaly/streaming.md)

Hampel outlier scorer on the causal window.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### LODA


*Module:* `dense_armor.utility.online.anomaly.loda`
*Bases:* AnomalyDetector

Lightweight Online Detector of Anomalies.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### OneClassSGD


*Module:* `dense_armor.utility.online.anomaly.ocsvm`
*Bases:* AnomalyDetector

One-Class SVM solved by SGD on random Fourier features.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### OnlineIsolationForest


*Module:* `dense_armor.utility.online.anomaly.oiforest`
*Bases:* AnomalyDetector

Online Isolation Forest.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### OnlineRobustMahalanobis


*Module:* `dense_armor.utility.online.anomaly.mahalanobis`
*Bases:* AnomalyDetector
*Guide:* [anomaly/streaming.md](../anomaly/streaming.md)

Online robust Mahalanobis scorer for multichannel signals.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### SigmaClipScorer


*Module:* `dense_armor.utility.online.anomaly.filters`
*Bases:* AnomalyDetector
*Guide:* [anomaly/streaming.md](../anomaly/streaming.md)

Iterative sigma-clipping outlier scorer on the causal window.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### StreamingDeviationScorer


*Module:* `dense_armor.utility.online.anomaly.deviation`
*Bases:* AnomalyDetector
*Guide:* [anomaly/streaming.md](../anomaly/streaming.md)

`StreamingDeviationDetector` as an anomaly detector compatible with an existing online-learning library.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### TukeyScorer


*Module:* `dense_armor.utility.online.anomaly.filters`
*Bases:* AnomalyDetector
*Guide:* [anomaly/streaming.md](../anomaly/streaming.md)

Tukey-fences outlier scorer on the causal window.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### MultiChannelStreamingDeviationDetector


*Module:* `dense_armor.utility.online.anomaly.streaming`
*Guide:* [anomaly/streaming.md](../anomaly/streaming.md)

N independent ``StreamingDeviationDetector`` instances, one per channel.

Public methods:

- `update(x_vec)`

### StreamingDeviationDetector


*Module:* `dense_armor.utility.online.anomaly.streaming`
*Guide:* [anomaly/streaming.md](../anomaly/streaming.md)

Causal per-sample deviation flag, one point at a time.

Public methods:

- `update(x: float)`

### LocalOutlierFactor


*Module:* `dense_armor.utility.online.anomaly.lof`
*Bases:* AnomalyDetector
*Guide:* [anomaly/lof.md](../anomaly/lof.md)

Local Outlier Factor over a sliding window of the last samples.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, t=None)`
- `learn_one(x: dict, t: float | None = None)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `score_many(X, t=None)`
- `score_one(x: dict, t: float | None = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

## Drift

### ADWIN


*Module:* `dense_armor.utility.online.drift.adwin`
*Bases:* DriftDetector
*Guide:* [drift/detectors.md](../drift/detectors.md)

Adaptive Windowing drift detector.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep=True)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `update(x: float, t=None)`
- `update_many(X, t=None)`
- `window_samples(seconds: float)`

### BayesianOnlineChangepoint


*Module:* `dense_armor.utility.online.drift.bocpd`
*Bases:* DriftDetector

Online changepoint detection with a Normal-Inverse-Gamma model.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `set_params(**params: Any)`
- `state_dict()`
- `update(x: 'float', t: 'float | None' = None)`
- `update_many(X, t=None)`
- `window_samples(seconds: float)`

### CUSUMDriftDetector


*Module:* `dense_armor.utility.online.drift.detector`
*Bases:* DriftDetector
*Guide:* [drift/detectors.md](../drift/detectors.md)

CUSUM drift detector matching the batch `cusum_detector` behaviour.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `expected_detection_delay(shift: float)`
- `expected_false_alarm_run()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep=True)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `update(x: float, t=None)`
- `update_many(X, t=None)`
- `window_samples(seconds: float)`

### KSWIN


*Module:* `dense_armor.utility.online.drift.kswin`
*Bases:* DriftDetector
*Guide:* [drift/detectors.md](../drift/detectors.md)

Kolmogorov-Smirnov Windowing drift detector.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep=True)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `update(x: float, t=None)`
- `update_many(X, t=None)`
- `window_samples(seconds: float)`

### PageHinkley


*Module:* `dense_armor.utility.online.drift.page_hinkley`
*Bases:* DriftDetector
*Guide:* [drift/detectors.md](../drift/detectors.md)

Page-Hinkley test for a change in the mean of a scalar stream.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep=True)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `update(x: float, t=None)`
- `update_many(X, t=None)`
- `window_samples(seconds: float)`

## Trees

### BatchDecisionTreeClassifier


*Module:* `dense_armor.utility.online.tree.batch`
*Bases:* Classifier

Batch CART classifier.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X: 'Any', y: 'Any' = None, t: 'Any' = None, epochs: 'int' = 1, shuffle: 'bool' = False, seed: 'int | None' = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: 'dict', y: 'Any', t: 'float | None' = None)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `predict_many(X, t=None)`
- `predict_one(x: 'dict', t: 'float | None' = None)`
- `predict_proba_many(X, t=None)`
- `predict_proba_one(x: 'dict', t: 'float | None' = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### FIMTDDRegressor


*Module:* `dense_armor.utility.online.tree.fimt`
*Bases:* Regressor

Fast Incremental Model Tree with Drift Detection (FIMT-DD).

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: dict, y: Any, t: float | None = None)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `predict_many(X, t=None, return_std=False)`
- `predict_one(x: dict, t: float | None = None, return_std: bool = False)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### HoeffdingAdaptiveTreeClassifier


*Module:* `dense_armor.utility.online.tree.adaptive`
*Bases:* HoeffdingTreeClassifier
*Guide:* [tree/index.md](../tree/index.md)

Hoeffding Adaptive Tree classifier.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `explain_one(x: Any, t=None)`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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


*Module:* `dense_armor.utility.online.tree.efdt`
*Bases:* HoeffdingTreeClassifier
*Guide:* [tree/index.md](../tree/index.md)

Hoeffding Anytime Tree classifier.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `explain_one(x: Any, t=None)`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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


*Module:* `dense_armor.utility.online.tree.hoeffding`
*Bases:* Classifier
*Guide:* [tree/index.md](../tree/index.md)

Hoeffding tree for incremental classification.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `explain_one(x: Any, t=None)`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### MondrianForestClassifier


*Module:* `dense_armor.utility.online.tree.mondrian`
*Bases:* Classifier, _MondrianBase
*Guide:* [tree/index.md](../tree/index.md)

Online Mondrian forest for classification.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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


*Module:* `dense_armor.utility.online.tree.mondrian`
*Bases:* Regressor, _MondrianBase
*Guide:* [tree/index.md](../tree/index.md)

Online Mondrian forest for regression.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### SGTClassifier


*Module:* `dense_armor.utility.online.tree.sgt`
*Bases:* Classifier, _SGTBase
*Guide:* [tree/index.md](../tree/index.md)

Stochastic gradient tree for binary classification.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `explain_one(x: Any, t=None)`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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


*Module:* `dense_armor.utility.online.tree.sgt`
*Bases:* Regressor, _SGTBase
*Guide:* [tree/index.md](../tree/index.md)

Stochastic gradient tree for regression.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `explain_one(x: Any, t=None)`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

## Naive Bayes

### MultinomialNB


*Module:* `dense_armor.utility.online.bayes.multinomial`
*Bases:* Classifier
*Guide:* [bayes/index.md](../bayes/index.md)

Multinomial naive Bayes with Laplace smoothing.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: dict, y: Any, t: float | None = None)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `predict_many(X, t=None)`
- `predict_one(x: dict, t: float | None = None)`
- `predict_proba_many(X, t=None)`
- `predict_proba_one(x: dict, t: float | None = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### BernoulliNB


*Module:* `dense_armor.utility.online.bayes.bernoulli`
*Bases:* Classifier
*Guide:* [bayes/index.md](../bayes/index.md)

Bernoulli naive Bayes with Laplace smoothing.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: dict, y: Any, t: float | None = None)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `predict_many(X, t=None)`
- `predict_one(x: dict, t: float | None = None)`
- `predict_proba_many(X, t=None)`
- `predict_proba_one(x: dict, t: float | None = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### ComplementNB


*Module:* `dense_armor.utility.online.bayes.complement`
*Bases:* Classifier
*Guide:* [bayes/index.md](../bayes/index.md)

Complement naive Bayes with Laplace smoothing.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: dict, y: Any, t: float | None = None)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `predict_many(X, t=None)`
- `predict_one(x: dict, t: float | None = None)`
- `predict_proba_many(X, t=None)`
- `predict_proba_one(x: dict, t: float | None = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### CategoricalNB


*Module:* `dense_armor.utility.online.bayes.categorical`
*Bases:* Classifier
*Guide:* [bayes/index.md](../bayes/index.md)

Categorical naive Bayes with Laplace smoothing.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: dict, y: Any, t: float | None = None)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `predict_many(X, t=None)`
- `predict_one(x: dict, t: float | None = None)`
- `predict_proba_many(X, t=None)`
- `predict_proba_one(x: dict, t: float | None = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

## Distributions

### Beta


*Module:* `dense_armor.utility.online.proba.beta`
*Bases:* object
*Guide:* [proba/index.md](../proba/index.md)

Beta distribution with online update and revert.

Public methods:

- `init_state()`
- `mean()`
- `mode()`
- `pdf(x: float)`
- `revert(x: Any, w: float = 1.0)`
- `sample(rng: numpy.random._generator.Generator)`
- `step(state: tuple, v: object)`
- `update(x: Any, w: float = 1.0)`
- `variance()`

### Gaussian


*Module:* `dense_armor.utility.online.proba.gaussian`
*Bases:* object
*Guide:* [proba/index.md](../proba/index.md)

Univariate Gaussian updated online by Welford's algorithm.

Public methods:

- `cdf(x: float)`
- `init_state()`
- `mean()`
- `mode()`
- `n()`
- `pdf(x: float)`
- `revert(x: float, w: float = 1.0)`
- `sample(rng: numpy.random._generator.Generator)`
- `step(state: tuple, v: object)`
- `update(x: float, w: float = 1.0)`
- `variance()`

### Multinomial


*Module:* `dense_armor.utility.online.proba.multinomial`
*Bases:* object
*Guide:* [proba/index.md](../proba/index.md)

Dirichlet count model over K categories.

Public methods:

- `init_state()`
- `mean()`
- `mode()`
- `pmf(k: Any)`
- `revert(k: Any, w: float = 1.0)`
- `sample(rng: numpy.random._generator.Generator)`
- `step(state, v: object)`
- `update(k: Any, w: float = 1.0)`

## Nearest neighbours

### KNNClassifier


*Module:* `dense_armor.utility.online.neighbors.knn`
*Bases:* Classifier
*Guide:* [neighbors/index.md](../neighbors/index.md)

k-nearest-neighbour classifier on a sliding window.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: dict, y: Any, t: float | None = None)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `predict_many(X, t=None)`
- `predict_one(x: dict, t: float | None = None)`
- `predict_proba_many(X, t=None)`
- `predict_proba_one(x: dict, t: float | None = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### KNNRegressor


*Module:* `dense_armor.utility.online.neighbors.knn`
*Bases:* Regressor
*Guide:* [neighbors/index.md](../neighbors/index.md)

k-nearest-neighbour regressor on a sliding window.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: dict, y: Any, t: float | None = None)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `predict_many(X, t=None, return_std=False)`
- `predict_one(x: dict, t: float | None = None, return_std: bool = False)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### LazySearch


*Module:* `dense_armor.utility.online.neighbors.knn`
*Bases:* object
*Guide:* [neighbors/index.md](../neighbors/index.md)

Sliding window of recent samples with an exact k-NN scan.

Public methods:

- `add(x: dict, y: Any)`
- `forget()`
- `neighbours(x: dict)`

## Multiclass

### OneVsRestClassifier


*Module:* `dense_armor.utility.online.multiclass.multiclass`
*Guide:* [multiclass/index.md](../multiclass/index.md)

One-vs-all multiclass wrapper.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: dict, y: Any, t: float | None = None)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `predict_many(X, t=None)`
- `predict_one(x: dict, t: float | None = None)`
- `predict_proba_many(X, t=None)`
- `predict_proba_one(x: dict, t: float | None = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### OneVsOneClassifier


*Module:* `dense_armor.utility.online.multiclass.multiclass`
*Guide:* [multiclass/index.md](../multiclass/index.md)

Round-robin (pairwise) multiclass wrapper.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: dict, y: Any, t: float | None = None)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `predict_many(X, t=None)`
- `predict_one(x: dict, t: float | None = None)`
- `predict_proba_many(X, t=None)`
- `predict_proba_one(x: dict, t: float | None = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### OutputCodeClassifier


*Module:* `dense_armor.utility.online.multiclass.multiclass`
*Guide:* [multiclass/index.md](../multiclass/index.md)

Error-correcting output code wrapper.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: dict, y: Any, t: float | None = None)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `predict_many(X, t=None)`
- `predict_one(x: dict, t: float | None = None)`
- `predict_proba_many(X, t=None)`
- `predict_proba_one(x: dict, t: float | None = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

## Multi-output

### PerOutputClassifier


*Module:* `dense_armor.utility.online.multioutput.multioutput`
*Bases:* Classifier
*Guide:* [multioutput/index.md](../multioutput/index.md)

One binary classifier per target, learned independently.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: dict, y: dict, t: float | None = None)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `predict_many(X, t=None)`
- `predict_one(x: dict, t: float | None = None)`
- `predict_proba_many(X, t=None)`
- `predict_proba_one(x: dict, t: float | None = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### PerOutputRegressor


*Module:* `dense_armor.utility.online.multioutput.multioutput`
*Bases:* Regressor
*Guide:* [multioutput/index.md](../multioutput/index.md)

One regressor per target, learned independently.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: dict, y: dict, t: float | None = None)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `predict_many(X, t=None, return_std=False)`
- `predict_one(x: dict, t: float | None = None, return_std: bool = False)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### ClassifierChain


*Module:* `dense_armor.utility.online.multioutput.multioutput`
*Bases:* Classifier
*Guide:* [multioutput/index.md](../multioutput/index.md)

Classifier chain: chained binary classifiers.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: dict, y: dict, t: float | None = None)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `predict_many(X, t=None)`
- `predict_one(x: dict, t: float | None = None)`
- `predict_proba_many(X, t=None)`
- `predict_proba_one(x: dict, t: float | None = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### RegressorChain


*Module:* `dense_armor.utility.online.multioutput.multioutput`
*Bases:* Regressor
*Guide:* [multioutput/index.md](../multioutput/index.md)

Regressor chain: chained regressors.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: dict, y: dict, t: float | None = None)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `predict_many(X, t=None, return_std=False)`
- `predict_one(x: dict, t: float | None = None, return_std: bool = False)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### ProbabilisticClassifierChain


*Module:* `dense_armor.utility.online.multioutput.multioutput`
*Bases:* Classifier
*Guide:* [multioutput/index.md](../multioutput/index.md)

Probabilistic classifier chain (Bayes-optimal for 0/1 subset loss).

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `joint_proba_one(x: dict, t: float | None = None)`
- `learn_many(X, y, t=None)`
- `learn_one(x: dict, y: dict, t: float | None = None)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `predict_many(X, t=None)`
- `predict_one(x: dict, t: float | None = None)`
- `predict_proba_many(X, t=None)`
- `predict_proba_one(x: dict, t: float | None = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

## Forests

### AdaptiveRandomForestClassifier


*Module:* `dense_armor.utility.online.forest.arf`
*Bases:* Classifier
*Guide:* [forest/index.md](../forest/index.md)

Adaptive Random Forest of Gomes et al. (2017).

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: Any, y: Any, t: float | None = None)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `predict_many(X, t=None)`
- `predict_one(x: Any, t: float | None = None)`
- `predict_proba_many(X, t=None)`
- `predict_proba_one(x: Any, t: float | None = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### AggregatedMondrianForestClassifier


*Module:* `dense_armor.utility.online.forest.amf`
*Bases:* Classifier
*Guide:* [forest/index.md](../forest/index.md)

Aggregated Mondrian forest classifier (Mourtada et al.).

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: Any, y: Any, t: float | None = None)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `predict_many(X, t=None)`
- `predict_one(x: Any, t: float | None = None)`
- `predict_proba_many(X, t=None)`
- `predict_proba_one(x: Any, t: float | None = None)`
- `predict_unaggregated_one(x: Any)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### AggregatedMondrianForestRegressor


*Module:* `dense_armor.utility.online.forest.amf`
*Bases:* Regressor
*Guide:* [forest/index.md](../forest/index.md)

Aggregated Mondrian forest regressor (Mourtada et al.).

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: Any, y: Any, t: float | None = None)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `predict_many(X, t=None, return_std=False)`
- `predict_one(x: Any, t: float | None = None, return_std: bool = False)`
- `predict_unaggregated_one(x: Any)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### SubspaceHoeffdingTree


*Module:* `dense_armor.utility.online.forest.arf`
*Bases:* HoeffdingTreeClassifier
*Guide:* [forest/index.md](../forest/index.md)

Hoeffding tree whose split attempts see a random subset of features.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `explain_one(x: Any, t: float | None = None)`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: Any, y: Any, t: float | None = None)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `predict_many(X, t=None)`
- `predict_one(x: Any, t: float | None = None)`
- `predict_proba_many(X, t=None)`
- `predict_proba_one(x: Any, t: float | None = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`


### OnlineExtraTreesRegressor


*Module:* `dense_armor.utility.online.forest.oxt`
*Bases:* Regressor
*Guide:* [forest/index.md](../forest/index.md)

Online Extra Trees regressor (Mastelini, Nakano, Vens and de Carvalho).

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: Any, y: Any, t: float | None = None)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `predict_many(X, t=None, return_std=False)`
- `predict_one(x: Any, t: float | None = None, return_std: bool = False)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

## Ensembles

### ExponentiallyWeightedAverageRegressor


*Module:* `dense_armor.utility.online.ensemble.ewa`
*Bases:* Regressor
*Guide:* [ensemble/index.md](../ensemble/index.md)

Weighted average of regressors with the Hedge rule (Freund and Schapire 1997).

Public methods:

- `bound()`
- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x, y, t: float | None = None)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `predict_many(X, t=None, return_std=False)`
- `predict_one(x, t: float | None = None, return_std: bool = False)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### OnlineBaggingClassifier


*Module:* `dense_armor.utility.online.ensemble.bagging`
*Bases:* Classifier
*Guide:* [ensemble/index.md](../ensemble/index.md)

Bagging of online classifiers with a Poisson(1) resample (Oza and Russell 2001).

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x, y, t: float | None = None)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `predict_many(X, t=None)`
- `predict_one(x, t: float | None = None)`
- `predict_proba_many(X, t=None)`
- `predict_proba_one(x, t: float | None = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### OnlineBoostingClassifier


*Module:* `dense_armor.utility.online.ensemble.boosting`
*Bases:* Classifier
*Guide:* [ensemble/index.md](../ensemble/index.md)

Online AdaBoost with a Poisson resample and sample weights (Oza and Russell 2001, Section 3).

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x, y, t: float | None = None)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `predict_many(X, t=None)`
- `predict_one(x, t: float | None = None)`
- `predict_proba_many(X, t=None)`
- `predict_proba_one(x, t: float | None = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### StackingClassifier


*Module:* `dense_armor.utility.online.ensemble.stacking`
*Bases:* Classifier
*Guide:* [ensemble/index.md](../ensemble/index.md)

Stacking of a meta-model on top of a list of base classifiers.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x, y, t: float | None = None)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `predict_many(X, t=None)`
- `predict_one(x, t: float | None = None)`
- `predict_proba_many(X, t=None)`
- `predict_proba_one(x, t: float | None = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### VotingClassifier


*Module:* `dense_armor.utility.online.ensemble.voting`
*Bases:* Classifier
*Guide:* [ensemble/index.md](../ensemble/index.md)

Hard vote or mean of probabilities over a fixed list of classifiers.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x, y, t: float | None = None)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `predict_many(X, t=None)`
- `predict_one(x, t: float | None = None)`
- `predict_proba_many(X, t=None)`
- `predict_proba_one(x, t: float | None = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

## Bandits

### BanditPolicy


*Module:* `dense_armor.utility.online.bandit.base`
*Bases:* Root
*Guide:* [bandit/index.md](../bandit/index.md)

Base class for bandit policies.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `pull(context: dict | None = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `update(arm, reward: float, context: dict | None = None)`
- `window_samples(seconds: float)`

### BayesUCB


*Module:* `dense_armor.utility.online.bandit.bayes_ucb`
*Bases:* BanditPolicy
*Guide:* [bandit/index.md](../bandit/index.md)

Bayes-UCB policy.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `pull(context: dict | None = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `update(arm, reward: float, context: dict | None = None)`
- `window_samples(seconds: float)`

### EpsilonGreedy


*Module:* `dense_armor.utility.online.bandit.ucb`
*Bases:* BanditPolicy
*Guide:* [bandit/index.md](../bandit/index.md)

Epsilon-greedy policy.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `pull(context: dict | None = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `update(arm, reward: float, context: dict | None = None)`
- `window_samples(seconds: float)`

### Exp3


*Module:* `dense_armor.utility.online.bandit.exp3`
*Bases:* BanditPolicy
*Guide:* [bandit/index.md](../bandit/index.md)

Exp3 policy for the non-stochastic multi-armed bandit.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `pull(context: dict | None = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `update(arm, reward: float, context: dict | None = None)`
- `window_samples(seconds: float)`

### LinUCBDisjoint


*Module:* `dense_armor.utility.online.bandit.linucb`
*Bases:* Root
*Guide:* [bandit/index.md](../bandit/index.md)

LinUCB with disjoint linear models.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `pull(context: dict | None = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `update(arm, reward: float, context: dict | None = None)`
- `window_samples(seconds: float)`

### RobotKpSelector


*Module:* `dense_armor.utility.online.bandit.robot`
*Bases:* Root
*Guide:* [bandit/index.md](../bandit/index.md)

Choose ``kp`` among a set of values by UCB1 on a robot stream.

Public methods:

- `best_gain()`
- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `run_round()`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### ThompsonSampling


*Module:* `dense_armor.utility.online.bandit.ts`
*Bases:* BanditPolicy
*Guide:* [bandit/index.md](../bandit/index.md)

Thompson sampling for a Bernoulli or a Gaussian bandit.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `pull(context: dict | None = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `update(arm, reward: float, context: dict | None = None)`
- `window_samples(seconds: float)`

### UCB1


*Module:* `dense_armor.utility.online.bandit.ucb`
*Bases:* BanditPolicy
*Guide:* [bandit/index.md](../bandit/index.md)

UCB1 policy.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `pull(context: dict | None = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `update(arm, reward: float, context: dict | None = None)`
- `window_samples(seconds: float)`

### UCB1Tuned


*Module:* `dense_armor.utility.online.bandit.ucb`
*Bases:* UCB1
*Guide:* [bandit/index.md](../bandit/index.md)

UCB1-Tuned policy.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `pull(context: dict | None = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `update(arm, reward: float, context: dict | None = None)`
- `window_samples(seconds: float)`

## Active learning

### ActiveStrategy


*Module:* `dense_armor.utility.online.active.base`
*Bases:* Root
*Guide:* [active/index.md](../active/index.md)

Base class for active learning strategies on a stream.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `set_params(**params: Any)`
- `should_label(x: dict, model)`
- `state_dict()`
- `window_samples(seconds: float)`

### FixedUncertainty


*Module:* `dense_armor.utility.online.active.strategies`
*Bases:* ActiveStrategy
*Guide:* [active/index.md](../active/index.md)

Fixed-threshold uncertainty strategy.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `set_params(**params: Any)`
- `should_label(x: dict, model)`
- `state_dict()`
- `window_samples(seconds: float)`

### RandomStrategy


*Module:* `dense_armor.utility.online.active.strategies`
*Bases:* ActiveStrategy
*Guide:* [active/index.md](../active/index.md)

Random strategy: label with probability ``B``.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `set_params(**params: Any)`
- `should_label(x: dict, model)`
- `state_dict()`
- `window_samples(seconds: float)`

### VariableRandomizedUncertainty


*Module:* `dense_armor.utility.online.active.strategies`
*Bases:* VariableUncertainty
*Guide:* [active/index.md](../active/index.md)

Variable-threshold uncertainty with a randomised threshold.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `set_params(**params: Any)`
- `should_label(x: dict, model)`
- `state_dict()`
- `window_samples(seconds: float)`

### VariableUncertainty


*Module:* `dense_armor.utility.online.active.strategies`
*Bases:* ActiveStrategy
*Guide:* [active/index.md](../active/index.md)

Variable-threshold uncertainty strategy.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `set_params(**params: Any)`
- `should_label(x: dict, model)`
- `state_dict()`
- `window_samples(seconds: float)`

## Factorization machines

### FFMClassifier


*Module:* `dense_armor.utility.online.facto.ffm`
*Bases:* _FFMBase, Classifier
*Guide:* [facto/index.md](../facto/index.md)

Field-aware factorization machine for binary classification.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: dict, y, t: float | None = None)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `predict_many(X, t=None)`
- `predict_one(x: dict, t: float | None = None)`
- `predict_proba_many(X, t=None)`
- `predict_proba_one(x: dict, t: float | None = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### FFMRegressor


*Module:* `dense_armor.utility.online.facto.ffm`
*Bases:* _FFMBase, Regressor
*Guide:* [facto/index.md](../facto/index.md)

Field-aware factorization machine with squared loss.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: dict, y, t: float | None = None)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `predict_many(X, t=None, return_std=False)`
- `predict_one(x: dict, t: float | None = None, return_std: bool = False)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### FMClassifier


*Module:* `dense_armor.utility.online.facto.fm`
*Bases:* _FMBase, Classifier
*Guide:* [facto/index.md](../facto/index.md)

Factorization machine of degree 2, for binary classification.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: dict, y, t: float | None = None)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `predict_many(X, t=None)`
- `predict_one(x: dict, t: float | None = None)`
- `predict_proba_many(X, t=None)`
- `predict_proba_one(x: dict, t: float | None = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### FMRegressor


*Module:* `dense_armor.utility.online.facto.fm`
*Bases:* _FMBase, Regressor
*Guide:* [facto/index.md](../facto/index.md)

Factorization machine of degree 2, for regression.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: dict, y, t: float | None = None)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `predict_many(X, t=None, return_std=False)`
- `predict_one(x: dict, t: float | None = None, return_std: bool = False)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### FwFMClassifier


*Module:* `dense_armor.utility.online.facto.fwfm`
*Bases:* _FwFMBase, Classifier
*Guide:* [facto/index.md](../facto/index.md)

Field-weighted factorization machine for binary classification.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: dict, y, t: float | None = None)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `predict_many(X, t=None)`
- `predict_one(x: dict, t: float | None = None)`
- `predict_proba_many(X, t=None)`
- `predict_proba_one(x: dict, t: float | None = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### FwFMRegressor


*Module:* `dense_armor.utility.online.facto.fwfm`
*Bases:* _FwFMBase, Regressor
*Guide:* [facto/index.md](../facto/index.md)

Field-weighted factorization machine with squared loss.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: dict, y, t: float | None = None)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `predict_many(X, t=None, return_std=False)`
- `predict_one(x: dict, t: float | None = None, return_std: bool = False)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### HOFMClassifier


*Module:* `dense_armor.utility.online.facto.hofm`
*Bases:* _HOFMBase, Classifier
*Guide:* [facto/index.md](../facto/index.md)

Higher-order factorization machine for binary classification.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: dict, y, t: float | None = None)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `predict_many(X, t=None)`
- `predict_one(x: dict, t: float | None = None)`
- `predict_proba_many(X, t=None)`
- `predict_proba_one(x: dict, t: float | None = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### HOFMRegressor


*Module:* `dense_armor.utility.online.facto.hofm`
*Bases:* _HOFMBase, Regressor
*Guide:* [facto/index.md](../facto/index.md)

Higher-order factorization machine for regression.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: dict, y, t: float | None = None)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `predict_many(X, t=None, return_std=False)`
- `predict_one(x: dict, t: float | None = None, return_std: bool = False)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

## Recommenders

### Baseline


*Module:* `dense_armor.utility.online.reco.reco`
*Bases:* _RecoBase
*Guide:* [reco/index.md](../reco/index.md)

Global mean plus per-user and per-item biases.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_one(user: collections.abc.Hashable, item: collections.abc.Hashable, rating)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `predict_one(user: collections.abc.Hashable, item: collections.abc.Hashable)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### BiasedMF


*Module:* `dense_armor.utility.online.reco.reco`
*Bases:* _RecoBase
*Guide:* [reco/index.md](../reco/index.md)

Global mean, biases and latent factors, with SGD.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_one(user: collections.abc.Hashable, item: collections.abc.Hashable, rating)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `predict_one(user: collections.abc.Hashable, item: collections.abc.Hashable)`
- `rank(user: collections.abc.Hashable, items: collections.abc.Sequence[collections.abc.Hashable], top_k: int | None = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### FunkMF


*Module:* `dense_armor.utility.online.reco.reco`
*Bases:* _RecoBase
*Guide:* [reco/index.md](../reco/index.md)

Plain regularized MF with SGD, no user or item biases.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_one(user: collections.abc.Hashable, item: collections.abc.Hashable, rating)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `predict_one(user: collections.abc.Hashable, item: collections.abc.Hashable)`
- `rank(user: collections.abc.Hashable, items: collections.abc.Sequence[collections.abc.Hashable], top_k: int | None = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

## Rules

### AMRulesRegressor


*Module:* `dense_armor.utility.online.rules.amrules`
*Bases:* Regressor
*Guide:* [rules/index.md](../rules/index.md)

Adaptive Model Rules regressor (Duarte, Gama and Bifet).

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `explain_one(x: Any)`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: Any, y: Any, t: float | None = None)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `predict_many(X, t=None, return_std=False)`
- `predict_one(x: Any, t: float | None = None, return_std: bool = False)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

## Clustering

### DenStream


*Module:* `dense_armor.utility.online.cluster.denstream`
*Bases:* Transformer
*Guide:* [cluster/index.md](../cluster/index.md)

Density-based stream clustering with a damped window.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### OnlineKMeans


*Module:* `dense_armor.utility.online.cluster.kmeans`
*Bases:* Transformer
*Guide:* [cluster/index.md](../cluster/index.md)

Sequential k-means, one sample at a time.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

## Stats

### Autocorrelation


*Module:* `dense_armor.utility.signals.stats.dependence`
*Bases:* Transformer

Running autocorrelation of a single channel at a fixed lag.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### BloomFilter


*Module:* `dense_armor.utility.signals.stats.sketches`
*Bases:* Transformer

Bloom membership filter (Bloom 1970).

Public methods:

- `check(item: Any)`
- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### CountMinSketch


*Module:* `dense_armor.utility.signals.stats.sketches`
*Bases:* Transformer

Count-Min frequency sketch (Cormode & Muthukrishnan 2005).

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `estimate(item: Any)`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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


*Module:* `dense_armor.utility.signals.stats.quantiles`
*Bases:* Transformer

Relative-error quantile sketch (Masson, Rim, Lee 2019).

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### EWStats


*Module:* `dense_armor.utility.signals.stats.moments`
*Bases:* Transformer

Exponentially weighted mean and variance, one sample at a time.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### HyperLogLog


*Module:* `dense_armor.utility.signals.stats.sketches`
*Bases:* Transformer

HyperLogLog distinct-count sketch (Flajolet et al. 2007).

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `count()`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### OnlineGaussianMixture


*Module:* `dense_armor.utility.signals.stats.mixture`
*Bases:* Transformer

Online Gaussian mixture with diagonal covariance.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, t=None)`
- `learn_one(x: dict, y: Any = None, t: float | None = None)`
- `load_state_dict(state: dict[str, Any])`
- `merge(other: 'OnlineGaussianMixture')`
- `mutate(new_attrs: dict[str, Any])`
- `score_one(x: dict)`
- `set_params(**params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: dict, t: float | None = None)`
- `update(x: dict, y: Any = None, t: float | None = None)`
- `window_samples(seconds: float)`

### OnlineShrinkageCovariance


*Module:* `dense_armor.utility.signals.stats.shrinkage`
*Bases:* Transformer

Online covariance with linear shrinkage toward ``(1/p) I``.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, t=None)`
- `learn_one(x: 'dict', y=None, t: 'float | None' = None)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `sample_covariance()`
- `set_params(**params: Any)`
- `shrinkage()`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: 'dict', t: 'float | None' = None)`
- `window_samples(seconds: float)`

### RollingCorrelation


*Module:* `dense_armor.utility.signals.stats.dependence`
*Bases:* Transformer

Pearson correlation of two channels over a bounded window.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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


*Module:* `dense_armor.utility.signals.stats.dependence`
*Bases:* Transformer

Sample covariance of two channels over a bounded window.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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


*Module:* `dense_armor.utility.signals.stats.robust`
*Bases:* _RollingWindow

Interquartile range of the last ``window`` samples.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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


*Module:* `dense_armor.utility.signals.stats.robust`
*Bases:* _RollingWindow

Median absolute deviation of the last ``window`` samples, scaled.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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


*Module:* `dense_armor.utility.signals.stats.robust`
*Bases:* _RollingWindow

Median of the last ``window`` samples.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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


*Module:* `dense_armor.utility.signals.stats.robust`
*Bases:* Transformer

One :class:`RollingMedian` per channel, one update call.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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


*Module:* `dense_armor.utility.signals.stats.robust`
*Bases:* _RollingWindow

Arbitrary quantile of the last ``window`` samples.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### RunningCorrelation


*Module:* `dense_armor.utility.signals.stats.dependence`
*Bases:* Transformer

Incremental Pearson correlation between two channels.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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


*Module:* `dense_armor.utility.signals.stats.dependence`
*Bases:* Transformer

Incremental covariance of two channels.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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


*Module:* `dense_armor.utility.signals.stats.moments`
*Bases:* Transformer

Incremental mean, variance, skewness, kurtosis for one feature.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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


*Module:* `dense_armor.utility.signals.stats.moments`
*Bases:* Transformer

One ``RunningMoments`` per channel, updated in a single call.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### SpaceSaving


*Module:* `dense_armor.utility.signals.stats.sketches`
*Bases:* Transformer

Space-Saving heavy hitters (Metwally et al. 2005).

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `estimate(item: Any)`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### SparseCosineIndex


*Module:* `dense_armor.utility.signals.stats.similarity`

A sparse index of documents for cosine ranking, built once.

Public methods:

- `from_csr(M, vocabulary: 'dict')`
- `query(q: 'dict')`
- `query_many(queries: 'list[dict]')`
- `query_many_sparse(Q: 'sparse.spmatrix', normalized: 'bool' = False)`
- `top_k_threshold(q: 'dict', k: 'int' = 5)`

### StudentT


*Module:* `dense_armor.utility.signals.stats.student_t`
*Bases:* Transformer

Online Student-t location-scale estimator with fixed ``nu``.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, t=None)`
- `learn_one(x: dict, y: Any = None, t: float | None = None)`
- `load_state_dict(state: dict[str, Any])`
- `merge(other: 'StudentT')`
- `mutate(new_attrs: dict[str, Any])`
- `score_one(x: dict)`
- `set_params(**params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: dict, t: float | None = None)`
- `update(x: dict, y: Any = None, t: float | None = None)`
- `window_samples(seconds: float)`

### TDigest


*Module:* `dense_armor.utility.signals.stats.quantiles`
*Bases:* Transformer

Clustered sketch with small clusters near the tails (Dunning 2019).

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### SDFT


*Module:* `dense_armor.utility.signals.stats.sdft`
*Bases:* Root
*Guide:* [reco/index.md](../reco/index.md)

Sliding DFT: the N-point DFT of the last N samples, O(N) per update.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `set_params(**params: Any)`
- `spectrum()`
- `state_dict()`
- `update(v: float)`
- `update_many(values)`
- `window_samples(seconds: float)`

### Skyline


*Module:* `dense_armor.utility.signals.stats.skyline`
*Bases:* Root
*Guide:* [reco/index.md](../reco/index.md)

Skyline of a sliding window over a stream of points.

Public methods:

- `clone(new_params: dict[str, typing.Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `load_state_dict(state: dict[str, typing.Any])`
- `mutate(new_attrs: dict[str, typing.Any])`
- `set_params(**params: Any)`
- `skyline()`
- `skyline_points()`
- `state_dict()`
- `update(point)`
- `update_many(points)`
- `window_samples(seconds: float)`

## Preprocessing

### BagOfWords


*Module:* `dense_armor.utility.signals.preprocessing.text`
*Bases:* Transformer
*Guide:* [preprocessing/index.md](../preprocessing/index.md)

Term counts of one document, from a token list.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### EWScaler


*Module:* `dense_armor.utility.signals.preprocessing.scale`
*Bases:* Transformer
*Guide:* [preprocessing/index.md](../preprocessing/index.md)

Per-feature standardization with exponentially weighted stats.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### FeatureHasher


*Module:* `dense_armor.utility.signals.preprocessing.text`
*Bases:* Transformer
*Guide:* [preprocessing/index.md](../preprocessing/index.md)

Signed feature hashing (Weinberger et al. 2009, Theorem 3).

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### JointDerivatives


*Module:* `dense_armor.utility.signals.preprocessing.joints`
*Bases:* Transformer
*Guide:* [preprocessing/index.md](../preprocessing/index.md)

Velocity, acceleration and jerk from joint positions.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### JointPower


*Module:* `dense_armor.utility.signals.preprocessing.joints`
*Bases:* Transformer
*Guide:* [preprocessing/index.md](../preprocessing/index.md)

Mechanical power ``tau * qd`` per joint and its total.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### MinMaxScaler


*Module:* `dense_armor.utility.signals.preprocessing.scale`
*Bases:* Transformer
*Guide:* [preprocessing/index.md](../preprocessing/index.md)

Rescale each feature to ``[0, 1]`` with running min and max.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### QueueResampler


*Module:* `dense_armor.utility.signals.preprocessing.imbalance`
*Bases:* Classifier
*Guide:* [preprocessing/index.md](../preprocessing/index.md)

Resampling wrapper for imbalanced online streams.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### RobustScaler


*Module:* `dense_armor.utility.signals.preprocessing.scale`
*Bases:* Transformer
*Guide:* [preprocessing/index.md](../preprocessing/index.md)

Robust per-feature standardization with rolling median and MAD.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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


*Module:* `dense_armor.utility.signals.preprocessing.select`
*Bases:* Transformer
*Guide:* [preprocessing/index.md](../preprocessing/index.md)

Keep the ``k`` features with the highest absolute running correlation.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### StandardScaler


*Module:* `dense_armor.utility.signals.preprocessing.scale`
*Bases:* Transformer
*Guide:* [preprocessing/index.md](../preprocessing/index.md)

Per-feature standardization with running mean and std.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### TextVectorizer


*Module:* `dense_armor.utility.signals.preprocessing.vectorizer`
*Bases:* Transformer

TF-IDF vectorizer with a fixed vocabulary, batch or online.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: 'int' = 1, shuffle: 'bool' = False, seed: 'int | None' = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, t=None)`
- `learn_one(x, y=None, t: 'float | None' = None)`
- `load_state_dict(state: 'dict')`
- `mutate(new_attrs: dict[str, Any])`
- `set_params(**params: Any)`
- `state_dict()`
- `transform(documents: 'Iterable')`
- `transform_many(X, t=None)`
- `transform_matrix(documents)`
- `transform_one(x, t: 'float | None' = None)`
- `window_samples(seconds: float)`

### TFIDF


*Module:* `dense_armor.utility.signals.preprocessing.text`
*Bases:* Transformer
*Guide:* [preprocessing/index.md](../preprocessing/index.md)

Online TF-IDF following Silajev 2026, section 1.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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


*Module:* `dense_armor.utility.signals.preprocessing.text`
*Bases:* Transformer
*Guide:* [preprocessing/index.md](../preprocessing/index.md)

Lower-case word tokens, with optional n-grams.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### VarianceThreshold


*Module:* `dense_armor.utility.signals.preprocessing.select`
*Bases:* Transformer
*Guide:* [preprocessing/index.md](../preprocessing/index.md)

Drop features whose running variance is at or below ``threshold``.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### Metro


*Module:* `dense_armor.utility.signals.preprocessing.metro`

Public methods:

- `dec(v_new: float, fact: float)`
- `enc(raw_v: float)`

## Compose

### Discard


*Module:* `dense_armor.utility.infrastructure.compose.select`
*Bases:* Transformer

Drop the named channels.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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


*Module:* `dense_armor.utility.infrastructure.compose.func`
*Bases:* Transformer

Apply ``fn`` to channels, or to the whole dict.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### Select


*Module:* `dense_armor.utility.infrastructure.compose.select`
*Bases:* Transformer

Keep only the named channels.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

## Vision

### BagOfVisualWords


*Module:* `dense_armor.utility.signals.vision.vocabulary`
*Bases:* Transformer
*Guide:* [cluster/index.md](../cluster/index.md)

Visual vocabulary over image patches and per-frame histogram.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### CubeStream


*Module:* `dense_armor.utility.signals.vision.streams`
*Bases:* VideoStream

Iterate over a multi-channel cube ``(T, H, W, C)``.

### DenseFlow


*Module:* `dense_armor.utility.signals.vision.dense`
*Bases:* Transformer

Dense optical flow with the Horn-Schunck iteration.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, t=None)`
- `learn_one(x, y=None, t=None)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `set_params(**params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x, t=None)`
- `window_samples(seconds: float)`

### DifferenceImage


*Module:* `dense_armor.utility.signals.vision.registration`
*Bases:* Transformer

Difference between the stabilised frame and a robust reference.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, t=None)`
- `learn_one(x, y=None, t=None)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `set_params(**params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x, t=None)`
- `window_samples(seconds: float)`

### EventFrame


*Module:* `dense_armor.utility.signals.vision.quality`
*Bases:* Transformer

Accumulate an event stream into a frame, one event at a time.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, t=None)`
- `learn_one(x, y=None, t=None)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `set_params(**params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x=None, t=None)`
- `window_samples(seconds: float)`

### FalseColor


*Module:* `dense_armor.utility.signals.vision.visualize`
*Bases:* Transformer

Bring any single-channel frame into a visible RGB image.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, t=None)`
- `learn_one(x, y=None, t=None)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `set_params(**params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x, t=None)`
- `window_samples(seconds: float)`

### FeatureToSignal


*Module:* `dense_armor.utility.signals.vision.adapters`
*Bases:* Transformer

Turn a dict of scalar features into a Signal.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, t=None)`
- `learn_one(x, y=None, t=None)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `set_params(**params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x, t=None)`
- `window_samples(seconds: float)`

### FrameAnomaly


*Module:* `dense_armor.utility.signals.vision.scene`
*Bases:* Transformer

Score a frame with the patch memory of the library.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, t=None)`
- `learn_one(x, y=None, t=None)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `set_params(**params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x, t=None)`
- `window_samples(seconds: float)`

### FrameFeatures


*Module:* `dense_armor.utility.signals.vision.features`
*Bases:* Transformer
*Guide:* [vision/features.md](../vision/features.md)

Turn one frame into a fixed-length dict of float features.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### FrameJointFusion


*Module:* `dense_armor.utility.signals.vision.fusion`
*Bases:* Transformer

Estimate the camera latency by correlating video motion and joint speed.

Public methods:

- `add_frame(frame, t: float)`
- `add_joint_speed(speed: float, t: float)`
- `add_video_velocity(velocity: float, t: float)`
- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `estimate()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, t=None)`
- `learn_one(x, y=None, t=None)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `set_params(**params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x, t=None)`
- `window_samples(seconds: float)`

### FrameNormalizer


*Module:* `dense_armor.utility.signals.vision.robustness`
*Bases:* Transformer

Bring a frame into ``[0, 1]`` and flag the difficult cases.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, t=None)`
- `learn_one(x, y=None, t=None)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `set_params(**params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x, t=None)`
- `window_samples(seconds: float)`

### FrameQuality


*Module:* `dense_armor.utility.signals.vision.quality`
*Bases:* Transformer

Per-frame sharpness, exposure, noise and difficulty flags.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, t=None)`
- `learn_one(x, y=None, t=None)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `set_params(**params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x, t=None)`
- `window_samples(seconds: float)`

### FrameWindow


*Module:* `dense_armor.utility.signals.vision.window`
*Bases:* Transformer

Ring buffer of the last ``n`` frames or ``s`` seconds, with statistics.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, t=None)`
- `learn_one(x, y=None, t=None)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `set_params(**params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x, t=None)`
- `window_samples(seconds: float)`

### GlobalMotion


*Module:* `dense_armor.utility.signals.vision.motion`
*Bases:* Transformer

Estimate the global scene shift with phase correlation.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, t=None)`
- `learn_one(x, y=None, t=None)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `set_params(**params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x, t=None)`
- `window_samples(seconds: float)`

### IncrementalPCA


*Module:* `dense_armor.utility.signals.vision.reduce`
*Bases:* Transformer
*Guide:* [vision/reduce.md](../vision/reduce.md)

Online PCA: Oja's rule for one component, stochastic subspace iteration with QR for several.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### Kinematics


*Module:* `dense_armor.utility.signals.vision.landmarks`
*Bases:* Transformer

Angles, speeds, symmetry and cycle residual from landmark points.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, t=None)`
- `learn_one(x, y=None, t=None)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `set_params(**params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x, t=None)`
- `window_samples(seconds: float)`

### LandmarkCleaner


*Module:* `dense_armor.utility.signals.vision.landmarks`
*Bases:* Transformer

Causal Hampel filter plus gap filling on a landmark stream.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, t=None)`
- `learn_one(x, y=None, t=None)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `set_params(**params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x, t=None)`
- `window_samples(seconds: float)`

### LandmarkStream


*Module:* `dense_armor.utility.signals.vision.landmarks`

A stream of landmark samples, one per frame.

### MotionEnergy


*Module:* `dense_armor.utility.signals.vision.motion`
*Bases:* Transformer

Per-frame motion energy from the difference against the previous frame.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, t=None)`
- `learn_one(x, y=None, t=None)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `set_params(**params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x, t=None)`
- `window_samples(seconds: float)`

### MotionMask


*Module:* `dense_armor.utility.signals.vision.dense`
*Bases:* Transformer

Boolean mask of the pixels whose residual flow is large.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, t=None)`
- `learn_one(x, y=None, t=None)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `set_params(**params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x, t=None)`
- `window_samples(seconds: float)`

### MovingObjectTracker


*Module:* `dense_armor.utility.signals.vision.tracking`
*Bases:* Transformer

Track moving objects from a frame stream or from external boxes.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, t=None)`
- `learn_one(x, y=None, t=None)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `set_params(**params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x, t=None)`
- `update(boxes)`
- `window_samples(seconds: float)`

### Overlay


*Module:* `dense_armor.utility.signals.vision.visualize`
*Bases:* Transformer

Draw the analysis on a frame.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, t=None)`
- `learn_one(x, y=None, t=None)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `set_params(**params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x, t=None)`
- `window_samples(seconds: float)`

### PatchFeatures


*Module:* `dense_armor.utility.signals.vision.patches`
*Bases:* Transformer
*Guide:* [vision/patches.md](../vision/patches.md)

Cut a frame into patches and describe each one locally.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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


*Module:* `dense_armor.utility.signals.vision.patches`
*Bases:* AnomalyDetector
*Guide:* [vision/patches.md](../vision/patches.md)

Memory bank of normal patch descriptors with a bounded size.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `finalize()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### RandomProjection


*Module:* `dense_armor.utility.signals.vision.reduce`
*Bases:* Transformer
*Guide:* [vision/reduce.md](../vision/reduce.md)

Fixed Gaussian random projection to ``k`` dimensions.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### ArrayStream


*Module:* `dense_armor.utility.signals.vision.streams`
*Guide:* [vision/frames.md](../vision/frames.md)

Iterate over a numpy array of frames.

### CameraStream


*Module:* `dense_armor.utility.signals.vision.streams`
*Guide:* [vision/frames.md](../vision/frames.md)

Iterate over camera frames through OpenCV.

### Frame


*Module:* `dense_armor.utility.signals.vision.streams`
*Guide:* [vision/frames.md](../vision/frames.md)

One image frame with its timestamp.

Public methods:

- `to_dict()`

### ImageFolderStream


*Module:* `dense_armor.utility.signals.vision.streams`
*Guide:* [vision/frames.md](../vision/frames.md)

Iterate over image files in a folder.

### Registration


*Module:* `dense_armor.utility.signals.vision.registration`
*Bases:* Transformer

Estimate the translation between a frame and a reference.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, t=None)`
- `learn_one(x, y=None, t=None)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `set_params(**params: Any)`
- `set_reference(img)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x, t=None)`
- `window_samples(seconds: float)`

### SceneChange


*Module:* `dense_armor.utility.signals.vision.scene`
*Bases:* Transformer

Scalar distance between consecutive frames, given to a drift detector.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, t=None)`
- `learn_one(x, y=None, t=None)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `set_params(**params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x, t=None)`
- `window_samples(seconds: float)`

### Stabilizer


*Module:* `dense_armor.utility.signals.vision.registration`
*Bases:* Transformer

Shift a frame back to the reference, so the scene looks fixed.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, t=None)`
- `learn_one(x, y=None, t=None)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `set_params(**params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x, t=None)`
- `window_samples(seconds: float)`

### Sync


*Module:* `dense_armor.utility.signals.vision.quality`
*Bases:* Transformer

Estimate and apply the clock offset between two streams.

Public methods:

- `add_a(value: float, t: float)`
- `add_b(value: float, t: float)`
- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `estimate()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, t=None)`
- `learn_one(x, y=None, t=None)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `set_params(**params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x=None, t=None)`
- `window_samples(seconds: float)`

### TrackToSignal


*Module:* `dense_armor.utility.signals.vision.adapters`
*Bases:* Transformer

Turn one track of the tracker into a Signal.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, t=None)`
- `learn_one(x, y=None, t=None)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `set_params(**params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x, t=None)`
- `window_samples(seconds: float)`

### VideoFileStream


*Module:* `dense_armor.utility.signals.vision.streams`
*Bases:* VideoStream

Read a video from a file, an URL, a glob of files or a network stream.

### VideoStream


*Module:* `dense_armor.utility.signals.vision.streams`

Common interface of the frame sources.

### VideoWriter


*Module:* `dense_armor.utility.signals.vision.visualize`

Write a video and a summary image of the signals over time.

Public methods:

- `close()`
- `log(name: str, value: float)`
- `write(rgb)`

## Learn

### DriftAdaptiveClassifier


*Module:* `dense_armor.utility.online.learn.online_classifiers`
*Bases:* Classifier
*Guide:* [learn/online_classifiers.md](../learn/online_classifiers.md)

Classifier with a background copy swapped in on drift detection.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### LEGO


*Module:* `dense_armor.utility.online.learn.metric_learning`
*Bases:* MetricLearner
*Guide:* [learn/metric_learning.md](../learn/metric_learning.md)

Online metric learning via LogDet regularization and exact gradient.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `distance(x: dict, x2: dict)`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep=True)`
- `learn_pair(x: dict, x2: dict, y: float)`
- `learn_triplet(x: dict, x_pos: dict, x_neg: dict)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `percentile_targets(X, y, low=5.0, high=95.0)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### MetricKNNClassifier


*Module:* `dense_armor.utility.online.learn.metric_learning`
*Bases:* Classifier
*Guide:* [learn/metric_learning.md](../learn/metric_learning.md)

K-nearest-neighbours classifier that learns its metric online.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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


*Module:* `dense_armor.utility.online.learn.metric_learning`
*Bases:* Root
*Guide:* [learn/metric_learning.md](../learn/metric_learning.md)

Root class for online metric learners.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `distance(x: dict, x2: dict)`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep=True)`
- `learn_pair(x: dict, x2: dict, y)`
- `learn_triplet(x: dict, x_pos: dict, x_neg: dict)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### OASIS


*Module:* `dense_armor.utility.online.learn.metric_learning`
*Bases:* MetricLearner
*Guide:* [learn/metric_learning.md](../learn/metric_learning.md)

Online metric learning via passive-aggressive bilinear similarity.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `distance(x: dict, x2: dict)`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep=True)`
- `learn_pair(x: dict, x2: dict, y)`
- `learn_triplet(x: dict, x_pos: dict, x_neg: dict)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### OnlineGaussianNB


*Module:* `dense_armor.utility.online.learn.online_classifiers`
*Bases:* Classifier
*Guide:* [learn/online_classifiers.md](../learn/online_classifiers.md)

Gaussian naive Bayes updated one sample at a time.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### OnlinePlattScaling


*Module:* `dense_armor.utility.online.learn.calibration`
*Bases:* ModelWrapper, Classifier
*Guide:* [learn/calibration.md](../learn/calibration.md)

Online Platt scaling.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### OnlineSoftmaxRegression


*Module:* `dense_armor.utility.online.learn.online_classifiers`
*Bases:* Classifier
*Guide:* [learn/online_classifiers.md](../learn/online_classifiers.md)

Multinomial logistic regression with AdaGrad.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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


*Module:* `dense_armor.utility.online.learn.metric_learning`
*Bases:* MetricLearner
*Guide:* [learn/metric_learning.md](../learn/metric_learning.md)

Online pseudo-metric learning with PSD projection.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `distance(x: dict, x2: dict)`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep=True)`
- `learn_pair(x: dict, x2: dict, y: int)`
- `learn_triplet(x: dict, x_pos: dict, x_neg: dict)`
- `load_state_dict(state: dict)`
- `mutate(new_attrs: dict)`
- `set_params(params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### RecursiveLeastSquares


*Module:* `dense_armor.utility.online.learn.online_dynamics`
*Bases:* Regressor
*Guide:* [learn/online_dynamics.md](../learn/online_dynamics.md)

Recursive least squares with exponential forgetting.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### DriftAwareResidualDynamicsLearner


*Module:* `dense_armor.utility.online.learn.online_dynamics`
*Bases:* ResidualDynamicsLearner
*Guide:* [learn/online_dynamics.md](../learn/online_dynamics.md)

``ResidualDynamicsLearner`` with a per-joint drift trigger on the residual.

Public methods:

- `learn_one(q, qd, qdd, tau_measured)`
- `predict_torque(q, qd, qdd)`

### ResidualDynamicsLearner


*Module:* `dense_armor.utility.online.learn.online_dynamics`
*Guide:* [learn/online_dynamics.md](../learn/online_dynamics.md)

Online residual torque learner over a nominal ``RigidBodyModel``.

Public methods:

- `learn_one(q, qd, qdd, tau_measured)`
- `predict_torque(q, qd, qdd)`

## Datasets

### Casper


*Module:* `dense_armor.utility.signals.datasets.casper`

CASPER UR3e joint stream.

Public methods:

- `n_rows()`
- `stream(columns: list, start_h=None, stop_h=None)`

### DriftStream


*Module:* `dense_armor.utility.signals.datasets.drift_stream`

Synthetic binary stream with concept drift.

Public methods:

- `data()`
- `stream()`

### SyntheticArm


*Module:* `dense_armor.utility.signals.datasets.synthetic_arm`

Periodic joint stream from a URDF with an optional fault.

Public methods:

- `nominal_torque(q: ndarray, qd: ndarray, qdd: ndarray)`
- `payload_torque(q: ndarray)`
- `stream()`
- `trajectory(t: float)`

## Protect

### Orca


*Module:* `dense_armor.utility.online.protect.orca`
*Guide:* [shields/orca.md](../shields/orca.md)

Public methods:

- `protect_and_forward(ai_model_callable: Optional, x_corrupted: ndarray, x_reference=None, use_input_shield=True, use_model_injection=True, use_output_shield=True, use_arbiter=False)`
- `tipi_corruzione_visti(slice_shape: tuple)`

### StreamingArbiter


*Module:* `dense_armor.utility.online.protect.streaming_arbiter`
*Guide:* [protect/arbiter.md](../protect/arbiter.md)

Bounded-delay streaming version of `arbiter.classify_segments`.

Public methods:

- `flush()`
- `update(x: float)`

### StreamingHealing


*Module:* `dense_armor.utility.online.protect.streaming_arbiter`
*Guide:* [protect/arbiter.md](../protect/arbiter.md)

One-sample-at-a-time version of `healing.healing_filter`.

Public methods:

- `flush()`
- `update(x: float)`

## Bridges

### LLMEmbeddingClassifier


*Module:* `dense_armor.bridges.llm`
*Bases:* Estimator

Classifier whose input is text and whose features come from ``embed``.

Public methods:

- `clone(new_params=None, include_attributes=False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
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

### SklearnClassifier


*Module:* `dense_armor.bridges.sklearn`
*Bases:* ClassifierMixin, _DenseWrapper, BaseEstimator

A Dense-Armor classifier behind the scikit-learn classifier interface.

Public methods:

- `fit(X: 'Any', y: 'Any')`
- `get_metadata_routing()`
- `get_params(deep=True)`
- `partial_fit(X: 'Any', y: 'Any', classes: 'Any' = None)`
- `predict(X: 'Any')`
- `predict_proba(X: 'Any')`
- `score(X, y, sample_weight=None)`
- `set_params(**params)`
- `set_partial_fit_request(*, classes: Union[bool, NoneType, str] = '$UNCHANGED$')`
- `set_score_request(*, sample_weight: Union[bool, NoneType, str] = '$UNCHANGED$')`

### SklearnRegressor


*Module:* `dense_armor.bridges.sklearn`
*Bases:* RegressorMixin, _DenseWrapper, BaseEstimator

A Dense-Armor regressor behind the scikit-learn regressor interface.

Public methods:

- `fit(X: 'Any', y: 'Any')`
- `get_metadata_routing()`
- `get_params(deep=True)`
- `partial_fit(X: 'Any', y: 'Any')`
- `predict(X: 'Any')`
- `score(X, y, sample_weight=None)`
- `set_params(**params)`
- `set_score_request(*, sample_weight: Union[bool, NoneType, str] = '$UNCHANGED$')`

### SklearnTransformer


*Module:* `dense_armor.bridges.sklearn`
*Bases:* TransformerMixin, _DenseWrapper, BaseEstimator

A Dense-Armor transformer behind the scikit-learn transformer interface.

Public methods:

- `fit(X: 'Any', y: 'Any' = None)`
- `fit_transform(X, y=None, **fit_params)`
- `get_feature_names_out(input_features: 'Any' = None)`
- `get_metadata_routing()`
- `get_params(deep=True)`
- `partial_fit(X: 'Any', y: 'Any' = None)`
- `set_output(*, transform=None)`
- `set_params(**params)`
- `transform(X: 'Any')`

### TorchAdapter


*Module:* `dense_armor.bridges.torch`

A ``torch.nn.Module`` usable inside Dense-Armor.

Public methods:

- `apply(x: 'Any')`

### TorchModule


*Module:* `dense_armor.bridges.torch`
*Bases:* Module

A Dense-Armor model wrapped as a ``torch.nn.Module``.

Public methods:

- `add_module(name: str, module: Optional[ForwardRef('Module')])`
- `apply(fn: collections.abc.Callable[['Module'], None])`
- `bfloat16()`
- `buffers(recurse: bool = True)`
- `children()`
- `compile(*args, **kwargs)`
- `cpu()`
- `cuda(device: int | torch.device | None = None)`
- `double()`
- `eval()`
- `extra_repr()`
- `float()`
- `forward(x: 'torch.Tensor')`
- `get_buffer(target: str)`
- `get_extra_state()`
- `get_parameter(target: str)`
- `get_submodule(target: str)`
- `half()`
- `ipu(device: int | torch.device | None = None)`
- `load_state_dict(state_dict: collections.abc.Mapping[str, Any], strict: bool = True, assign: bool = False)`
- `modules(remove_duplicate: bool = True)`
- `mtia(device: int | torch.device | None = None)`
- `named_buffers(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True)`
- `named_children()`
- `named_modules(memo: set['Module'] | None = None, prefix: str = '', remove_duplicate: bool = True)`
- `named_parameters(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True)`
- `parameters(recurse: bool = True)`
- `register_backward_hook(hook: collections.abc.Callable[['Module', tuple[torch.Tensor, ...] | torch.Tensor, tuple[torch.Tensor, ...] | torch.Tensor], tuple[torch.Tensor, ...] | torch.Tensor | None])`
- `register_buffer(name: str, tensor: torch.Tensor | None, persistent: bool = True)`
- `register_forward_hook(hook: collections.abc.Callable[[~T, tuple[Any, ...], Any], Any | None] | collections.abc.Callable[[~T, tuple[Any, ...], dict[str, Any], Any], Any | None], *, prepend: bool = False, with_kwargs: bool = False, always_call: bool = False)`
- `register_forward_pre_hook(hook: collections.abc.Callable[[~T, tuple[Any, ...]], Any | None] | collections.abc.Callable[[~T, tuple[Any, ...], dict[str, Any]], tuple[Any, dict[str, Any]] | None], *, prepend: bool = False, with_kwargs: bool = False)`
- `register_full_backward_hook(hook: collections.abc.Callable[['Module', tuple[torch.Tensor, ...] | torch.Tensor, tuple[torch.Tensor, ...] | torch.Tensor], tuple[torch.Tensor, ...] | torch.Tensor | None], prepend: bool = False)`
- `register_full_backward_pre_hook(hook: collections.abc.Callable[['Module', tuple[torch.Tensor, ...] | torch.Tensor], tuple[torch.Tensor, ...] | torch.Tensor | None], prepend: bool = False)`
- `register_load_state_dict_post_hook(hook)`
- `register_load_state_dict_pre_hook(hook)`
- `register_module(name: str, module: Optional[ForwardRef('Module')])`
- `register_parameter(name: str, param: torch.nn.parameter.Parameter | None)`
- `register_state_dict_post_hook(hook)`
- `register_state_dict_pre_hook(hook)`
- `requires_grad_(requires_grad: bool = True)`
- `set_extra_state(state: Any)`
- `set_submodule(target: str, module: 'Module', strict: bool = False)`
- `share_memory()`
- `state_dict(*args, destination=None, prefix='', keep_vars=False)`
- `to(*args, **kwargs)`
- `to_empty(*, device: str | torch.device | int | None, recurse: bool = True)`
- `train(mode: bool = True)`
- `type(dst_type: torch.dtype | str)`
- `xpu(device: int | torch.device | None = None)`
- `zero_grad(set_to_none: bool = True)`

## Armor

### Armor


*Module:* `dense_armor.armor`
*Guide:* [shields/armor.md](../shields/armor.md)

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

## Deep learning

### AccumulatingStep


*Module:* `dense_armor.utility.deep.neural.compiled`

The compiled pieces of a step that accumulates the gradient.

### AddPositions


*Module:* `dense_armor.utility.deep.neural.models`

Add the sinusoidal positions of page 6 of Vaswani et al. 2017.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `params()`

### AGEM


*Module:* `dense_armor.utility.deep.neural.continual`

A-GEM (Chaudhry et al. 2019, Eq. 11 p. 5).

Public methods:

- `memory()`
- `observe(x: 'Any', y: 'Any', task_id: 'int')`
- `project(g: 'jnp.ndarray', g_ref: 'jnp.ndarray')`

### ArchivalMemory


*Module:* `dense_armor.utility.deep.neural.consolidation.archival`

Unforgetting architetturale: residual tree + lifecycle + dual memory.

Public methods:

- `importance(x: 'jnp.ndarray', t: 'float' = 0.0)`
- `read(x: 'jnp.ndarray')`
- `reconstruct(idx: 'int')`
- `transition(state: 'str', accesses: 'int', age_s: 'float')`
- `write(x: 'jnp.ndarray', t: 'float' = 0.0)`
- `write_depth(importance: 'float')`

### AsTransformer


*Module:* `dense_armor.utility.deep.neural.adapter`
*Bases:* Transformer

A layer exposed as a ``Transformer``.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, t=None)`
- `learn_one(x: 'dict', y: 'Any' = None, t: 'float | None' = None)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `set_params(**params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: 'dict', t: 'float | None' = None)`
- `window_samples(seconds: float)`

### Autoencoder


*Module:* `dense_armor.utility.deep.neural.autoencoder`

An encoder and a decoder, joined end to end.

Public methods:

- `backward(g: 'jax.Array')`
- `decode(z: 'jax.Array')`
- `encode(x: 'jax.Array')`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `params()`
- `reconstruction_error(x: 'jax.Array')`

### AvgPool1D


*Module:* `dense_armor.utility.deep.neural.conv`

1-D average pooling. Input ``(L, c)`` or ``(B, L, c)``.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `params()`

### AvgPool2D


*Module:* `dense_armor.utility.deep.neural.conv`

2-D average pooling. Input ``(H, W, c)`` or ``(B, H, W, c)``.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `params()`

### BackgroundLearner


*Module:* `dense_armor.utility.deep.neural.background`

Learn on a worker thread; predict on the last served copy.

Public methods:

- `close()`
- `flush(timeout: 'float | None' = None)`
- `learn_one(x: 'Any', y: 'Any' = None, t: 'float | None' = None)`
- `predict_one(x: 'Any', t: 'float | None' = None)`

### BatchNorm


*Module:* `dense_armor.utility.deep.neural.norm`

Batch normalisation of a 2-D batch ``(m, n_features)``.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `params()`

### BCEWithLogits


*Module:* `dense_armor.utility.deep.neural.losses`

Binary cross-entropy with logits, with a cached last logits and targets.

Public methods:

- `backward()`
- `forward(logits: 'jax.Array', targets: 'jax.Array')`

### Bidirectional


*Module:* `dense_armor.utility.deep.neural.bidirectional`

A forward and a backward recurrent layer, concatenated.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `params()`

### BottleneckFusion


*Module:* `dense_armor.utility.deep.neural.bottleneck`

One fusion layer of Nagrani et al. 2021, Eq.(7) to Eq.(9).

Public methods:

- `backward(ga: 'Any', gb: 'Any', gfsn: 'Any')`
- `forward(a: 'Any', b: 'Any', fsn: 'Any' = None, training: 'bool' = True)`
- `params()`

### CompiledStep


*Module:* `dense_armor.utility.deep.neural.compiled`

The compiled steps of a Trainer.

### ConstantLR


*Module:* `dense_armor.utility.deep.neural.schedules`

A constant learning rate.

Public methods:

- `multiplier(epoch: 'float')`

### Conv1D


*Module:* `dense_armor.utility.deep.neural.conv`

1-D cross-correlation with zero padding, optional causal shift.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `init_buffer()`
- `params()`
- `step(x_t: 'jax.Array', buffer: 'jax.Array')`

### Conv2D


*Module:* `dense_armor.utility.deep.neural.conv`

2-D cross-correlation with zero padding, channels last.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `params()`

### Conv3D


*Module:* `dense_armor.utility.deep.neural.conv_more`

3-D cross-correlation with zero padding, channels last.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `params()`

### ConvTranspose1D


*Module:* `dense_armor.utility.deep.neural.convtranspose`

1-D transposed convolution, channels last.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `params()`

### ConvTranspose2D


*Module:* `dense_armor.utility.deep.neural.convtranspose`

2-D transposed convolution, channels last.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `params()`

### CosineWarmRestarts


*Module:* `dense_armor.utility.deep.neural.schedules`

Cosine annealing with warm restarts.

Public methods:

- `multiplier(epoch: 'float')`

### DataStream


*Module:* `dense_armor.utility.deep.neural.stream`

Batches of ``(x, y)`` from arrays, memory-mapped files or a generator.

### DeepGenerativeReplay


*Module:* `dense_armor.utility.deep.neural.generative_replay`

Deep generative replay (Shin et al. 2017, sec. 2.3 p. 3).

Public methods:

- `predict(x: 'Any')`
- `sample(generator: 'Sequential', n: 'int')`
- `train_task(x_real: 'Any', y_real: 'Any', n_epochs: 'int' = 5, batch_size: 'int' = 32)`

### Dense


*Module:* `dense_armor.utility.deep.neural.layers`

A dense layer ``y = x @ W + b``.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `params()`

### DepthwiseConv2D


*Module:* `dense_armor.utility.deep.neural.conv_more`

Depthwise 2-D convolution: one square kernel per channel.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `params()`

### Dropout


*Module:* `dense_armor.utility.deep.neural.layers`

Random dropout with the scaling of Hinton et al. 2012.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `params()`

### DualMemory


*Module:* `dense_armor.utility.deep.neural.consolidation.memory`

Dual memory (HiCL, Kapoor et al. 2026, adapted to dense networks).

Public methods:

- `consolidate()`
- `replay(n: 'int', t_now: 'float' = 0.0)`
- `size()`
- `top_k_separation(x: 'jnp.ndarray')`
- `write(x: 'jnp.ndarray', t: 'float' = 0.0)`

### ELU


*Module:* `dense_armor.utility.deep.neural.activations_more`

ELU with a scale ``alpha`` (Eq.(15), page 5 of Clevert et al.).

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array')`

### Embedding


*Module:* `dense_armor.utility.deep.neural.layers`

A lookup table ``E[ids]``.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(ids: 'jax.Array', training: 'bool' = True)`
- `params()`

### EWC


*Module:* `dense_armor.utility.deep.neural.continual`

Elastic weight consolidation (Kirkpatrick et al. 2017, Eq. 3 p. 3).

Public methods:

- `consolidate(x: 'Any', y: 'Any')`
- `flat_params()`
- `penalty(theta: 'jnp.ndarray | None' = None)`
- `set_flat_params(theta: 'jnp.ndarray')`

### Flatten


*Module:* `dense_armor.utility.deep.neural.conv`

Flatten every axis after the batch axis.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `params()`

### FocalLoss


*Module:* `dense_armor.utility.deep.neural.losses_more`

Focal loss ``-alpha_t (1 - p_t)**gamma * log(p_t)``.

Public methods:

- `backward()`
- `forward(logits: 'jax.Array', y: 'jax.Array')`

### GatedCrossAttention


*Module:* `dense_armor.utility.deep.neural.gated_attention`

Cross-attention gated by a per-query signal.

Public methods:

- `backward(g_out: 'Any')`
- `forward(xq: 'Any', xc: 'Any', signal: 'Any', training: 'bool' = True)`
- `params()`

### GELU


*Module:* `dense_armor.utility.deep.neural.activations_more`

GELU (exact or approximate) with a forward and a backward.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array')`

### GEM


*Module:* `dense_armor.utility.deep.neural.continual`

Gradient Episodic Memory (Lopez-Paz & Ranzato 2017, Eq. 8 QP and Eq. 11 dual, p. 4).

Public methods:

- `memory()`
- `observe(x: 'Any', y: 'Any', task_id: 'int')`
- `project(g: 'jnp.ndarray', mem_grads: 'Sequence[jnp.ndarray]')`

### GLUFeedForward


*Module:* `dense_armor.utility.deep.neural.glu`

The feed-forward layer ``(act(x W) * (x V)) W2`` of Shazeer 2020.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `params()`

### GroupNorm


*Module:* `dense_armor.utility.deep.neural.norm_more`

Group normalization of the channels of a sample.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `params()`

### GRU


*Module:* `dense_armor.utility.deep.neural.recurrent`

GRU layer, Eqs.(5) to (8), page 3 of Cho et al. 2014.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `init_state(batch: 'int | None' = None)`
- `params()`
- `step(x_t: 'jax.Array', state: 'jax.Array')`

### HedgeNetClassifier


*Module:* `dense_armor.utility.deep.neural.hedge`
*Bases:* Classifier

Online deep learning with Hedge Backpropagation.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: 'bool' = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: 'dict', y: 'Any', t: 'float | None' = None)`
- `load_state_dict(state: 'dict[str, Any]')`
- `mutate(new_attrs: dict[str, Any])`
- `params()`
- `predict_many(X, t=None)`
- `predict_one(x: 'dict', t: 'float | None' = None)`
- `predict_proba_many(X, t=None)`
- `predict_proba_one(x: 'dict', t: 'float | None' = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### HingeLoss


*Module:* `dense_armor.utility.deep.neural.losses_more`

Hinge loss ``max(0, 1 - y * pred)`` for ``y`` in {-1, +1}.

Public methods:

- `backward()`
- `forward(pred: 'jax.Array', y: 'jax.Array')`

### Huber


*Module:* `dense_armor.utility.deep.neural.losses`

Huber loss with a cached last prediction and target.

Public methods:

- `backward()`
- `forward(pred: 'jax.Array', target: 'jax.Array')`

### InstanceNorm


*Module:* `dense_armor.utility.deep.neural.norm_more`

Instance normalization: one group per channel.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `params()`

### KLDivLoss


*Module:* `dense_armor.utility.deep.neural.losses_more`

Kullback-Leibler divergence ``sum_k y_k (log y_k - log p_k)``.

Public methods:

- `backward()`
- `forward(logits: 'jax.Array', target_probs: 'jax.Array')`

### L1Loss


*Module:* `dense_armor.utility.deep.neural.losses_more`

Mean absolute error with cached last call.

Public methods:

- `backward()`
- `forward(pred: 'jax.Array', target: 'jax.Array')`

### LabelSmoothingCE


*Module:* `dense_armor.utility.deep.neural.losses_more`

Softmax cross-entropy with smoothed labels, cached last call.

Public methods:

- `backward()`
- `forward(logits: 'jax.Array', labels: 'jax.Array')`

### LastStep


*Module:* `dense_armor.utility.deep.neural.models`

Take the last step of a sequence ``(B, T, n)``, giving ``(B, n)``.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `params()`

### LatentCrossAttention


*Module:* `dense_armor.utility.deep.neural.latent_attention`

Cross-attention whose queries are a learned latent array.

Public methods:

- `backward(g: 'Any')`
- `forward(xc: 'Any', training: 'bool' = True)`
- `params()`

### LayerNorm


*Module:* `dense_armor.utility.deep.neural.norm`

Layer normalisation over the features of each sample.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `params()`

### LeakyReLU


*Module:* `dense_armor.utility.deep.neural.activations_more`

Leaky ReLU with a fixed slope (no Param).

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array')`

### LearnedPositions


*Module:* `dense_armor.utility.deep.neural.positions`

Add a learned vector to each position of a sequence.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `params()`

### LifecycleMemory


*Module:* `dense_armor.utility.deep.neural.consolidation.memory`

Lifecycle-quantised memory (SuperLocalMemory V3.3, Bhardwaj 2026, p.1).

Public methods:

- `access(idx: 'int', t: 'float' = 0.0)`
- `mse_of(idx: 'int')`
- `read(idx: 'int')`
- `state_of(idx: 'int')`
- `tick(t: 'float' = 0.0)`
- `write(x: 'jnp.ndarray', t: 'float' = 0.0)`

### LinearWarmup


*Module:* `dense_armor.utility.deep.neural.schedules_more`

Linear warmup: ``lr_start`` then a linear ramp to ``lr_end``.

Public methods:

- `multiplier(epoch: 'float')`

### LSTM


*Module:* `dense_armor.utility.deep.neural.recurrent`

Vanilla LSTM with peepholes, Greff et al. 2017 pages 1-2.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `init_state(batch: 'int | None' = None)`
- `params()`
- `step(x_t: 'jax.Array', state: 'tuple[jax.Array, jax.Array]')`

### MatryoshkaLoss


*Module:* `dense_armor.utility.deep.neural.matryoshka`

The multi-size loss of Matryoshka Representation Learning.

Public methods:

- `backward()`
- `forward(z: 'jax.Array', y: 'Any')`
- `logits(z: 'jax.Array', m: 'int')`
- `params()`
- `truncate(z: 'jax.Array', m: 'int')`

### MaxPool1D


*Module:* `dense_armor.utility.deep.neural.conv`

1-D max pooling with the first maximum going to the gradient.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `params()`

### MaxPool2D


*Module:* `dense_armor.utility.deep.neural.conv`

2-D max pooling with the first maximum going to the gradient.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `params()`

### MeanOverTime


*Module:* `dense_armor.utility.deep.neural.models`

Mean over the time axis of ``(B, T, n)`` or ``(T, n)``.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `params()`

### MSE


*Module:* `dense_armor.utility.deep.neural.losses`

Mean squared error with a cached last prediction and target.

Public methods:

- `backward()`
- `forward(pred: 'jax.Array', target: 'jax.Array')`

### MultiHeadCrossAttention


*Module:* `dense_armor.utility.deep.neural.cross_attention`

Multi-head attention with queries and context from two streams.

Public methods:

- `append_context(cache: 'tuple[jax.Array, jax.Array]', xc_t: 'jax.Array')`
- `backward(g: 'jax.Array')`
- `forward(xq: 'jax.Array', xc: 'jax.Array', training: 'bool' = True, mask: 'jax.Array | None' = None)`
- `init_cache()`
- `params()`
- `step(xq_t: 'jax.Array', cache: 'tuple[jax.Array, jax.Array]')`

### MultiHeadSelfAttention


*Module:* `dense_armor.utility.deep.neural.attention`

Multi-head self-attention of Vaswani et al. 2017, page 5.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `init_cache()`
- `init_static_cache(length: 'int')`
- `params()`
- `step(x_t: 'jax.Array', cache: 'tuple[jax.Array, jax.Array]')`
- `step_static(x_t: 'jax.Array', cache: 'tuple[jax.Array, jax.Array]', pos: 'jax.Array')`

### NextTokenLoss


*Module:* `dense_armor.utility.deep.neural.losses_lm`

Cross-entropy of the logits at ``t`` against the token at ``t+1``.

Public methods:

- `backward()`
- `forward(logits: 'jax.Array', y: 'jax.Array')`

### NoamSchedule


*Module:* `dense_armor.utility.deep.neural.schedules_more`

Noam schedule: linear warmup then inverse-square-root decay.

Public methods:

- `multiplier(epoch: 'float')`

### OnlineAutoencoder


*Module:* `dense_armor.utility.deep.neural.online_autoencoder`
*Bases:* AnomalyDetector

An autoencoder trained one sample at a time, as an anomaly score.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `encode_one(x: 'dict', t: 'float | None' = None)`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: 'bool' = True)`
- `learn_many(X, t=None)`
- `learn_one(x: 'dict', t: 'float | None' = None)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `score_many(X, t=None)`
- `score_one(x: 'dict', t: 'float | None' = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### OnlineNetClassifier


*Module:* `dense_armor.utility.deep.neural.online`
*Bases:* Classifier

A network classifier trained one sample at a time.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: 'bool' = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: 'dict', y: 'Any', t: 'float | None' = None)`
- `load_state_dict(state: 'dict[str, Any]')`
- `mutate(new_attrs: dict[str, Any])`
- `predict_many(X, t=None)`
- `predict_one(x: 'dict', t: 'float | None' = None)`
- `predict_proba_many(X, t=None)`
- `predict_proba_one(x: 'dict', t: 'float | None' = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### OnlineNetRegressor


*Module:* `dense_armor.utility.deep.neural.online`
*Bases:* Regressor

A network trained one sample at a time.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: 'bool' = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: 'dict', y: 'Any', t: 'float | None' = None)`
- `load_state_dict(state: 'dict[str, Any]')`
- `mutate(new_attrs: dict[str, Any])`
- `predict_many(X, t=None, return_std=False)`
- `predict_one(x: 'dict', t: 'float | None' = None, return_std: 'bool' = False)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### OnlineNetVector


*Module:* `dense_armor.utility.deep.neural.online_vector`
*Bases:* Regressor

A network trained one sample at a time, on vector targets.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: 'bool' = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: 'Any', y: 'Any', t: 'float | None' = None)`
- `load_state_dict(state: 'dict[str, Any]')`
- `mutate(new_attrs: dict[str, Any])`
- `predict_many(X, t=None, return_std=False)`
- `predict_one(x: 'Any', t: 'float | None' = None, return_std: 'bool' = False)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### Parallel


*Module:* `dense_armor.utility.deep.neural.chain`

Several branches on the same input, merged on the output.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `params()`
- `zero_grad()`

### Param


*Module:* `dense_armor.utility.deep.neural.params`

A jax array with a gradient of the same shape and dtype.

Public methods:

- `update(optimizer: 'Any')`
- `zero_grad()`

### PQMemory


*Module:* `dense_armor.utility.deep.neural.consolidation.vq`

Product quantisation (Jégou, Douze, Schmid 2011, IEEE TPAMI 33(1)).

Public methods:

- `asymmetric_distance(q: 'jnp.ndarray', code: 'jnp.ndarray')`
- `decode(code: 'jnp.ndarray')`
- `encode(x: 'jnp.ndarray')`
- `fit(documents: 'jnp.ndarray')`

### PRelu


*Module:* `dense_armor.utility.deep.neural.activations`

PReLU with ``a`` as a ``Param``; ``backward`` fills ``a.grad``.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array')`
- `params()`

### Relu


*Module:* `dense_armor.utility.deep.neural.activations`

ReLU: ``forward`` caches ``x`` and ``y``; ``backward`` returns ``dx``.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array')`

### Residual


*Module:* `dense_armor.utility.deep.neural.layers`

``y = x + inner.forward(x)`` (He et al. 2015, Eqn.(1) page 3).

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `params()`

### ResidualTree


*Module:* `dense_armor.utility.deep.neural.consolidation.tree`

Residual tree (DeltaMem, Tan et al. 2026, adapted to dense networks).

Public methods:

- `consolidate()`
- `insert(x: 'jnp.ndarray')`
- `max_nodes()`
- `n_events()`
- `n_nodes()`
- `read(x: 'jnp.ndarray')`
- `state_size_bytes()`

### RMSNorm


*Module:* `dense_armor.utility.deep.neural.norm_rms`

RMS layer normalisation over the features of each sample.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `params()`

### RNN


*Module:* `dense_armor.utility.deep.neural.recurrent`

Elman recurrent layer ``h = tanh(W x + U h_prev + b)``.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `init_state(batch: 'int | None' = None)`
- `params()`
- `step(x_t: 'jax.Array', state: 'jax.Array')`

### RotaryDecoder


*Module:* `dense_armor.utility.deep.neural.rotary_decoder`

``Embedding``, ``n_layers`` modern blocks, ``RMSNorm`` and a ``Dense`` head.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(ids: 'jax.Array', training: 'bool' = True)`
- `init_static_cache(length: 'int')`
- `params()`
- `step_compiled(token: 'jax.Array', caches: 'list[tuple[jax.Array, jax.Array]]', pos: 'int')`
- `step_static(token: 'jax.Array', caches: 'list[tuple[jax.Array, jax.Array]]', pos: 'jax.Array')`

### RotaryDecoderBlock


*Module:* `dense_armor.utility.deep.neural.rotary_decoder`

Pre-norm decoder block: ``x + Attn(RMSNorm(x))``, then ``+ FFN(RMSNorm(.))``.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `init_static_cache(length: 'int')`
- `params()`
- `step_static(x_t: 'jax.Array', cache: 'tuple[jax.Array, jax.Array]', pos: 'jax.Array')`

### SeparableConv2D


*Module:* `dense_armor.utility.deep.neural.conv_more`

Separable 2-D convolution: depthwise then pointwise.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `params()`

### Seq2Seq


*Module:* `dense_armor.utility.deep.neural.seq2seq`

A recurrent encoder, a recurrent decoder, a dense head.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `params()`

### Sequential


*Module:* `dense_armor.utility.deep.neural.chain`

A chain of layers, applied in order.

Public methods:

- `add(layer: 'Any')`
- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `load_state_dict(state: 'list[np.ndarray]')`
- `params()`
- `state_dict()`
- `zero_grad()`

### Sigmoid


*Module:* `dense_armor.utility.deep.neural.activations`

Sigmoid: ``forward`` caches ``x`` and ``y``; ``backward`` returns ``dx``.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array')`

### SiLU


*Module:* `dense_armor.utility.deep.neural.activations_more`

SiLU: ``x * sigmoid(x)``.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array')`

### Softmax


*Module:* `dense_armor.utility.deep.neural.activations`

Softmax over the last axis: ``forward`` caches ``x`` and ``y``; ``backward`` returns ``dx``.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array')`

### SoftmaxCrossEntropy


*Module:* `dense_armor.utility.deep.neural.losses`

Softmax cross-entropy with a cached last logits and labels.

Public methods:

- `backward()`
- `forward(logits: 'jax.Array', labels: 'jax.Array')`

### Softplus


*Module:* `dense_armor.utility.deep.neural.activations_more`

Softplus with a scale ``beta``.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array')`

### SparseMoE


*Module:* `dense_armor.utility.deep.neural.moe`

Sparse mixture of experts with dispatch.

Public methods:

- `backward(g: 'jax.Array')`
- `dropped_fraction()`
- `expert_usage()`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `load_balancing_loss()`
- `params()`

### StepLR


*Module:* `dense_armor.utility.deep.neural.schedules`

Step schedule: ``lr * gamma ** floor(epoch / step)``.

Public methods:

- `multiplier(epoch: 'float')`

### Tanh


*Module:* `dense_armor.utility.deep.neural.activations`

Tanh: ``forward`` caches ``x`` and ``y``; ``backward`` returns ``dx``.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array')`

### TestTimeCompute


*Module:* `dense_armor.utility.deep.neural.test_time`

Test-time compute wrapper.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `params()`

### TheoreticalBound


*Module:* `dense_armor.utility.deep.neural.consolidation.memory`

Classical Hopfield capacity as a forgetting bound (placeholder).

Public methods:

- `bound(n_events: 'int')`
- `capacity()`

### Trainer


*Module:* `dense_armor.utility.deep.neural.trainer`

Train a model on a dataset, with mini-batches and schedules.

Public methods:

- `evaluate(X: 'Any', y: 'Any')`
- `fit(X: 'Any', y: 'Any', epochs: 'int', batch_size: 'int | None', key: 'Any', X_val: 'Any' = None, y_val: 'Any' = None, patience: 'int | None' = None, min_delta: 'float' = 0.0, shuffle: 'bool' = True, stream: 'DataStream | None' = None)`
- `predict(X: 'Any')`

### TransformerBlock


*Module:* `dense_armor.utility.deep.neural.attention`

The encoder block of Vaswani et al. 2017.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `init_cache()`
- `params()`
- `step(x_t: 'jax.Array', cache: 'tuple[jax.Array, jax.Array]')`

### TransformerDecoderBlock


*Module:* `dense_armor.utility.deep.neural.decoder`

One decoder layer of Vaswani et al. 2017.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array', memory: 'jax.Array | None' = None, training: 'bool' = True)`
- `params()`

### Upsample1D


*Module:* `dense_armor.utility.deep.neural.convtranspose`

Nearest-neighbour upsampling along the length, channels last.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `params()`

### Upsample2D


*Module:* `dense_armor.utility.deep.neural.convtranspose`

Nearest-neighbour upsampling along both spatial axes, channels last.

Public methods:

- `backward(g: 'jax.Array')`
- `forward(x: 'jax.Array', training: 'bool' = True)`
- `params()`

### VectorQuantizer


*Module:* `dense_armor.utility.deep.neural.quantize`

Nearest-code quantizer with the straight-through gradient.

Public methods:

- `backward(g: 'jax.Array')`
- `decode_indices(idx: 'jax.Array')`
- `encode_indices(z_e: 'jax.Array')`
- `forward(z_e: 'jax.Array', training: 'bool' = True)`
- `loss()`
- `params()`

### VQMemory


*Module:* `dense_armor.utility.deep.neural.consolidation.vq`

Vector-quantised memory (van den Oord et al. 2017, Eq. 1-4 p. 3).

Public methods:

- `backward(g: 'jnp.ndarray')`
- `decode(z_q: 'jnp.ndarray')`
- `encode(x: 'jnp.ndarray')`
- `fit(x: 'jnp.ndarray', n_steps: 'int' = 5, lr: 'float' = 0.01)`
- `forward(x: 'jnp.ndarray')`
- `params()`
- `quantize(z_e: 'jnp.ndarray')`
- `update_codebook()`

## Infrastructure

### SharedArray


*Module:* `dense_armor.utility.infrastructure.data.shared`

A numeric numpy array in one shared memory block.

Public methods:

- `close()`
- `unlink()`

### SharedDict


*Module:* `dense_armor.utility.infrastructure.data.shared`
*Bases:* Mapping

Separate key and value buffers with offsets.

Public methods:

- `close()`
- `get(key, default=None)`
- `items()`
- `keys()`
- `unlink()`
- `values()`

### SharedList


*Module:* `dense_armor.utility.infrastructure.data.shared`

A flat buffer of pickled items plus int64 offsets.

Public methods:

- `close()`
- `unlink()`

### SharedStrings


*Module:* `dense_armor.utility.infrastructure.data.shared`

A list of strings as one UTF-8 blob plus int64 offsets.

Public methods:

- `close()`
- `unlink()`

## Linear

### BayesianLinearRegression


*Module:* `dense_armor.utility.online.linear.bayes`
*Bases:* Regressor

Online Bayesian linear regression with a conjugate Gaussian prior.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: dict, y: Any, t: float | None = None)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `predict_many(X, t=None, return_std=False)`
- `predict_one(x: dict, t: float | None = None, return_std: bool = False, return_estimate: bool = False)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### LinearRegression


*Module:* `dense_armor.utility.online.linear.glm`
*Bases:* Regressor

Online linear regression with squared loss.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `init_state(dim: int = 1)`
- `learn_many(X, y, t=None)`
- `learn_one(x: dict, y: Any, t: float | None = None)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `predict_many(X, t=None, return_std=False)`
- `predict_one(x: dict, t: float | None = None, return_std: bool = False)`
- `set_params(**params: Any)`
- `state_dict()`
- `step(state: tuple, v: Any)`
- `window_samples(seconds: float)`

### LogisticRegression


*Module:* `dense_armor.utility.online.linear.glm`
*Bases:* Classifier

Online binary logistic regression with log loss.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `init_state(dim: int = 1)`
- `learn_many(X, y, t=None)`
- `learn_one(x: dict, y: Any, t: float | None = None)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `predict_many(X, t=None)`
- `predict_one(x: dict, t: float | None = None, return_std: bool = False)`
- `predict_proba_many(X, t=None)`
- `predict_proba_one(x: dict, t: float | None = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `step(state: tuple, v: Any)`
- `window_samples(seconds: float)`

### PayloadEstimator


*Module:* `dense_armor.utility.online.linear.payload`
*Bases:* Regressor

Recursive least squares estimate of a point mass on a link.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, y, t=None)`
- `learn_one(x: Any, y: Any = None, t: float | None = None)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `predict_many(X, t=None, return_std=False)`
- `predict_one(x: dict | None = None, t: float | None = None, return_std: bool = False)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

## ML

### BaggingClassifier


*Module:* `dense_armor.ml.ensemble`
*Bases:* MLBase

Bagging: one model per bootstrap sample, probabilities averaged.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None)`
- `fit_transform(X: 'Any', y: 'Any' = None, **fit_params: 'Any')`
- `get_params(deep: bool = True)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `predict(X)`
- `predict_proba(X)`
- `score(X: 'Any', y: 'Any', sample_weight: 'Any' = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### BatchDecisionTreeClassifier


*Module:* `dense_armor.ml.tree`
*Bases:* MLBase

Binary decision tree with the gini impurity, grown recursively on the whole table.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None)`
- `fit_transform(X: 'Any', y: 'Any' = None, **fit_params: 'Any')`
- `get_params(deep: bool = True)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `predict(X)`
- `predict_proba(X)`
- `score(X: 'Any', y: 'Any', sample_weight: 'Any' = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### BayesianLinearRegression


*Module:* `dense_armor.ml.linear`
*Bases:* MLBase

Bayesian linear regression with a Gaussian prior, in closed form.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None)`
- `fit_transform(X: 'Any', y: 'Any' = None, **fit_params: 'Any')`
- `get_params(deep: bool = True)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `predict(X, return_std: 'bool' = False)`
- `score(X: 'Any', y: 'Any', sample_weight: 'Any' = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### BoostingClassifier


*Module:* `dense_armor.ml.ensemble`
*Bases:* MLBase

AdaBoost with the multiclass weights SAMME (Zhu et al. 2009).

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None)`
- `fit_transform(X: 'Any', y: 'Any' = None, **fit_params: 'Any')`
- `get_params(deep: bool = True)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `predict(X)`
- `predict_proba(X)`
- `score(X: 'Any', y: 'Any', sample_weight: 'Any' = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### ColumnMoments


*Module:* `dense_armor.ml.stats`
*Bases:* MLBase

The moments of every column of a matrix.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None)`
- `fit_transform(X: 'Any', y: 'Any' = None, **fit_params: 'Any')`
- `get_params(deep: bool = True)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `score(X: 'Any', y: 'Any', sample_weight: 'Any' = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### Correlation


*Module:* `dense_armor.ml.stats`
*Bases:* Covariance

Pearson correlation of two channels over the whole dataset.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None)`
- `fit_transform(X: 'Any', y: 'Any' = None, **fit_params: 'Any')`
- `get_params(deep: bool = True)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `score(X: 'Any', y: 'Any', sample_weight: 'Any' = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### Covariance


*Module:* `dense_armor.ml.stats`
*Bases:* MLBase

Sample covariance (``n - 1``) and Pearson correlation of two channels.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None)`
- `fit_transform(X: 'Any', y: 'Any' = None, **fit_params: 'Any')`
- `get_params(deep: bool = True)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `score(X: 'Any', y: 'Any', sample_weight: 'Any' = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### DecisionTreeClassifier


*Module:* `dense_armor.ml.tree`
*Bases:* _Tree

CART classification tree: splits that maximise the Gini decrease.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, sample_weight=None)`
- `fit_transform(X: 'Any', y: 'Any' = None, **fit_params: 'Any')`
- `get_params(deep: bool = True)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `predict(X)`
- `predict_proba(X)`
- `score(X: 'Any', y: 'Any', sample_weight: 'Any' = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### DecisionTreeRegressor


*Module:* `dense_armor.ml.tree`
*Bases:* _Tree

CART regression tree: splits that maximise the variance reduction; leaves hold the mean.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None)`
- `fit_transform(X: 'Any', y: 'Any' = None, **fit_params: 'Any')`
- `get_params(deep: bool = True)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `predict(X)`
- `score(X: 'Any', y: 'Any', sample_weight: 'Any' = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### DenseFlow


*Module:* `dense_armor.ml.vision`
*Bases:* MLBase

Horn-Schunck optical flow between two frames.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None)`
- `fit_transform(X: 'Any', y: 'Any' = None, **fit_params: 'Any')`
- `get_params(deep: bool = True)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `score(X: 'Any', y: 'Any', sample_weight: 'Any' = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### ExponentiallyWeightedAverage


*Module:* `dense_armor.ml.ensemble`
*Bases:* MLBase

Weights ``exp(-eta L_m)`` from the total loss of each model on the data (Hedge).

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None)`
- `fit_transform(X: 'Any', y: 'Any' = None, **fit_params: 'Any')`
- `get_params(deep: bool = True)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `predict(X)`
- `score(X: 'Any', y: 'Any', sample_weight: 'Any' = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### ExtraTreesRegressor


*Module:* `dense_armor.ml.forest`
*Bases:* MLBase

Extremely randomised trees: the whole sample and a random threshold at each split.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None)`
- `fit_transform(X: 'Any', y: 'Any' = None, **fit_params: 'Any')`
- `get_params(deep: bool = True)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `predict(X)`
- `score(X: 'Any', y: 'Any', sample_weight: 'Any' = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### GlobalMotion


*Module:* `dense_armor.ml.vision`
*Bases:* MLBase

Global shift between two frames by phase correlation.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None)`
- `fit_transform(X: 'Any', y: 'Any' = None, **fit_params: 'Any')`
- `get_params(deep: bool = True)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `score(X: 'Any', y: 'Any', sample_weight: 'Any' = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### LinearRegression


*Module:* `dense_armor.ml.linear`
*Bases:* MLBase

Ridge regression in closed form.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None)`
- `fit_transform(X: 'Any', y: 'Any' = None, **fit_params: 'Any')`
- `get_params(deep: bool = True)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `predict(X)`
- `score(X: 'Any', y: 'Any', sample_weight: 'Any' = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### LogisticRegression


*Module:* `dense_armor.ml.linear`
*Bases:* MLBase

Binary logistic regression by Newton steps on the mean log loss.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None)`
- `fit_transform(X: 'Any', y: 'Any' = None, **fit_params: 'Any')`
- `get_params(deep: bool = True)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `predict(X)`
- `predict_proba(X)`
- `score(X: 'Any', y: 'Any', sample_weight: 'Any' = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### MLBase


*Module:* `dense_armor.ml.base`
*Bases:* SklearnCompat, Root

Root of the classes: parameters, ``clone`` and ``state_dict``.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None)`
- `fit_transform(X: 'Any', y: 'Any' = None, **fit_params: 'Any')`
- `get_params(deep: bool = True)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `score(X: 'Any', y: 'Any', sample_weight: 'Any' = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### Moments


*Module:* `dense_armor.ml.stats`
*Bases:* MLBase

Mean, variance (``n - 1``), skewness, excess kurtosis, min, max, range.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None)`
- `fit_transform(X: 'Any', y: 'Any' = None, **fit_params: 'Any')`
- `get_params(deep: bool = True)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `score(X: 'Any', y: 'Any', sample_weight: 'Any' = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### MotionMask


*Module:* `dense_armor.ml.vision`
*Bases:* MLBase

Mask of the pixels whose flow magnitude is large, from the two components of a flow field.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None)`
- `fit_transform(X: 'Any', y: 'Any' = None, **fit_params: 'Any')`
- `get_params(deep: bool = True)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `score(X: 'Any', y: 'Any', sample_weight: 'Any' = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### PCA


*Module:* `dense_armor.ml.vision`
*Bases:* MLBase

Principal components of a table: the top ``k`` eigenvectors of its covariance.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None)`
- `fit_transform(X: 'Any', y: 'Any' = None, **fit_params: 'Any')`
- `get_params(deep: bool = True)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `score(X: 'Any', y: 'Any', sample_weight: 'Any' = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `transform(X)`
- `window_samples(seconds: float)`

### RandomForestClassifier


*Module:* `dense_armor.ml.forest`
*Bases:* MLBase

Random forest: trees on bootstrap samples with ``sqrt(d)`` random features per split.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None)`
- `fit_transform(X: 'Any', y: 'Any' = None, **fit_params: 'Any')`
- `get_params(deep: bool = True)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `predict(X)`
- `predict_proba(X)`
- `score(X: 'Any', y: 'Any', sample_weight: 'Any' = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### ShrinkageCovariance


*Module:* `dense_armor.ml.stats`
*Bases:* MLBase

Linear shrinkage toward ``(1/p) I`` on the sample covariance of all rows.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None)`
- `fit_transform(X: 'Any', y: 'Any' = None, **fit_params: 'Any')`
- `get_params(deep: bool = True)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `score(X: 'Any', y: 'Any', sample_weight: 'Any' = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `window_samples(seconds: float)`

### TextVectorizer


*Module:* `dense_armor.ml.preprocessing`
*Bases:* MLBase

TF-IDF matrix of a corpus with a fixed vocabulary.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None)`
- `fit_transform(X: 'Any', y: 'Any' = None, **fit_params: 'Any')`
- `get_params(deep: bool = True)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `score(X: 'Any', y: 'Any', sample_weight: 'Any' = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `transform(documents)`
- `window_samples(seconds: float)`

## Optimizers

### AdaGrad


*Module:* `dense_armor.utility.online.optim.adagrad`

AdaGrad with per-coordinate adaptive step sizes.

Public methods:

- `init_state(dim: int)`
- `step(weights: numpy.ndarray, gradient: numpy.ndarray)`
- `update(state: dict[str, Any], weights: jax.Array, gradient: jax.Array)`

### Adam


*Module:* `dense_armor.utility.online.optim.adam`

Adam with bias-corrected moment estimates.

Public methods:

- `init_state(dim: int)`
- `step(weights: numpy.ndarray, gradient: numpy.ndarray)`
- `update(state: dict[str, Any], weights: jax.Array, gradient: jax.Array)`

### AdamW


*Module:* `dense_armor.utility.online.optim.adamw`
*Bases:* Adam

Adam with weight decay decoupled from the gradient.

Public methods:

- `init_state(dim: int)`
- `step(weights: numpy.ndarray, gradient: numpy.ndarray)`
- `update(state: dict[str, Any], weights: Any, gradient: Any)`

### FTRLProximal


*Module:* `dense_armor.utility.online.optim.ftrl`

FTRL with a quadratic + L1 + L2 regularizer.

Public methods:

- `init_state(dim: int)`
- `step(weights: numpy.ndarray, gradient: numpy.ndarray)`
- `update(state: dict[str, Any], weights: jax.Array, gradient: jax.Array)`

### OnlineNewtonStep


*Module:* `dense_armor.utility.online.optim.newton`

Second-order online update with a rank-one running Hessian.

Public methods:

- `from_bounds(G: float, D: float, alpha: float)`
- `init_state(dim: int)`
- `step(weights: numpy.ndarray, gradient: numpy.ndarray)`
- `update(state: dict[str, Any], weights: jax.Array, gradient: jax.Array)`

### RMSProp


*Module:* `dense_armor.utility.online.optim.rmsprop`

RMSProp with an exponentially decaying average of squared gradients.

Public methods:

- `init_state(dim: int)`
- `step(weights: numpy.ndarray, gradient: numpy.ndarray)`
- `update(state: dict[str, Any], weights: jax.Array, gradient: jax.Array)`

### SGD


*Module:* `dense_armor.utility.online.optim.sgd`

Gradient descent with optional momentum and Nesterov momentum.

Public methods:

- `init_state(dim: int)`
- `step(weights: numpy.ndarray, gradient: numpy.ndarray)`
- `update(state: dict[str, Any], weights: jax.Array, gradient: jax.Array)`

## Time series

### CycleResidual


*Module:* `dense_armor.utility.online.timeseries.cycle`
*Bases:* Transformer

Cycle residual of a repetitive scalar stream.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, t=None)`
- `learn_one(x: dict, y: Any = None, t: float | None = None)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `set_params(**params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: dict, t: float | None = None)`
- `window_samples(seconds: float)`

### GARCH11


*Module:* `dense_armor.utility.online.timeseries.garch`
*Bases:* Transformer

GARCH(1, 1) conditional variance with fixed parameters.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, t=None)`
- `learn_one(x: dict, y: Any = None, t: float | None = None)`
- `load_state_dict(state: dict[str, Any])`
- `merge(other: 'GARCH11')`
- `mutate(new_attrs: dict[str, Any])`
- `score_one(x: dict)`
- `set_params(**params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: dict, t: float | None = None)`
- `update(x: dict, y: Any = None, t: float | None = None)`
- `window_samples(seconds: float)`

### InnovationAdaptiveKalman


*Module:* `dense_armor.utility.online.timeseries.kalman`
*Bases:* KalmanFilter

Adaptive Kalman filter after Popescu et al. (2026).

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `init_state(dim: int = 1)`
- `learn_many(X, t=None)`
- `learn_one(x: dict, y: Any = None, t: float | None = None)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `predict_one(x: dict | None = None, t: float | None = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `step(state: tuple, v: Any)`
- `transform_many(X, t=None)`
- `transform_one(x: dict, t: float | None = None)`
- `window_samples(seconds: float)`

### KalmanFilter


*Module:* `dense_armor.utility.online.timeseries.kalman`
*Bases:* Transformer

Linear Kalman filter with fixed matrices.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `init_state(dim: int = 1)`
- `learn_many(X, t=None)`
- `learn_one(x: dict, y: Any = None, t: float | None = None)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `predict_one(x: dict | None = None, t: float | None = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `step(state: tuple, v: Any)`
- `transform_many(X, t=None)`
- `transform_one(x: dict, t: float | None = None)`
- `window_samples(seconds: float)`

### OnlineAR


*Module:* `dense_armor.utility.online.timeseries.ar`
*Bases:* Transformer

Online univariate autoregressive model of order ``order``.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `init_state(dim: int = 1)`
- `learn_many(X, t=None)`
- `learn_one(x: dict, y: Any = None, t: float | None = None)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `predict_one(x: dict, t: float | None = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `step(state: tuple, v: Any)`
- `transform_many(X, t=None)`
- `transform_one(x: dict, t: float | None = None)`
- `window_samples(seconds: float)`

### OnlineVAR


*Module:* `dense_armor.utility.online.timeseries.ar`
*Bases:* Transformer

Online vector autoregressive model, TIRSO of Zaman et al. 2019.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `learn_many(X, t=None)`
- `learn_one(x: dict, y: Any = None, t: float | None = None)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `predict_one(x: dict, t: float | None = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `transform_many(X, t=None)`
- `transform_one(x: dict, t: float | None = None)`
- `window_samples(seconds: float)`

### VBAdaptiveKalman


*Module:* `dense_armor.utility.online.timeseries.kalman`
*Bases:* KalmanFilter

Variational Bayesian adaptive Kalman filter.

Public methods:

- `clone(new_params: dict[str, Any] | None = None, include_attributes: bool = False)`
- `describe()`
- `fit(X, y=None, t=None, epochs: int = 1, shuffle: bool = False, seed: int | None = None)`
- `get_params(deep: bool = True)`
- `init_state(dim: int = 1)`
- `learn_many(X, t=None)`
- `learn_one(x: dict, y: Any = None, t: float | None = None)`
- `load_state_dict(state: dict[str, Any])`
- `mutate(new_attrs: dict[str, Any])`
- `predict_one(x: dict | None = None, t: float | None = None)`
- `set_params(**params: Any)`
- `state_dict()`
- `step(state: tuple, v: Any)`
- `transform_many(X, t=None)`
- `transform_one(x: dict, t: float | None = None)`
- `window_samples(seconds: float)`
