# `$ internal architecture`

Reference for people who want to contribute to the code, not for those who only want to install and use the package — for that see the [main README](../README.md). This page maps `core/`, the library: nine groups, each split into subfolders by topic.

---

## `$ core/`

| group | what it holds |
|---|---|
| `online/` | Online learning: models, optimizers, detectors and protection that learn one sample at a time (21 subfolders) |
| `deep/` | Deep learning: layers, networks and training (`neural/`, including the language networks with the static key-value cache) |
| `signals/` | Streams, statistics, preprocessing, vision and datasets (5 subfolders) |
| `audio/` | Sound as a value: WAV files, windows, STFT, generators, filters, NMF and streams |
| `graph/` | Graph engine: graph, GFT, spectral filters, Chebyshev, ontology, execution DAG |
| `finance/` | Building blocks for trading systems: schedules, pre-trade limits, feeds, audit |
| `evaluation/` | Metrics and test-then-train evaluation (`evaluate/`, `metrics/`) |
| `robot/` | Control and safety filters (`control/`) |
| `infrastructure/` | Pipeline pieces, shared data containers and small tools (`compose/`, `data/`) |

### `online/`

| subfolder | what it does |
|---|---|
| `learn` | calibration, metric learning, online classifiers, online dynamics |
| `optim` | online optimizers, as stateful objects and as pure JAX forms (SGD, Adam, AdamW, AdaGrad, RMSProp, FTRL, Newton) |
| `linear` | online linear models: GLM, Bayesian linear regression, payload estimation |
| `timeseries` | online time-series models: autoregression, Kalman filtering, GARCH, cycle residual |
| `cluster` | `kmeans`, `denstream` |
| `tree` | incremental decision trees: Hoeffding, EFDT, FIMT, SGT, Mondrian, adaptive, batch |
| `forest` | forests of online trees: AMF, ARF, OXT |
| `ensemble` | bagging, boosting, EWA, stacking, voting |
| `bandit` | bandit policies, exploration and exploitation on a reward stream: UCB, Bayes-UCB, LinUCB, Thompson sampling, EXP3 |
| `active` | active learning strategies for evolving data streams |
| `facto` | factorization machines: FM (Rendle 2010), FFM (Juan 2016), FwFM (Pan 2018), HOFM (Blondel 2016) |
| `reco` | recommenders: baseline, biased MF (Paterek 2007, Koren 2009), Funk MF |
| `rules` | rule-based learners (`amrules`) |
| `bayes` | Naive Bayes classifiers on counts and categories: Bernoulli, categorical, complement, multinomial |
| `neighbors` | exact nearest-neighbour classifiers and regressors on a sliding window |
| `multiclass` | strategies to turn a binary classifier into a multiclass one |
| `multioutput` | multi-output classifiers and regressors (one prediction per target) |
| `proba` | conjugate distributions updated one observation at a time: beta, Gaussian, multinomial |
| `anomaly` | anomaly detectors: HST, LODA, LOF, one-class SVM, online isolation forest, Mahalanobis, streaming scores, robust filters, curvature, resonance search |
| `drift` | drift detectors: CUSUM, Page-Hinkley, ADWIN, KSWIN, BOCPD |
| `protect` | protection around a model: arbiter, healing, stable-frame filter, streaming arbiter, `orca` |


### `deep/`

`deep/neural/` is the only subfolder. Everything is a plain layer with a `forward` and a `backward`, on a batch or on one sample.

| theme | module | what it does |
|---|---|---|
| core | `params` | a parameter that holds its gradient, and the update through an optimizer |
| core | `init` | He and Xavier initialisations for rectifier networks |
| core | `layers` | `Dense`, `Embedding`, `Dropout`, `Residual`: the layers with weights |
| core | `chain` | chains and branches of layers (`Sequential`, `Parallel`) |
| core | `models` | two classic architectures: the MLP and the residual MLP |
| activations | `activations` | `Relu`, `PRelu`, `Sigmoid`
| activations | `activations_more` | GELU, SiLU, ELU, LeakyReLU, Softplus |
| activations | `glu` | feed-forward layers with a gated lU |
| normalisation | `norm` | BatchNorm and LayerNorm |
| normalisation | `norm_more` | GroupNorm and InstanceNorm
| normalisation | `norm_rms` | root mean square layer normalisation |
| losses | `losses` | MSE, softmax cross-entropy, BCE with
| losses | `losses_more` | L1, label-smoothed cross-entropy, KL divergence, hinge |
| losses | `losses_lm` | next-token loss for a causal lang
| losses | `matryoshka` | the Matryoshka Representation Learning loss (Kusupati et al. 2022) |
| convolution | `conv` | convolution and pooling layers |
| convolution | `conv_more` | three-dimensional, depthwise and separable convolutions |
| convolution | `convtranspose` | transposed convolutions ing |
| recurrent | `recurrent` | RNN, GRU, LSTM, with backpropagation through time |
| recurrent | `bidirectional` | bidirectional recurrent la
| recurrent | `seq2seq` | a fixed-horizon sequence encoder and decoder |
| attention | `attention` | scaled dot-product, multi-head
| attention | `cross_attention` | queries from one stream, keys and values from another |
| attention | `gated_attention` | a gate on a signal: lookn it matters |
| attention | `latent_attention` | read an input of any length into a fixed set of latents |
| attention | `bottleneck` | two streams exchange informat
| attention | `positions`, `rotary` | learned positional embedding; rotary position embedding of queries and keys |
| transformer | `decoder` | decoder block of the Transformion 3.1) |
| transformer | `rotary_decoder` | RMSNorm, rotary causal attention and a gated feed-forward |
| transformer | `moe` | sparse mixture of experts, dispatc
| transformer | `generate` | greedy and sampled generation for a causal language model |
| transformer | `test_time` | test-time compute: generate |
| representation | `autoencoder` | an encoder, a decoder, and the reconstruction error |
| representation | `quantize` | the vector quantizer (van
| training | `trainer` | the standard training loop: mini-batches, schedules, weight decay |
| training | `compiled` | a compiled training step: forwaray |
| training | `schedules`, `schedules_more` | learning-rate schedules; schedules with warmup |
| training | `clip` | gradient clipping by global L2 norm
| training | `batching`, `stream` | mini-batch generator; a data stream that serves batches without loading the dataset |
| training | `data_parallel` | linear scaling of the learnaining |
| training | `background` | a background learner: the main program is never blocked |
| training | `gradcheck` | gradient check against central
| continual | `continual` | continual-learning baselines: EWC, GEM, A-GEM |
| continual | `generative_replay` | deep generative replay
| continual | `hedge` | Hedge Backpropagation (Sahoo et al. 2018) |
| online | `online` | online estimators of a network: one
| online | `online_vector` | online learner for vector targets |
| online | `online_autoencoder` | an autoencoder trained oector |
| bridge | `adapter` | a layer seen as a `Transformer` of the library |

### `audio/`

Sound as a value (`Audio`: samples, sample rate, start time), then the tools to read it, filter it, analyse it and stream it.

| area | module | what it does |
|---|---|---|
| value and files | `signal` | the value type for sound: samples, sample rate and start time |
| value and files | `wav` | reading and writing WAV files  float 32) |
| value and files | `streams` | audio streams: from a WAV file and from a live input |
| value and files | `adapter` | one adapter that opens anynd |
| value and files | `_state` | a `Root` whose `state_dict` also holds the private attributes in `_state_attrs` |
| spectrum | `windows` | rectangular, Hann, Hamming, Black
| spectrum | `fourier` | short-time Fourier transform, its inverse by overlap-add, and a streaming form |
| spectrum | `spectral` | spectral features, one value per
| spectrum | `perceptual` | mel scale, mel filterbank, mel spectrogram, MFCC |
| spectrum | `features` | frame features: the magnitude spnds |
| spectrum | `nmf` | non-negative matrix factorisation and source separation |
| filters | `fir` | convolution, the FIR filter and the de
| filters | `biquad` | biquad filters: Audio EQ Cookbook formulas and a Direct Form 1 filter |
| filters | `allpole` | the all-pole filter and the reflect stable |
| filters | `diff_filter` | differentiable filters in JAX: cookbook low-pass, biquad layer, time-varying all-pole |
| filters | `resample` | resampling by an arbitrary, possi
| dynamics | `dynamics` | a compressor, a lookahead limiter and a soft clipper with oversampling |
| dynamics | `clipping`, `blamp` | hard clipping; four-poid triangle wave |
| dynamics | `envelope` | the analytic signal and an envelope follower with attack and release |
| synthesis | `generators` | oscillators, white noise, KarDSR |
| effects | `modulation` | chorus and phaser |
| effects | `reverb` | Schroeder comb and all-pass filters
| effects | `timescale` | phase vocoder (time-stretch, pitch-shift), PSOLA and a harmonizer |
| analysis | `pitch` | fundamental frequency by the YIN me
| analysis | `pitch_units` | frequency, cents and note coordinates in equal temperament |
| analysis | `onset` | onset detection: spectral flux and
| analysis | `events` | from a sound to an alarm: block features, a causal Hampel score and a drift detector |

### `graph/`

| area | module | what it does |
|---|---|---|
| structure | `core` | weighted undirected graph and Laplacians (`Graph`) |
| structure | `_light` | small directed multigraph: the paph` and `ExecutionDAG` use |
| structure | `signal` | `GraphSignal`: a `Signal` whose values live on the graph nodes |
| spectral | `spectral` | spectral decomposition of the gr
| spectral | `fourier` | Graph Fourier Transform |
| spectral | `filters` | spectral filters (heat kernel, loter |
| spectral | `wavelet` | graph spectral wavelets from Chebyshev polynomials of even and odd order |
| learning and analysis | `learning` | informed graph learnals and domain knowledge |
| learning and analysis | `adacgp` | AdaCGP: online estimation of the graph shift operator from a time series |
| learning and analysis | `recovery` | recovery of a time-amic graph |
| learning and analysis | `change` | structural change between two graphs |
| learning and analysis | `anomaly` | PageRank, random wal score |
| learning and analysis | `causal` | `CausalGraph`: a DAG with the DoWhy do-calculus |
| ontology | `ontology` | `TypedGraph`: a graph with node  (`Constraint`, `ConstraintViolation`) |
| ontology | `actions` | `Action`: a STRIPS action with preconditions and effects |
| execution | `dag` | `ExecutionDAG`: a deterministic netw
| execution | `dataflow` | describe where each signal comes from, then build the DAG |
| execution | `gate` | `GateNode`: an `AnomalyGate` as a n
| execution | `provenance` | a JSONL log of every run of a DAG node |

### `finance/`

Building blocks for trading systems, from the market feed to the audit trail.

| area | module | what it does |
|---|---|---|
| market data | `itch` | decode one Nasdaq TotalView-ITCH 5.0 order message |
| market data | `feed` | read a file of framed ITCH 5.0 or
| market data | `synthetic` | synthetic ITCH 5.0 order messages, deterministic from SHA-256 |
| orders | `schedule` | schedules that split a parent orde
| orders | `pretrade` | pre-trade limits for one order, from SEC Rule 15c3-5 (c)(1) (`PreTradeChecker`, `Decision`) |
| orders | `fix_gateway` | a FIX 4.4 initiator that sends  reports |
| infrastructure | `ring` | a ring buffer with preallocated storage for one producer and one consumer |
| infrastructure | `config` | one frozen dataclass per blodeConfig`, `RingConfig`, `PathsConfig`) |
| audit | `decision_log` | a decision log with a SHA-256 hash chain, one JSON line per record |
| audit | `audit_db` | the same decision log in a PostgreSchain |
| commands | `demo` | synthetic feed, checker, log, one JSON line on stdout |
| commands | `status` | one JSON object with the configura

### `signals/`

| subfolder | what it does |
|---|---|
| `stream` | iterators of `(Signal, y)` pairs in time order: arrays, CSV, HDF5/NetCDF (`iodat`), ROS bags, merge, shuffle |
| `stats` | streaming statistics and sketches: moments, quantiles, robust estimates, shrinkage, similarity, dependence, mixtures, Student-t, stable laws, sliding DFT |
| `preprocessing` | imbalance, joints, `metro` (logarithmic scaler), scale, select, text, vectorizer |
| `vision` | native online vision: features, landmarks, motion, patches, quality, registration, tracking, video input |
| `datasets` | ready-made streams for tests and experiments: `casper`, `drift_stream`, `synthetic_arm` |

## `$ ml/'  Machine Learning
Sits at `dense_armor/ml/`, outside `core/`; it is meant to move under it. Machine learning algorithms: the algorithm of the paper fitted on the whole dataset. All classes derive from `MLBase` (`base.py`); `_sklearn_compat.py` (`SklearnCompat`) makes them work with scikit-learn tools.


| module | what it does |
|---|---|
| `base` | base class of the machine learning algorithms (`MLBase`) |
| `_sklearn_compat` | compatibility of the batch classes with scikit-learn tools |
| `linear` | linear models fitted on the whole table: `LinearRegression`, `LogisticRegression`, `BayesianLinearRegression` |
| `tree` | decision trees and Mondrian forests grown on the whole table |
| `forest` | forests of trees grown on bootstrap samples or with random thresholds: `RandomForestClassifier`, `ExtraTreesRegressor` |
| `ensemble` | `BaggingClassifier`, `BoostingClassifier`, `ExponentiallyWeightedAverage` |
| `bayes` | Naive Bayes fitted by counting: `MultinomialNB`, `BernoulliNB`, `CategoricalNB` |
| `neighbors` | k-nearest-neighbour models on the whole training table: `KNNClassifier`, `KNNRegressor` |
| `multiclass` | multiclass strategies on a binary classifier: `OneVsRestClassifier`, `OneVsOneClassifier`, `OutputCodeClassifier` |
| `multioutput` | several targets on a single-target model: `PerOutputClassifier`, `PerOutputRegressor`, `ClassifierChain` |
| `proba` | distributions fitted on the whole sample, conjugate posterior in closed form: `Beta`, `Gaussian`, `Multinomial` |
| `anomaly` | anomaly detection on a whole series or table: `HampelFilter`, `TukeyScorer`, `ChauvenetScorer` |
| `drift` | change detection on a whole series: `CUSUMDriftDetector` |
| `stats` | statistics on the whole table: covariance, correlation, moments, medians, shrinkage |
| `preprocessing` | scalers, selectors, joint derivatives and text on the whole table: `StandardScaler`, `EWScaler`, `RobustScaler` |
| `vision` | flow between two frames, global shift, reduction of a table of features: `DenseFlow`, `MotionMask`, `GlobalMotion` |
| `optim`, `rules` | "the classes of the family that have no batch form": the files hold only this docstring, no classes |

### `evaluation/`, `robot/`, `infrastructure/`

| subfolder | what it does |
|---|---|
| `evaluation/evaluate` | online evaluation helpers: prequential, model selection, diagnostics, events |
| `evaluation/metrics` | online metrics for streams, robots and LLMs: classification, regression, ROC AUC, rolling windows |
| `robot/control` | `cbf_filter`, `kinematic_controller`, `rate_limiter`, `trajectory` |
| `infrastructure/compose` | small transformers to select, drop and reshape channels |
| `infrastructure/data` | shared data containers |

---

## `$ old paths`

The old paths remain valid (`_compat.py`), for example `dense_armor.utility.bandit` or `dense_armor.utility.online.bandit`, and so do the old engine paths, for example `dense_armor.core.engine`. Up to this version the library was called `utility/` and the shielding engine `core/`: the two names have been swapped.

---

## `$ note`

The shielding modules (`Armor`, `Orca` and their engine) are a separate component and are not this program: the program is the `core/` library.

- Old-paths sentence: dense_armor/utility/ still exists on disk with online/, deep/, engine.py and the rest. The "names swapped" sentence is therefore about the old import paths, and I kept it as it was. I did not verify the dense_armor.core.engine path, so check that line before you commit.
- Remaining checks: the one-line descriptions come from docstrings, not from reading each module. anomaly, learn, protect and preprocessing have no docstring, so their rows are my summary of the module names. I did not run any of the code.

