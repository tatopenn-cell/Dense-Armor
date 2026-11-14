# Changelog

Formato basato su [Keep a Changelog](https://keepachangelog.com/it/1.0.0/).

## [Unreleased]

## [1.6.0] - 2026-11-14

### Added
- The ontology, the package `dense_armor.core.graph`: a weighted `Graph` with its Laplacians, `GraphSignal`, the Fourier transform
  (`GraphFourier`) and the spectral filters (`SpectralFilter`, `ChebyshevFilter`, `SpectralWavelet`); a `TypedGraph` with
  `Constraint` rules and `Action` with preconditions, validated on the subgraph an action touches (`apply_action`); an
  `ExecutionDAG` of estimators with `Dataflow` and `GateNode`; a `Provenance` log chained by SHA-256 (`head`, `broken_at`,
  `verify`); the scores `page_rank`, `random_walk_restart`, `root_cause_walk` and `graph_anomaly_score`;
  `structural_change`; `CausalGraph` (DoWhy). Each part follows one paper (`REFERENCES.md`).
- Graphs from signals: `AdaCGP`, an estimator that learns the weights of a graph one sample at a time (it follows the estimator
  contract: `budget_s`, `memory_class`, `state_dict`), `recover_graph_signal` (a time-varying signal on a dynamic graph),
  `learn_graph` and `weights_from_drops` (a graph from smooth signals and domain knowledge). Golden files in `test/graph/`.
- The documentation section "Ontology" (11 pages, three figures): types, rules and actions, spectrum, filters, the execution DAG,
  provenance, anomaly scores, recovery, learning, following a graph online, causality. The extras `onto`, `graph` and `causal`.
- `docs/api/general.md`: the sections Audio, Graph (ontology) and Finance, and 15 classes that were missing (449 classes in
  all). `ARCHITECTURE.md`: sections 2.33 audio, 2.34 graph and 2.35 finance. The README lists the audio section and the ontology.
- `REFERENCES.md` cites 236 works (48 more: the papers of the audio section and of the ontology), and `.zenodo.json` lists the
  11 new arXiv identifiers.
- `docs-check.yml`: a pull request that touches the documents runs the strict build of the site.
- Audio, differentiable filters (`dense_armor/core/audio/diff_filter.py`): `jax_lowpass`, `jax_biquad`, `JaxBiquad`
  (a layer in the pattern of `core/deep/neural`) and `jax_allpole_time_varying`, with the gradients of
  Yu and others (2024); equal to the NumPy filters to 1e-15 in 64 bits and 3e-7 in 32 bits. A page
  (`docs/audio/differentiable.md`) and a golden file.
- Audio, time and pitch (`dense_armor/core/audio/timescale.py`): `time_stretch` (a phase vocoder), `pitch_shift`,
  `psola_time_stretch` and `Harmonizer`; the pitch of the output is kept to a few cents, the level is kept, and a rate of 1 gives
  the input back. Multichannel live input (`AudioLiveStream(channels=...)`, `Audio.channel`, `run_channels`). A page
  (`docs/audio/timescale.md`) and two more golden files.
- Audio, effects (`dense_armor/core/audio/dynamics.py`, `reverb.py`, `modulation.py`): `Compressor` and `gain_curve`, a lookahead
  `Limiter` (no sample above the ceiling), `SoftClip` with oversampling, `CombFilter`, `AllpassFilter` and `SchroederReverb`
  (the reverberation time measured on the impulse response is within 0.7% of the one asked), `reverberation_time`, `Chorus` and
  `Phaser`. Filters and smoothing agree with `scipy.signal` to rounding. A page with figures (`docs/audio/effects.md`) and two
  more golden files.
- Audio, perceptual analysis (`dense_armor/core/audio/perceptual.py`, `spectral.py`, `onset.py`, `envelope.py`): `hz_to_mel`,
  `mel_to_hz`, `mel_filterbank`, `mel_spectrogram`, `mfcc`, `spectral_centroid`, `spectral_rolloff`, `spectral_flatness`,
  `spectral_flux`, `onset_detect`, `hilbert` and `envelope_follower`. They agree with `librosa` and `scipy` (largest difference
  between 0 and 1.5e-7 on the filterbank, the mel spectrogram, the MFCC, the spectral features and the analytic signal; the onsets
  agree with `librosa` at F1 0.945). A page with figures (`docs/audio/perceptual.md`) and a golden file (`test/audio/golden/`).
- The dashboard works on the visitor's own files: a CSV of prices into the chain (with a report to download as CSV),
  a volume history into the TWAP and VWAP plan, and a CSV of orders into the pre-trade checker (with the decisions and
  the hash-chained log to download). Each page has a sample file and says what is wrong with a file, naming the row or
  the column. A notice on every page says that the demo runs on synthetic data and that the results are not for trading
  decisions. The code is `tools/dashboard/own_data.py`.
- `DecisionLog.append_many`, which verifies the file once for a whole batch (`append` is now one call of it).
- `AudioInput(scaling=...)`: `AMRulesRegressor`, `OneClassSGD` (standardised band by band) and `AggregatedMondrianForestRegressor`,
  `OnlineExtraTreesRegressor` (`log(1 + x)` and standardised) get their band magnitudes scaled by default, because they learned
  better so (5 series of synthetic tones, then 2 more series through `AudioInput`); every other primitive gets the magnitudes as
  they are, and `scaling=None` turns it off. The measurement is in `docs/audio/sources.md`. `TextInput` and `VideoInput` do not use it.
- `TextInput` (`core/signals/preprocessing/text_input.py`) and `VideoInput` (`core/signals/vision/video_input.py`): one
  adapter each that gives documents or frames to any primitive of the library in the form it takes (a drift detector
  gets one number, `score_one` scores and learns, `predict_one` and `learn_one` predict and learn with the label,
  `transform_one` learns and transforms). They extend `AudioInput`, so the choice of the method is not copied, and
  use the existing `TextVectorizer` and `FrameFeatures` to make the numbers. The 40 classifiers, regressors, anomaly
  scorers and drift detectors of `core/online` that are built without arguments ran on 24 documents and on 24 frames
  without an error; the 28 that need arguments were not tried.

### Changed
- The workflow `tests` runs only the nine pytest jobs (Python 3.10, 3.11 and 3.12 on Ubuntu, Windows and macOS). A change that
  touches only `.md` files, `docs/**`, `mkdocs.yml` or `LICENSE` does not run it. The other workflows run on request.
- The 172 tests that take three seconds or more are marked `slow` from the list `test/slow_tests.txt` and are left out of the
  workflow; run them locally with `pytest -m slow`. On one core the selection of the workflow went from 1306 s to 610 s. The
  coverage that Codecov sees is lower, because it no longer counts them.
- The speed pages (`docs/speed.md`, `benchmarks/speed_log.md`, `benchmarks/speed_delivery.md`) show only the numbers of this
  library.
- The names of two folders are swapped. The library (the online learning, the deep learning, the signals, the
  robot, the evaluation, the infrastructure, the finance and the audio section) is now `dense_armor.core`; the
  shielding engine (`engine`, `hybrid_engine`, `damping_operator`, `memory`, `noise`, `preset`) is now
  `dense_armor.utility`. Before, they were the other way round. Every old path still resolves to the same
  module object through `dense_armor/_compat.py`: the flat ones (`dense_armor.utility.bandit.ucb`), the paths of
  the last release (`dense_armor.utility.online.bandit.ucb`) and the engine modules under the old name
  (`dense_armor.core.engine`). The only import that changes meaning is the package itself: `from
  dense_armor.utility import X` and `from dense_armor.core import X` now give the names of the other package.

### Fixed
- The chain and the section pages read `test/fixtures/finance/...` from the folder where the program was started, so they
  failed when it was started from `tools/dashboard` (as the icon of the installer does). The paths are now built from the
  place of the code (`tools/dashboard/paths.py`).
- The demo reported `verified: true` after its log file had been deleted, and a missing file counts as a valid chain, so
  nothing had been verified. The log is now checked while the file exists.
- `GET /api/demo/summary` ran the whole demo (about 7 seconds). It now returns the last result, and the demo is run by
  `POST /api/demo/run`, which takes a few hundredths of a second because the log is written in one batch.

## [1.5.0] - 2026-11-07

### Changed
- The library is float32 by default. `dense_armor.set_default_dtype("float64")` and `get_default_dtype()` change it.
  The classes that build numpy arrays (the vision, the anomaly scorers, the CUSUM detector, `Orca`, the arbiter, the continual-learning
  and consolidation classes, the similarity index, the MCP tools) use it. `core/engine.py` stays float64: it runs
  its computations inside a local x64 context, so the rest of the process is not forced to x64. The `dynamics`
  modules and the MCP server no longer turn on `jax_enable_x64` for the whole process on import.
  Left in float64: `Armor` and `hybrid_engine` (below about 1e-38 float32 loses the drift: measured), `BayesianLinearRegression`
  and the Mondrian forests (their float32 audit failed: negative variance, copied `M2`), `Beta` and `Multinomial`
  (prudential, not measured: float32 counters lose exactness above 2^24), `Metro`, `gradcheck`, and the `dynamics` modules (`RigidBodyModel` and the two passivity controllers: the QP goes to OSQP in float64; the drift of `forward_dynamics` in float32 is not measured).

### Added
- `dense_armor.utility.finance`, four building blocks for trading systems, each with tests, a benchmark and a page under `docs/finance/`: `twap_schedule` and `vwap_schedule`; `PreTradeChecker` and `Decision`, the five controls of SEC Rule 15c3-5 (c)(1) (size, price band, duplicate, rate, notional); `decode_message`, for the seven Nasdaq TotalView-ITCH 5.0 order messages (`A`, `F`, `E`, `C`, `X`, `D`, `U`); `RingBuffer`, with the storage allocated once at construction.

### Fixed
- `ResidualTree.consolidate()` raised `TypeError: dot_general requires contracting dimensions to have the same shape` for `d_model >= 5`: it passed the codebooks of the children, which live in latent space, to `VQMemory.fit`, which expects data in input space. It now decodes the codebooks of the children, runs k-means down to `n_codes // branching` centroids, rebuilds the parent `VQMemory` from them and fits it for 5 steps.

## [1.4.0] - 2026-11-02

### Changed
- `core/engine.py` no longer asks JAX for `float64`: it computes in the default dtype of JAX, so
  `float32` as it comes, and `float64` when the user sets `jax_enable_x64`. Before, the
  explicit request was silently cut to `float32` with a `UserWarning` unless x64 was on.
  The arrays returned by `AdaptiveSignalStabilizer` follow the same dtype.
- `AdaptiveConformalRegressor.budget_s` and `memory_class` are now properties that
  read the wrapped model. `budget_s` sums two model calls (predict_one and
  learn_one) plus 5e-4, and is `None` if the model declares no budget;
  `memory_class` is the larger of the model class and O(window), `unknown` if the
  model declares something else. With `budget_s` None, `RealtimePipeline` with
  `require_budget=True` refuses the wrapper.
- `_Pipeline` forwards `t` to every `transform_one` in `_forward`, `learn_one`,
  `transform_one`, `predict_one`, `predict_proba_one`, not just to `learn_one`.
- `SignalBatch` with zero rows has `t = None` instead of raising `IndexError`.
- `AnomalyDetector` declares `score_is_inverted = False` on the base class;
  a subclass whose `score_one` is not "high = anomalous" must set it to `True`
  (see the `ChauvenetScorer` entry under Fixed).
- New check `check_clone_drops_learned_state`: `clone()` must not carry learned
  state. Can make `check_estimator` fail on third-party estimators whose
  constructor is not rebuildable from `get_params` (reported as not applicable,
  which `check_estimator` turns into an `AssertionError` unless the check is in
  `allow_not_applicable` or in `_unit_test_skips`) or whose `clone()` copies
  learned state (`AssertionError`).
- `Protected` with a drift detector reads the flag from the last
  `learn_one`, so the first `fallback` is returned on the sample after
  the one that triggered drift. Documented in the class docstring and
  enforced by `test_predict_after_drift_returns_the_fallback_on_the_next_sample`.
- `Armatura` is now `Armor`, the module `dense_armor/armatura.py` is now
  `dense_armor/armor.py`, and the docs page `shields/armatura.md` is now
  `shields/armor.md`. The old import path `dense_armor.armatura` still works
  (it resolves to `dense_armor.armor`) and `dense_armor.Armatura` still works
  and raises a `DeprecationWarning`. The console script `dense-armor` is
  unchanged. The entries below keep the old name.
- The 33 folders of `dense_armor/utility` sit in six groups: `online`,
  `deep` (`neural`), `signals`, `robot` (`control`), `evaluation` and
  `infrastructure`. No file changed inside the folders. The old paths, for
  example `dense_armor.utility.bandit.ucb`, still work and resolve to the
  same module object as `dense_armor.utility.online.bandit.ucb`. The entries
  below keep the old paths.
- The `misc` folder is dissolved: `SDFT` and `Skyline` are in `utility/signals/stats`, `Metro` in
  `utility/signals/preprocessing`, `diag` in `utility/evaluation/evaluate` and `lodat` in
  `utility/signals/stream`. There are 32 subpackages in six groups. The old paths
  (`dense_armor.utility.misc.sdft`, `dense_armor.utility.infrastructure.misc.metro`, ...) still resolve to
  the same modules, and the old package `misc` gives `SDFT` and `Skyline`. The page `docs/misc/skyline.md`
  is now `docs/stats/skyline.md`; the tests moved with the modules.
- **Incompatible with 1.2.0:** `GatedCrossAttention` returns `g * attention(xq, xc)` again, as the item B of
  the task asked, and not `xq + g * attention(xq, xc)`. A closed step produces the zero vector and sends no
  gradient to the queries, to the context or to a parameter. The form of 1.2.0 is gone. To keep the static part of
  the query, add it outside of the class with `Residual` or `Parallel`; the page
  `docs/neural/composable_attention.md` shows it.

- **Incompatible:** `Signal` takes only a vector of shape `(n_channels,)`; a matrix raises `ValueError`. Several
  samples are a `SignalBatch`, and `Signal.stacked` returns one. The batch form of `Signal` was a dict whose
  values were tuples, which no estimator of the library accepts, and its docstring said that any estimator took it.

### Added
- `SignalBatch` (in `dense_armor.roles`): `n` samples of `d` channels for the code that works on `array`
  (shape `(n, d)`). It is not a dict; `row(i)` gives the sample `i` as a `Signal`; `times` and `t` carry the
  timestamps.
- `budget_s` and `memory_class` of the wrappers: `AnomalyGate`, `Protected`, `SafeEstimator`,
  `UnitCheckedPipeline`, the `|` pipeline and the union and product of transformers declare the cost of their
  parts (`None` if a part declares none), so that `RealtimePipeline` takes them as a step and checks their real
  cost. Before, none declared a budget and `RealtimePipeline` refused every wrapper.
- `dense_armor/utility/deep/neural/rotary_decoder.py`: `RotaryDecoderBlock`, `RotaryDecoder` and the builder
  `rotary_decoder`, a causal language model with RMSNorm, rotary causal attention and a SwiGLU feed-forward
  (pre-norm), with `remat=True` to recompute the activations in `backward`. `transformer_decoder` is unchanged.
- `MultiHeadSelfAttention.init_static_cache` and `step_static`: one token against a key/value cache of fixed
  shape, so the step compiles once. `generate` uses it for a model that has `init_static_cache` and
  `step_compiled` (`RotaryDecoder`); any other model is run again on the whole prefix, as before.
  `benchmarks/neural/kv_cache.py` measures the two paths; page `docs/neural/rotary_decoder.md`.
- `benchmarks/train_small_lm.py`, its two GPU JSONs and the page `docs/neural/lm_training.md`: a 14.2-million-parameter
  decoder trained on tiny Shakespeare on a Tesla T4, with the unigram and bigram baselines.

### Fixed
- `ChauvenetScorer` declares `score_is_inverted = True`; `AnomalyGate` and
  `Protected` raise `ValueError` when the detector declares that attribute,
  because its `score_one` is inverted with respect to the
  "high score = anomalous" contract. Before, both gates classified
  inliers as anomalies. Use `HampelScorer`, `SigmaClipScorer` or
  `TukeyScorer` with the gates, or call `ChauvenetScorer.is_outlier`
  directly.
- `rotary_angles` accepts a traced `offset` (it built the positions with `jnp.arange(offset, ...)`, which needs
  a concrete value, so a compiled step with a traced position failed).
- `UnitCheckedPipeline.learn_one` trains every transformer of the pipeline, as `|` does. It only
  transformed with them, so an adaptive transformer never learned inside it.
- `Transformer`: the result of `a + b` and `a * b` is a `Transformer`, so `(a + b) + c`, `(a * b) * c` and
  `a + (b * c)` work (they raised `AttributeError`).
- `AnomalyGate` takes a `threshold` argument, and raises `ValueError` for a detector that has no `threshold`
  attribute and no argument. The default of `0.0` classified as anomalous every positive score, and with
  `protect=True` the detector stopped learning after the first sample. `StreamingDeviationScorer`, `PatchMemory`
  and `OnlineAutoencoder` need `threshold=`.
- `AdaptiveConformalRegressor.predict_one(return_std=True)` and `(return_estimate=True)` give the root mean
  square of the residuals of the window, and its square as the variance. They gave the half-width of the
  conformal interval (a quantile, not a standard deviation) and its square. `predict_interval` is unchanged.
- `call_json` always writes valid JSON: a float that is not finite is the string `"Infinity"`, `"-Infinity"` or
  `"NaN"` (it wrote the constants `Infinity` and `NaN`, which JSON does not have, for the `inf` score of a
  flagged sample).
- `Protected` feeds a drift detector the value of `feature`, the smallest key when it is `None`, and not the
  first key of the dict. New argument `feature`. A detector with `score_one` and a `threshold` attribute is
  applied (it was never flagged). New argument `train_detector` (default `False`, behaviour unchanged): with
  `True`, `learn_one` trains the detector on the samples that are not flagged; a detector that nobody trains
  flags nothing. A drift detector is fed by `learn_one` only, once per sample: `predict_one` and `score_one`
  updated it too, so a sample that was predicted and then learned counted twice; they now read its current state.
- Documented, no change of behaviour: the batch form of `Signal` holds a tuple per channel and is not accepted
  by `learn_one`; `SafeEstimator.learn_one` costs a `predict_one` and a `learn_one` of the model.
- `GatedCrossAttention` refuses a signal of shape `(Tq, Tq)` for a 2-D query (it was accepted and gave an output
  of the wrong shape); the example of `sinusoidal_positions` no longer depends on the float type.

### Removed
- Nothing yet.

## [1.3.1] - 2026-10-30

### Added
- `dense_armor/bridges/sklearn.py`: `SklearnClassifier`, `SklearnRegressor`,
  `SklearnTransformer` (a Dense-Armor estimator behind the scikit-learn
  interface) and `from_sklearn` (a scikit-learn estimator behind the
  Dense-Armor `Estimator` interface). The three wrappers pass
  `check_estimator` 27/27 with 0 not-applicable and 0 skipped on
  scikit-learn 1.6.1. New extra `sklearn = ["scikit-learn>=1.6"]`.
  Page `docs/bridges/sklearn.md`.
- `dense_armor/bridges/torch.py`: `to_jax` / `to_torch` (DLPack, no copy on
  a contiguous CPU tensor), `TorchModule` (a Dense-Armor model as a
  `torch.nn.Module`, backward through `jax.vjp`), `from_torch` (a
  `torch.nn.Module` as an adapter with `jax.custom_vjp`). New extra
  `torch = ["torch>=2.0"]`. Benchmark `benchmarks/bridge_torch.py` and
  page `docs/bridges/torch.md`.
- `dense_armor/bridges/tensorflow.py`: `to_jax` / `to_tf` (DLPack with the
  capsule), `KerasLayer` (a Dense-Armor model as a
  `tf.keras.layers.Layer` through `jax2tf.convert`), `from_keras` (a
  `tf.keras.Model` as an adapter with `jax.custom_vjp` and
  `tf.GradientTape`). New extra `tensorflow = ["tensorflow-cpu>=2.16"]`.
  Benchmark `benchmarks/bridge_tensorflow.py` and page
  `docs/bridges/tensorflow.md`.
- `docs/bridges/index.md`: overview of the available bridges. New
  "Bridges" section in the nav.

### Changed
- `dense_armor/__init__.py`: `__version__` aligned to the `pyproject.toml`
  version (was 1.2.1, the last release is 1.3.0).
- `CITATION.cff`: `version` aligned to the `pyproject.toml` version.

### Reported
- In all three bridges the cross-framework forward costs more than the
  same model called in the source framework: PyTorch 53x, TensorFlow
  53x, both declared in their pages with the JSON benchmark. The claim of
  a 20% overhead declared in the first commit of each task does not hold.
- `from_keras` returns a 2D output and works on float32 only.
- The TensorFlow tests are skipped in the CI of this branch (the extra is
  not installed); they pass locally and on Kaggle.


## [1.3.0] - 2026-10-29

### Added
- `dense_armor/utility/stats/stable.py`: `stable_var` and `stable_moments`, a
  float32 variance kernel. First-element anchor shift (shifted-data algorithm with K = x[0], Healy 2022
  Algorithm 2; Sterbenz exact on close values), two-term formula, and a
  compensated tree (binary merge of `TwoSum`,
  `log2(n)` vectorised levels) on `d**2`. Exactly `0.0` on identical values up
  to `1e38` in eager and under `jax.jit`, on CPU and on GPU; below the baseline
  two-term formula on 3 of 4 datasets of the task in each combination (the two
  exceptions are D4 on CPU and D3 under `jax.jit` on GPU, declared). Complex
  input returns `E|z - mean(z)|**2`, real. 69 tests, 99% coverage of the new
  file. Two JSON benchmarks (`benchmarks/stable_variance_float32.json` for CPU,
  `benchmarks/stable_variance_float32_gpu.json` for a T4 GPU), a page
  `docs/stats/stable_variance.md` and a README paragraph. The speed is 3.59x
  the baseline formula on CPU (range [3.55, 4.60] over three sessions) and
  2.24x on GPU (range [2.20, 2.27]).
- `dense_armor/utility/stats/`: an optional `dtype=None` keyword on the 11
  estimators of the statistics batch (`RunningMoments`, `EWStats`,
  `RunningMomentsVector`, `RunningCovariance`, `Autocorrelation`,
  `RunningCorrelation`, `RollingCorrelation`, `RollingCovariance`,
  `OnlineGaussianMixture`, `OnlineShrinkageCovariance`, `StudentT`). The
  default keeps float64 bit for bit; `dtype=numpy.float32` makes the state
  float32, and in `RunningMoments` `learn_many` of a float32 block uses
  `stable_moments`. Every class passes `check_estimator` 27/27 with 0
  not-applicable and 0 skip, in both dtypes. `OnlineShrinkageCovariance` now
  inherits `Transformer` (it had no `get_params`, so `check_estimator` could
  not run on it); `RunningCovariance._values` and `RollingCovariance._values`
  no longer raise `KeyError` on a sample without `x`/`y` (pre-existing, it
  broke `check_learn_one_does_not_modify_x` on master).

- `dense_armor/utility/stats/student_t.py`, `mixture.py`, `dense_armor/utility/timeseries/garch.py`:
  `StudentT`, `OnlineGaussianMixture`, `GARCH11`. Three online models: a robust
  location-scale estimator (Student-t with `nu` fixed, Liu & Rubin 1995's EM
  update of `nu` not implemented and declared), a Gaussian mixture with diagonal
  covariance (Cappé & Moulines 2009 online EM, Gaussian form derived and
  declared), and GARCH(1,1) conditional variance (Csáji 2018 Eq. 1b, parameters
  fixed and declared). Three test files (25, 24, 19 tests, 68 total, all
  green). `benchmarks/statistics_online.py` with four synthetic domains
  (robot, drone, finance, bimodal), a `--robust` flag, and a `--crunchdao`
  block that measures whether the three features add signal on CrunchDAO
  DataCrunch 2 data. Three docs pages: `docs/stats/student_t.md`,
  `docs/stats/mixture.md`, `docs/timeseries/garch.md`. The learning-rate
  schedules were measured (constant, sqrt, pow, adaptive) and none dominates:
  the adaptive schedule gives the wrong sign on the robot domain. No adaptive
  schedule is added to the library; the class docstring declares that a fixed
  learning rate converges to a neighbourhood of the MLE.
- `dense_armor/utility/neural/moe.py`, `test_time.py`: `SparseMoE` and
  `TestTimeCompute`. `SparseMoE` dispatches each token to its top-k experts
  with a capacity factor (Fedus et al. 2021, Section 2.2, page 6), so only
  the selected experts run, not all `n_experts`. `forward_pure` is pure and
  compiles under `jax.jit`, `backward` is `jax.vjp` of `forward_pure`, the
  load-balancing loss is Fedus Eq. (4), page 7, and the routing is a
  segment cumsum on the top-k indices, `O(n_tokens * top_k *
  log(n_tokens * top_k))`, independent of `n_experts`. Optional noisy
  gating (Shazeer et al. 2017, Eq. (4), page 4, fixed noise std). Two
  metrics after `forward`: `expert_usage()` and `dropped_fraction()`.
  `TestTimeCompute` generates `n_candidates` noisy copies and picks the
  one a verifier scores highest. Four verifiers: `"confidence"`
  (default), `"margin"`, `"entropy"`, `"ensemble"`. Two test files (16 +
  6 tests, 22 total, all green). `benchmarks/neural/moe_testtime.py` with
  two synthetic 2D experiments (MoE vs dense, test-time vs single pass),
  5 seeds, mean ± std, a `--smoke` flag; the JSON is
  `benchmarks/neural/moe_testtime.json`. One docs page,
  `docs/neural/moe_testtime.md`, with the dispatch mask image
  `docs/assets/moe_testtime/dispatch_mask.png` generated by the library.
  On the micro domain the MoE beats the dense on task 1 (0.722 ± 0.077 vs
  0.682 ± 0.022) and forgets less (0.095 ± 0.033 vs 0.149 ± 0.009), but
  loses on task 2 (0.806 ± 0.049 vs 0.830 ± 0.047); the test-time delta
  is at most +0.003 within the noise of 0.004-0.010, and the page declares
  that the wrapper does not improve accuracy on this domain.

### Fixed

- `RunningMoments.learn_one` returned `nan` on the first sample at `1e307`
  (`inf * n_old` with `n_old = 0`). The first sample now sets the state
  directly.
- Attach on Windows without the resource tracker: the `spawn` Pool no longer
  hangs. Tests for the pickled handle, docstrings, docs page.
- Review fixes 1-4 on `attention`, `continual` and `consolidation` (see the
  individual commits `2299a13` … `54fb537`).

### Added

- `dense_armor/utility/data/shared.py`: shared-memory containers
  (`SharedArray`, `SharedList`, `SharedDict`, `to_shared`) for process-to-process
  data sharing. They do not inherit `Root` and are not estimators; the module is
  independent and its tests do not call `check_estimator`. Page
  `docs/api/data_shared.md`, benchmark `benchmarks/data_shared_memory.py` and
  `benchmarks/data_shared_memory.json`.
- `dense_armor/utility/neural/consolidation/`: the consolidation code split
  into a package (`vq.py`, `tree.py`, `memory.py`, `archival.py`). No
  behaviour change; the tests of `test/neural/test_consolidation.py` still
  pass.

### Changed

- `.github/workflows`: the coverage flags are carried forward on docs-only
  commits (`codecov.yml` updated).
- `ARCHITECTURE.md`: lists the 12 missing modules (neural, stats, timeseries),
  drops `anwav` and `collatz`.


## [1.2.1] - 2026-10-27

### Fixed
- `OnlinePlattScaling`: handle NaN in base predictions (fallback to 0.5).
- Log-domain clipping (1e-15) in LogLoss, Beta, OnlineGaussianNB, Gaussian Mixture, GaussianNLL, Hoeffding Tree, and AMRules.
- Division and square root protections against zero variance in Gaussian PDFs, Correlation, Hoeffding leaves, SGT, OnlineGaussianNB, and Mixture.
- Exponential overflow protection (clamp to [-700, 700]) in Factorisation Machines, SGT, Mixture PDF, and Student-T PDF.
- Pre-existing mypy errors in hoeffding.py, mixture.py, and amrules.py.

## [Unreleased]
1.2.0] - 2026-10-26

### Changed
- README Speed section: one sentence on EFDT (26-58x faster than the adaptive random forest, at parity on sudden and gradual drift, 30.0 pp behind after the change on incremental drift, no native drift detection). The numbers come from `benchmarks/neural/efdt_vs_arf.py`.
- `GatedCrossAttention.forward` returns `xq + g * attention(xq, xc)` and not `g * attention(xq, xc)`: a closed step returns the query unchanged. Deviation from the task specification, declared in the class docstring: a zero vector at a closed step breaks the gradient path of the query. The hand case and the closed-step test are updated.

### Added
- `dense_armor/utility/neural/continual.py`, `generative_replay.py`, `consolidation/` (vq, tree, memory, archival): `EWC`, `GEM`, `AGEM`, `DeepGenerativeReplay`, `VQMemory`, `PQMemory`, `ResidualTree`, `LifecycleMemory`, `DualMemory`, `TheoreticalBound`, `ArchivalMemory`. `docs/neural/continual_learning.md` with the eleven pieces. `benchmarks/neural/continual_learning.py`, `memory_fidelity.py`, `public_benchmark.py`, `efdt_vs_arf.py` (EFDT vs HAT vs ARF-10 on three drift types, with the declared -30.0 pp limit of EFDT on incremental drift). Three test files (continual 19, generative replay 5, consolidation 57).
- `dense_armor/utility/neural/cross_attention.py`, `gated_attention.py`, `latent_attention.py`, `bottleneck.py`: `MultiHeadCrossAttention`, `GatedCrossAttention`, `LatentCrossAttention`, `BottleneckFusion`. `docs/neural/composable_attention.md` with the four bricks. `benchmarks/neural/composable_attention_figure.py`. Four test files (A: 12, B: 16, C: 11, D: 14).

### Fixed
- `HoeffdingTreeClassifier` with `leaf="nb"` (the default): a leaf in which any class has fewer than two samples now answers like `leaf="majority"`, in `predict_proba_one` and in `predict_one`. Before, that class skipped every likelihood term and kept only its prior, while the other classes added one log-density per feature, so on an imbalanced stream the rare class got a probability close to 1 for ordinary samples. On a leaf with 300 samples of class 0 and one of class 1 (synthetic), the median probability of class 1 on 200 new class-0 samples was 0.953 and is now 0.0033. On the credit card fraud stream (284,807 rows, 492 frauds) the default configuration measured AUROC 0.513 and AUPRC 0.0032 before the change (`dense-business`, `benchmarks/kaggle/n65_diag`). On the sudden-drift stream of `test/ensemble/test_comparison.py` (2,000 samples, 10 seeds) the single tree goes from a median accuracy of 0.9392 to 0.9965 and `OnlineBoostingClassifier` from 0.9758 to 0.9970, so the gain of boosting over the single tree (0.0365, Wilcoxon p = 0.0039) becomes 0.0005 (p = 0.125); the test `test_boosting_beats_single_tree` is replaced by `test_boosting_not_worse_than_single_tree`. Tests: `test_nb_leaf_one_minority_sample_does_not_saturate` and `test_nb_leaf_still_used_when_every_class_has_two_samples` in `test/tree/test_hoeffding.py`.

## [1.1.35] - 2026-10-24

### Changed
- The DOI of version 1.1.34 in the README badge, the README links and `CITATION.cff`.

### Added
- `docs/neural/attention.md`, the page of the attention; `docs/api/neural.md`, the signature of every public class and function of `neural` and `optim`.
- The 16 sections of the pages `docs/neural/index.md`, `docs/neural/vision_layers.md`, `docs/neural/deep.md`, `docs/neural/sequence_layers.md` and `docs/neural/continuous.md` for the neural classes that had no code example: `Sigmoid`, `Tanh`, `Softmax`, `PRelu`, `SoftmaxCrossEntropy`, `BCEWithLogits`, `Huber`, `MaxPool1D`, `AvgPool1D`, `AvgPool2D`, `Upsample1D`, `Upsample2D`, `LastStep`, `MeanOverTime`, `AddPositions`, `OnlineNetClassifier`.
- `benchmarks/neural/check_examples.py`, the script that checks that each class is in a code block of its page and that every block of the five pages prints the output written under it; `test/neural/test_check_examples.py` (4 tests).

### Fixed
- `FFMClassifier` allocates the factor array only for the features that have a field (before, a feature of `x` that was not in `fields` took a row of `n_features * k` floats that was never read).
- Sketches: `CountMinSketch`, `HyperLogLog`, `BloomFilter` and `SpaceSaving` have `n_missing_` and the read-only property `n_missing`; `learn_one` on a sample without the watched key counts it instead of raising `KeyError`, and `transform_one` counts it and returns `{}`; `merge` adds the counters.
- `check_estimator`: `check_mutate_idempotent` applies the same `mutate` twice and asserts that the state is the same as after the first (before it called `mutate` once and asserted nothing); `check_learn_one_does_not_modify_x` takes the snapshot with `copy.deepcopy`; `check_memory_growth_bounded` also checks `O(window)` (it fills the window plus 8 samples, then asserts that 32 more do not grow the memory; estimators without an integer `window` are skipped); the message of `check_p99_within_budget` for a `budget_s` under 1e-3 s says "is below" the lower bound instead of "exceeds" it.
- Stats: `RunningMoments.learn_many` counts every NaN or infinite value in `n_missing_` (before, 1 only for a block with no finite value) and `RunningMomentsVector.learn_many` counts a channel absent from the dict; `RunningMoments`, `RunningMomentsVector`, the quantile estimators, the robust estimators and `Autocorrelation` count a missing key or a value that is not a number in `n_missing_` instead of raising `KeyError`; `TDigest.quantile(0)` and `quantile(1)` return `min_` and `max_`, as `DDSketch` does; `OnlineShrinkageCovariance.shrinkage` uses the smallest number of complete pairs as `n`, so a sample with a NaN no longer inflates it; the sketch hash accepts a `seed` of `2**32` or more (taken modulo `2**32`).
- Streams: `iter_csv` raises `ValueError` for `start` or `stop` without `time_col` (before they were ignored); `merge_by_time` refuses a NaN or infinite timestamp as it does `None`; the docstring of `iter_rosbag` states that an unknown topic gives an empty iteration.
- `n_missing_` with the read-only property `n_missing` (as the rest of the library) in the active-learning strategies, in all the bandit policies, in the factorisation machines, the linear models (`glm`, `bayes`, `payload`), the recommenders, `skyline` and the neural adapter, which had a plain `n_missing` attribute; reads of `n_missing` are unchanged. `VariableUncertainty` loses a condition that was always true; the docstring of `VariableRandomizedUncertainty` states that a large `delta` can give a negative threshold.
- Anomaly: `StreamingDeviationScorer`, `HampelScorer`, `TukeyScorer`, `ChauvenetScorer`, `SigmaClipScorer` and `HampelFilter` no longer raise `KeyError` on a sample without the watched key (or empty, or with a value that is not a finite number): `learn_one` counts it in `n_missing_`, `score_one` returns the pre-warm-up value, `HampelFilter.transform_one` returns the input unchanged; `StreamingDeviationDetector.update` ignores a non-finite value instead of putting it in the window; the docstrings of `OIForest` (score before `ready`), `OneClassSGD` (meaning of `t`), `MahalanobisDetector` (identity fallback below `d + 1` samples) and `curvature` (not a differential curvature, name kept) state what the code does.
- Bandits: `BayesUCB` keeps the variance of the Gaussian rewards with the Welford update (`m2_`, replacing `sumsq_`), stable for large rewards; `offline_replay` refuses `n_bootstrap < 1`; the docstrings of `BayesUCB` and `ThompsonSampling` state that a reward `>= 0.5` is a success and that the Gaussian posterior of `ThompsonSampling` has a flat prior (not `N(0, 1)`).
- `CategoricalNB`: `predict_proba_one` counts the categories of each feature once instead of once per class (O(c d) instead of O(c² d) category scans); the docstrings of `CategoricalNB` and `ComplementNB` state how a missing key, a `None` or a non-finite value is handled.
- `OnlineKMeans`: during the warm-up a point of another dimension is counted in `n_missing_` (before it raised `n_seen_`); the docstring states that the first centre is the point nearest to the mean (PatchCore starts at random) and that `centers_` has fewer than `k` rows with fewer than `k` distinct warm-up points.
- `FuncTransformer`: with `whole=True` a function that does not return a dict raises `TypeError`, and a `Signal` keeps its units and missing mask when the returned names are the input names; the channel lookup is one dict instead of `names.index` per key; the module docstring said "empty keys" where the code (and the argument docstring) means `None`.
- Control: `kinematic_tracking_controller` and `quintic_trajectory` raise `ValueError` instead of `assert` (the check stays under `python -O`), `quintic_trajectory` also refuses arrays that are not 1-D, a `T` that is not finite and `n_samples < 1`; the CBF functions refuse `n_substeps < 1`; the docstrings state that `rate_limited_follower` does not limit the first sample, that `dt` is the sampling period, that the CBF filter returns `u_des` on the obstacle and that `cbf_filtered_trajectory` does not bound the speed.
- Metrics: `merge` of `Precision`, `Recall`, `FBeta` and `F1` refuses another `positive` or `average`; `MeanAbsoluteError`, `MeanSquaredError` and `RootMeanSquaredError` refuse a change between a scalar and a vector (before, the scalar was added to joint 0); `Rolling` refuses a metric that cannot undo an update (`R2Score`, `RollingAUC`) when it is built, and a missing `t` in the `window_s` mode; the docstrings state that `Rolling.merge` is for reporting, that `PointAdjustedF1.merge` does not exist, that `RollingAUC` ignores `w`, that `nab_score` is not clamped at 0 and the labels of `LogLoss` and `BrierScore`.
- `LogisticRegression` binarises the label as its docstring says (a nonzero label is 1); `LogisticRegression`, `LinearRegression` and `BayesianLinearRegression` skip an infinite value and count it in `n_missing`; `RecursiveLeastSquares` counts a missing key as 0.0 like the other linear models.
- `BatchDecisionTreeClassifier.fit` refuses a row without a feature, a value that is not a number and a NaN or an infinite value; the unused `_classes_` is removed; the docstring states the schema, the answer on a missing feature and the budget.
- `LEGO.percentile_targets` refuses a set without a pair of the same class or without a pair of different classes; `OnlinePlattScaling` asks the base classifier once per call and not twice.
- `OnlineModelSelection.current_best` raises before the models have run, as its docstring says; `progressive_val_score` and `progressive_val_proba_score` refuse a negative delay and a delay in seconds on samples without a timestamp (before, nothing was scored until the end); `evaluate_events` calls `score_one` and `learn_one` of an anomaly detector without `t` when they do not take it; the docstrings state what the metric receives from `progressive_val_proba_score` and what the trace holds.
- Drift: `ADWIN` takes its range `R` from the window that is left after an eviction (before, the minimum and the maximum of the evicted samples stayed and made the threshold grow); `dense_armor.utility.drift` exports `ADWIN`, `CUSUMDriftDetector`, `KSWIN` and `PageHinkley`; `BayesianOnlineChangepoint` sets `n_missing_` in `_reset` like the others; `cusum_detector` with `reference="fixed"` refuses a first window that holds a NaN or an infinite value (before, the whole stream was skipped without a message).
- `Metro.enc` keeps the sign of a negative value; `diag` scales each type of sample by its range and refuses one path with one array; `anwav` labels the loudness as an estimate (RMS plus 3 dB); the docstrings of `ABCollatz.calculate_jax_rad` and of the recommenders (`Baseline`, `BiasedMF`) state the limit of 255 and the order of the update of `mu`.
- `LSTM.backward` with `peepholes=True` added the term `p_o * do_pre` to the cell gradient twice, so the gradient carried to the previous step was wrong as soon as `p_o` was not zero (it starts at zero, so the gradient checks at initialisation could not see it); the second addition and a repeated assignment are removed, and two gradient checks run with non-zero peepholes.
- `FFMClassifier.learn_one` learns the label `0`: with `y = 0` the gradient of eq. (5)-(6) was zero and the model never learned the negative class. The labels `1` and `0`, or `+1` and `-1`, now give the update of the logit loss; the update for `+1` and `-1` is unchanged.
- `docs/neural/deep.md` held its sections 7 to 11, the figure and the details twice; the repeated half is gone.
- `ARCHITECTURE.md` lists `bocpd.py`, `vectorizer.py`, `shrinkage.py`, `similarity.py`, `batch.py` and `fimt.py`.
- Controllers: `passivity_cbf_controller.solve_control_qp` drops the joint-limit box when the CBF and the box are jointly infeasible, as the 6-DoF controller does; both controllers raise `RuntimeError` when even the CBF-only QP is not solved (before, OSQP's infeasibility certificate, of order 1e9, came back as the acceleration) and raise `ValueError` for a robot with fewer joints than the task dimensions (3 or 6), where the manipulability gradient is NaN.
- `DynamicAICodegen.run_pipeline_with_chunking` carries the PRNG key across chunks and across calls: a dropout no longer repeats the same mask, and the result no longer depends on `chunk_size`. `compile_pipeline` names the instruction whose parameter is not a number.
- `BitwisePermutationEngine.apply_bitwise_swap` accepts a JAX array.
- LLM bridge: `learn_json` refuses a missing or `null` label (before, `None` was learned as a class); the output of `embed` must be a 1-D sequence of numbers (before, a string of digits became one feature per character); `predict_proba_json` writes the class keys as `predict_json` writes the class (`"true"`, not `"True"`); `describe` names a callable instance by its class.
- Roles: `RealtimePipeline.learn_one` passes `sig.t` to every step that takes `t`; `call_json` dispatches `update(x, t)` to a drift detector and returns `drift_detected`; `SafeEstimator` and `UnitCheckedPipeline` pass `y` and `t` only to a `learn_one` that declares them (before, an unsupervised `learn_one(x, t)` received `y` as `t`); `SafeEstimator` monitors a classifier with the 0/1 error.
- Docstrings and notes: `filter_batch_scenarios` recalibrates the filter on `self`; the engine preserves the magnitude per element (it said "L2 norm"); `Signal.stacked` gives `t=None` for different timestamps; the MCP server drops a session idle for one hour at the next `dense_armor_stream_start`, and its standalone `requirements.txt` lists the core dependencies.

## [1.1.34] - 2026-10-23
### Added
- `dense_armor/utility/optim/`: `RMSProp`; `SGD` with momentum and Nesterov.
- `dense_armor/utility/neural/`: 10 new modules (clip, activations_more, schedules_more, convtranspose, bidirectional, norm_more, conv_more, positions, losses_more, online_vector); `PRelu.params()`; `clip_norm` in the four estimators.
- `docs/neural/deep.md` and `docs/assets/neural/deep.png`.


### Added
- `dense_armor/utility/neural/`: `Autoencoder`, `VectorQuantizer`, `MatryoshkaLoss`, `Seq2Seq`, `OnlineAutoencoder`; `docs/neural/encoder_decoder.md` with the guide and the figure.
### Added
- `dense_armor/utility/neural/`: `OnlineNetRegressor`, `OnlineNetClassifier`, `HedgeNetClassifier`, `BackgroundLearner`; `docs/neural/continuous.md` with the guide and the figure of continuous training.
### Added
- `dense_armor/utility/neural/`: `ConstantLR`, `StepLR`, `CosineWarmRestarts`, `minibatches`, `Trainer`; `docs/neural/training.md` with the guide and the figure of the training loop.
### Changed
- The marker `heavy` (tests above 14 seconds: `test/dynamics`, the payload and the experiment suites, `ensemble/test_comparison.py` and a few more) moves 1060 seconds of the suite out of the matrix of the workflow `tests`, which now runs `-m "not training and not heavy"`; the workflow `tests-slow` runs `training`, `heavy-dynamics` and `heavy-rest` as three parallel jobs (Python 3.12, ubuntu) with the coverage uploaded to Codecov.
- The five tests of `test/neural` that train a network for minutes carry the marker `training`; the workflow `tests` runs `-m "not training"` and the new workflow `tests-slow` runs them on one job (Python 3.12) when `dense_armor/utility/neural/`, `test/neural/` or `pytest.ini` change. A test for `MaxPool2D.backward` before `forward`, the format of three files, and a positive wording in `docs/neural/vision_layers.md`.

### Added
- `dense_armor/utility/neural/`: `Conv1D`, `Conv2D`, `MaxPool1D`, `MaxPool2D`, `AvgPool1D`, `AvgPool2D`, `Flatten`, `RNN`, `GRU`, `LSTM`, `MultiHeadSelfAttention`, `TransformerBlock`, `LastStep`, `MeanOverTime`, `AddPositions`, `cnn`, `recurrent_net`, `transformer_encoder`; `docs/neural/vision_layers.md` and `docs/neural/sequence_layers.md`.
- `dense_armor/ml/`: the machine learning classes, which read the whole table in `fit`: statistics, preprocessing, linear models, naive Bayes, nearest neighbours, multiclass, multi-output, distributions, trees (CART and Mondrian forests), anomaly detection (Hampel, Tukey, Chauvenet, sigma clipping, Local Outlier Factor), change detection (CUSUM with a fixed reference) and vision (Horn-Schunck flow, motion mask, phase correlation, random projection, PCA); each module lists in `STOPPED` the classes whose reference defines no algorithm for the whole dataset; `docs/ml/` and `docs/compare/` hold the pages and the comparison with the online classes.
### Added
- `dense_armor/utility/neural/`: `Sequential`, `Parallel`, `mlp`, `resnet_mlp`; `docs/neural/chains.md` with the guide and the figure of the MLP on a synthetic two-moons set.
### Added
- `dense_armor/utility/neural/`: `Dense`, `Embedding`, `Dropout`, `Residual`, `BatchNorm`, `LayerNorm`, `AsTransformer`; `docs/neural/layers.md` with the guide and the figure of the two normalisations.
### Added
- `dense_armor/utility/neural/`: `Param`, `he_normal`, `xavier_normal`, `glorot_uniform`, five activations and four losses (mse, softmax cross-entropy, bce-with-logits, Huber) with `forward`/`backward`, and `gradcheck` against central finite differences; `docs/neural/index.md` with the guide and the figure of the five activations.
### Changed
- The DOI of version 1.1.33 in the README badge, the README links and `CITATION.cff`.

## [1.1.33] - 2026-10-21

### Added
- `notebooks/drone_actuator_fault.ipynb`: the actuator-fault pipeline on synthetic flights, with the CUSUM threshold `h` against false alarms per hour.
- `Documentation`, `Changelog` and `Issues` in `[project.urls]`.
- `docs/drones/index.md` lists every page of the drone section.
- `edge/power.py`, `edge/energy_bench.py`, `test/drones/test_power.py`, `test/drones/test_energy_bench.py`, `docs/drones/jetson_energy.md`: the energy of a run (trapezoid integral of a power trace, net of the idle), a paired batch-vs-online energy benchmark, the four calibration lines of Table I of 2306.13107, and one command for a Jetson board. Every energy value is synthetic or null: no reduction on a board is claimed.
- `edge/board_selftest.sh`: the energy section after the existing steps, on `POWER_NODE`.
### Changed
- The licence is the GNU Affero General Public License v3.0 or later (`LICENSE`); the Business Source License 1.1 text of the releases up to 1.1.32 is in `LICENSE-BUSL-1.1-until-1.1.32.md`. `pyproject.toml`, `CITATION.cff`, the README and `docs/license.md` follow.
- The README says that the library also learns from a whole dataset (`fit` and the batch versions) in the three places that said the opposite.

### Changed
- The DOI of version 1.1.32 in the README badge, the README links and `CITATION.cff`.

## [1.1.32] - 2026-10-21

### Added
- `TextVectorizer` in `dense_armor/utility/preprocessing/vectorizer.py`, `stop_words` in `Tokenizer`, `norm` in `TFIDF`, `cosine_similarity`, `cosine_similarity_many`, `SparseCosineIndex` (`query_many`, `query_many_sparse`, `top_k_threshold`) in `dense_armor/utility/stats/similarity.py`, `TextVectorizer.transform_matrix`, `benchmarks/text_retrieval.py`, `benchmarks/primitives.py` and `docs/preprocessing/text_vectorizer.md`.
- `dense_armor/utility/drift/bocpd.py` (`BayesianOnlineChangepoint`, Adams and MacKay 2007) and `dense_armor/utility/stats/shrinkage.py` (`OnlineShrinkageCovariance`, Bodnar, Gupta and Parolya 2014). Seven finance pages and six finance modules: `regime.py` (BOCPD on daily returns, 6 alarms in 2.48 years of calm data on GSPC, synthetic delay 5 samples), `drift_vs_noise.py` (raw against pre-averaged, p=0.0082 A12=0.675 calm, p=0.0001 A12=1.000 delay), `portfolio.py` (minimum-variance and inverse-volatility from the shrunk covariance), `order_book.py` (order flow imbalance, equation 4 of Cont et al. 2010), `stress.py` (synthetic flash crash: surveillance 0/5 alarms, execution 2.0 refusals median), `connectors.py` (`BoundedStream` with three policies, `finance-stream` extra). `benchmarks/finance/` for the five of them and `docs/finance/{regime,drift_vs_noise,portfolio,order_book,stress,streaming,timing}.md`.
- `pipelines/drones/`, a drone section of nine pipelines over the library primitives, on synthetic streams: `actuator_fault` (`CUSUMDriftDetector`, `OnlineRobustMahalanobis`; one episode 0.02 s after the fault, 0 false alarms), `sensor_consistency` (`CUSUMDriftDetector`; 3 episodes at 0.28 / 1.47 / 2.67 s), `propeller_health` (`SDFT`; band [40, 55] Hz 0.74 to 57.10, peak 46.88 Hz), `battery_endurance` (`AdaptiveConformalRegressor`, `LinearRegression`, `AdaGrad`, `TDigest`; q50 22.5 min, coverage 0.923), `safety_filter` (`cbf_safety_filter_live`; 2.0 m/s to 0.285 m/s), `link_health` (two `CUSUMDriftDetector`; 0.19 s after the dropout), `visual_landing` (`GlobalMotion`, `DenseFlow`, `MotionMask`, `FrameAnomaly`; mean shift error 0.731 px, FrameAnomaly 0.3100 to 0.6433, MotionMask 0 to 185 px), `static_vs_online` (a CART tree against a Hoeffding tree; 1.0000 against 0.9920, paired Wilcoxon p = 0.0020, A12 1.000).
- `pipelines/drones/alfa.py`: reader for the ALFA dataset (Keipour, Mousaei & Scherer 2020, DOI 10.1184/R1/12707963), the same fields as `read_mavlink_log` plus the fault ground truth, and `alfa_fault_time`.
- `benchmarks/drones/alfa_actuator.py` and `kaggle/drones_alfa_actuator.py` with a `SMOKE` switch: the actuator-fault pipeline on the processed ALFA flights, delay and false alarms per hour over the flights.
- `dense_armor/utility/tree/batch.py`: `BatchDecisionTreeClassifier`, a CART tree fitted on the whole training set (Breiman et al. 1984, chapter 2; Gini index as in Bourel et al. 2026, page 8). `OnlineBaggingClassifier.fit_batch`, the batch protocol of Oza and Russell 2001.
- `edge/`: `edge_bench.py`, `mqtt_bridge.py`, `install.sh`, `Dockerfile.arm64`, `Dockerfile.jetson`, the systemd unit and `board_selftest.sh`. `docs/drones/`: nine pipeline pages, `start_here.md`, `ardupilot.md` (ArduPilot against PX4, field by field, in the `pymavlink` dialects) and `jetson_power.md` (the `nvpmodel` modes of each Orin module, the commands, and a measurements table to be filled with the numbers of a board).
- `.github/workflows/arm64-smoke.yml`: the arm64 image builds under QEMU emulation and `edge.edge_bench --smoke` starts on it, on `edge/**` and `pipelines/drones/**` changes.

### Changed
- `pyproject.toml` has a `[tool.ruff]` section (target py310, `.claude` excluded, rules E4, E7, E9, F, I, UP) and `ruff check --fix` is applied to 44 files of `dense_armor`: import order, annotations, unused `noqa`. The 30 errors left in `dense_armor` have no safe fix.
- `docs/fit.md` lists the algorithms that work on the whole dataset: `BatchDecisionTreeClassifier`, `OnlineBaggingClassifier.fit_batch` and `TextVectorizer.fit`.

## [1.1.31] - 2026-10-17

### Added
- `CUSUMDriftDetector(reset_on_alarm=False)` and the counter `n_alarms_`: with `reset_on_alarm=True` the statistic and the reference restart after an alarm, so a sustained step gives one alarm and not one per sample. The default is the old behaviour. `benchmarks/casper_hours_episodes.py` and `kaggle/casper_hours_episodes.py` count alarm episodes per hour next to the count of flagged samples.
- `learn_many` on `RunningMoments` and `RunningMomentsVector`: a vectorised batch update, equal to the row loop within `1e-9` (tested).
- `dense_armor/utility/_validation.py`: `as_dict`, `finite_row` and `is_number`, the validation that several estimators copy; the existing classes are not changed.
- `FIMTDDRegressor.n_missing`, the name the rest of the library uses; `n_missing_` stays.
- `benchmarks/flow_levels.py` and its JSON: the error of the optical flow against the true shift for one to four pyramid levels and for the intensity scales, with and without `normalize_intensity`; `docs/vision/flow.md` states what each number of levels recovers.
- `benchmarks/seed_audit.py` and its JSON: the estimators whose output depends on chance, and the default of their `seed`; `benchmarks/hotspots.py`: the profile of five models.

### Fixed
- Type errors of `mypy` in `vision/window.py`, `drift/cusum.py` and `vision/dense.py`, with no change of behaviour.

### Changed
- The DOI of version 1.1.30 in the README badge, the README links and `CITATION.cff`.

## [1.1.30] - 2026-10-17

### Master program 1 closed
The first master program, the online-learning toolbox in 21 sections, is complete. What each section delivers:

- **Base and checks, statistics, preprocessing, metrics and evaluation, pipelines and datasets:** the `roles` base classes and `check_estimator`, the robot layer (timestamped `Signal`, real-time contract, uncertainty, safe estimators, units), moments, robust statistics, quantile sketches, scalers and feature selection, prequential evaluation, streams and the synthetic datasets.
- **Models:** linear models and optimisers, naive Bayes, neighbours, multiclass and multi-output wrappers, probability distributions, Hoeffding trees, FIMT-DD regression model trees, adaptive random forests, Mondrian forests, extra trees and AMRules, with the numeric go/no-go for the aggregated Mondrian forest and the online extra trees.
- **Ensembles, bandits and active learning, factorization machines:** online bagging and boosting, exponentially weighted average, voting and stacking; seven bandit policies with offline replay and four labelling strategies; factorization machines (plain, field-aware, field-weighted, higher-order), three recommenders, the sliding DFT and the skyline.
- **Anomalies, drift, clustering, time series, uncertainty:** robust detectors, Half-Space Trees, LODA, one-class SGD, online isolation forest, the local outlier factor; CUSUM, Page-Hinkley, ADWIN2 and KSWIN; online k-means and DenStream; online AR, Kalman filters and the cycle residual; adaptive conformal intervals.
- **Protection, control and the LLM bridge:** arbiter, healing and the real-time guard; kinematic controller, trajectories and passivity; embeddings as features and estimators as JSON tools.
- **Native vision:** frame streams and video files, features, motion, tracking, registration, dense flow, patch memory, with eight guide pages and twenty figures.
- **Real robot (CASPER, UR3e):** the cycle-residual CUSUM detects the labelled anomaly 15.00 s after it begins. The same configuration on the 15 normal hours before it fires a median of 5,696 false alarms per hour, with 2 hours at zero: the page states both. The dataset has no lidar, so the page carries a lidar benchmark that is a simulation and says so. The IMU file has no timestamp column and cannot be tied to the arm by cross-correlation (strongest correlation 0.087), so no IMU delay is reported.


### Changed
- The video documentation is split into eight guide pages under `docs/vision/`, each with a figure from `docs/assets/vision/make_figures.py`.

### Added
- Bandit policies: `EpsilonGreedy`, `UCB1`, `UCB1Tuned` (Auer, Cesa-Bianchi
  and Fischer 2002), `ThompsonSampling` and `BayesUCB` (Chapelle and Li
  2011, Kaufmann, Cappe and Garivier 2012), `Exp3` (Auer et al. 2002) and
  `LinUCBDisjoint` (Li et al. 2010). Offline evaluation on a logged stream:
  `offline_replay` (Li et al. 2011). Robot test: `RobotKpSelector`, UCB over
  the gain `kp` of `kinematic_tracking_controller` on `SyntheticArm`. Guide:
  `docs/bandit/index.md`. Benchmark: `benchmarks/bandit_quality.py`, 20 seeds,
  one sudden change, regret against the oracle and against the best fixed arm.
- Active learning: `RandomStrategy`, `FixedUncertainty`, `VariableUncertainty`,
  `VariableRandomizedUncertainty` (Zliobaite et al. 2011). Guide:
  `docs/active/index.md`.
- `fit(X, y=None, t=None, epochs=1, shuffle=False, seed=None)` on the base class
  `Root`: one offline entry point for every online estimator. It feeds the whole
  dataset to `learn_one`, row by row, for `epochs` passes, optionally shuffled.
  Nothing existing changes. Guide: `docs/fit.md`.
- Ensembles of online learners: `OnlineBaggingClassifier` and
  `OnlineBoostingClassifier` (Oza and Russell 2001),
  `ExponentiallyWeightedAverageRegressor` (Freund and Schapire 1997, Hedge
  rule and its equation 9 bound), and the `VotingClassifier` and
  `StackingClassifier` wrappers. Guide: `docs/ensemble/index.md`. Benchmark:
  `benchmarks/ensemble_quality.py`, 20 seeds, one sudden change, paired
  Wilcoxon and A12 effect size.
- New package `dense_armor/utility/facto/` with `FMRegressor` and `FMClassifier` (Rendle 2010),
  `FFMClassifier` and `FFMRegressor` (Juan et al. 2016), `FwFMClassifier` and `FwFMRegressor`
  (Pan et al. 2018), `HOFMClassifier` and `HOFMRegressor` (Blondel et al. 2016).
- New package `dense_armor/utility/reco/` with `Baseline`, `BiasedMF` and `FunkMF` (Paterek 2007,
  Koren et al. 2009), `learn_one(user, item, rating)`, `predict_one(user, item)` and `rank`.
- New module `dense_armor/utility/misc/sdft.py` with `SDFT` (the N-point DFT of the last N
  samples in O(N) per sample, Jacobsen and Lyons 2003).
- New module `dense_armor/utility/misc/skyline.py` with `Skyline` (non-dominated points of a
  sliding window, Börzsönyi et al. 2001, Tao and Papadias 2006).
- New script `benchmarks/facto_quality.py` and `docs/assets/facto/figures.json` with every
  number quoted on the two guide pages.
- New pages `docs/facto/index.md` and `docs/reco/index.md`.
- `FIMTDDRegressor` (`utility/tree/fimt.py`): FIMT-DD regression model tree (Ikonomovska 2012) with an extended binary search tree per leaf, perceptron leaves and an alternate tree grown after a Page-Hinkley alarm.
- `DenseFlow(normalize_intensity=False)`: an optional normalisation, off by default, that makes the flow independent of the intensity scale; `FrameWindow` resets when the frame shape changes and counts the changes in `shape_changes_`; `FrameAnomaly` is tested on 1-, 3- and 4-channel frames.
- `benchmarks/video_quality.py --size WxH --smoke` and `kaggle/video_quality_kaggle.py` for the long measurements at 1280x720 and 1920x1080.
- `benchmarks/lidar_sim.py` and `benchmarks/lidar_sim.json`: a synthetic lidar benchmark (simulation) with two detectors and four faults; `benchmarks/casper_imu_alignment.py` and its JSON: the IMU alignment attempt.
- `benchmarks/casper_hours.py` and `benchmarks/casper_hours.json`: false alarms of the headline CASPER configuration on every normal hour.
- `ARCHITECTURE.md` and `docs/api/general.md` list the bandit, active, factorization, recommender and small-structure packages.

### Fixed
- `OnlinePlattScaling` (`learn/calibration.py`): the annotation `base.Classifier` named a module that does not exist; it is `Classifier`. `from __future__ import annotations` is removed from the seven files that had it, since Python 3.10 supports the syntax directly.

### Changed
- The claim register and the table of numerical claims left the published docs: they are now
  `benchmarks/claims_register.md` and `benchmarks/numbers_register.md`. The navigation lists
  Speed once.

## [1.1.29] - 2026-10-16

### Changed
- Speed (#184): ADWIN rewritten as ADWIN2 (exponential buckets, Hoeffding cut,
  Bifet and Gavalda 2006), 13x to 38x faster per update depending on the
  machine; `AdaptiveRandomForestClassifier.predict_one` no longer builds the
  probability dict (5x to 7x faster); the Hoeffding tree checks for a split
  every `grace_period` samples (1.6x to 1.8x faster). Before and after on
  three machines in `benchmarks/speed_log.md`, harness in `benchmarks/speed.py`.
- ADWIN can flag a change on several consecutive updates; the docs say so and
  the robot-stream table of `docs/drift/detectors.md` is measured again with
  this version (17.7 s delay, 0 missed, 1.4 false alarms per hour).
- `check_estimator` computes the ROC AUC with its own Mann-Whitney U, with no
  optional dependency.

### Fixed
- The provenance signature of `dense_armor.core` follows the package version
  (it fell back to 1.0.11) (#187).
- `SafeEstimator.restore` reads the checkpoint envelope with an unpickler that
  refuses every global and verifies the digest before the model state is
  unpickled; `save` and `restore` take an optional `key` for an HMAC-SHA256.
  The state is still a pickle: restore only trusted checkpoints (#187).
- MCP streaming sessions are capped at 256 and dropped after an hour idle;
  importing `dense_armor.dynamics` no longer loads the QP solver; `nvidia-smi`
  is called without a shell; `one_sided_arl` returns `inf` on exponent overflow
  instead of a warning (#187).

### Added
- Quality benchmark of the building blocks (20 seeds, 3 streams) and the claim
  register `docs/quality.md` (#181, #183); `check_estimator` reports "not
  applicable" checks instead of skipping them silently.
- Tests for the lines of the speed patch (#186).

## [1.1.28] - 2026-10-15

### Added
- Naive Bayes (`dense_armor.utility.bayes`): `MultinomialNB`, `BernoulliNB`,
  `ComplementNB`, `CategoricalNB`; distributions (`proba`): `Beta`, `Gaussian`,
  `Multinomial` with a pure `init_state` / `step` form; nearest neighbours
  (`neighbors`): `KNNClassifier`, `KNNRegressor`, `LazySearch`; multiclass
  wrappers (`multiclass`): `OneVsRestClassifier`, `OneVsOneClassifier`,
  `OutputCodeClassifier`; multi-output estimators (`multioutput`): per-output
  models and classifier, regressor and probabilistic chains. Five docs pages
  with figures (#167).
- Forests (`dense_armor.utility.forest`): `AdaptiveRandomForestClassifier`,
  `AggregatedMondrianForestClassifier` and `AggregatedMondrianForestRegressor`
  (#168), with a bound on the size of every tree (#170);
  `OnlineExtraTreesRegressor` (#173). Docs page `docs/forest/`.
- Rules (`dense_armor.utility.rules`): `AMRulesRegressor` (#172). Docs page
  `docs/rules/`.
- Anomaly detection: `LocalOutlierFactor` (#166) and the Streaming Half-Space
  Trees of Tan et al. 2011 (#157).
- Finance: nine pipelines (markets, fraud, intrusion, credit scoring), one page
  each (#161), with the numbers remeasured on closing prices and figures (#162).
- README: a five-feature online classifier quickstart, "Who is this for", "What
  it expects", "What changes" and a use-cases section (#147, #148, #149).
- One plain-words sentence at the top of every docs page (#160).

### Changed
- `check_estimator` runs with no declared skip on the classes of `bayes`,
  `neighbors`, `multiclass`, `multioutput` (#171) and `forest` (#169).
- API reference grouped by the real module (#152); docs of the anomaly pages
  aligned with the code (#154).

### Fixed
- LODA: the projection is normalised with its theoretical minimum and maximum
  (#153). `OnlineRobustMahalanobis`: the step index counts from the first sample
  (#156).
- Time series: code matched to the three papers, with a rolling cycle template
  (#163). Optimizers, linear models and payload estimator: fixes from the review
  (#165).
- `LocalOutlierFactor` gives finite scores on duplicates and a documented value
  on malformed input (#166).

Every commit since 1.1.27 is listed in the release notes on GitHub.

## [1.1.27] - 2026-10-13

### Added
- Time series package (`dense_armor.utility.timeseries`): `OnlineAR`, `OnlineVAR`
  (TIRSO, Zaman 2019 eq. 25 with the `1{n != n'}` indicator on the group-lasso
  penalty), `KalmanFilter`, `VBAdaptiveKalman` (Mbalawata 2013 Algorithm 1),
  `InnovationAdaptiveKalman` (Popescu 2026), `CycleResidual` (period detection
  by autocorrelation, median template, bounded buffer). Four new docs pages
  under `docs/timeseries/` and three figures.
- Linear models (`dense_armor.utility.linear`): `LinearRegression`,
  `LogisticRegression`, `BayesianLinearRegression` (predictive interval),
  `PayloadEstimator` (tracking figure). Docs under `docs/linear/`.
- Online optimizers (`dense_armor.utility.optim`): `SGD`, `AdaGrad`, `Adam`,
  `FTRLProximal` (McMahan 2011), `OnlineNewtonStep`. Each with a learning-rate
  choice figure and a docstring regret bound. Index page under `docs/optim/`.
- Three new comparison pages: `docs/vs_kalman.md`, `docs/robustness.md`
  (adversarial attacks on the shields, weakest point stated), `docs/limits.md`.

### Changed
- `core/` (14 files): docstrings in English, inline `#` comments removed,
  missing type hints added (`chunk.py`, `profiler.py`, `vector.py`). No
  behavioural change.
- README restructured: short answers at the top ("Who is this for",
  "Use cases", "What it expects from you", "What changes when you use it")
  before the long part. Quickstart is now a 5-feature online classifier on a
  drift stream with the measured accuracy, followed by a rate-limiter example
  on a real Panda URDF.
- `pyproject.toml`: `ruff` added to the `[docs]` extra so `mkdocstrings`
  formats signatures without the "install Black or Ruff" warning.

### Fixed
- `griffe` warnings in `mkdocs build --strict` for `codegen_engine`,
  `codegen_instance`, `stabilizer_instance`, `base_state`: missing parameter
  annotations. Strict build now passes.
- `dense_armor/__init__.py`: `__version__` was at `1.1.25` while
  `pyproject.toml` was at `1.1.26`. Both now at `1.1.27`.

## [1.1.25] - 2026-10-09

- Streams, datasets and pipelines: `dense_armor.utility.stream` (`iter_array`, `iter_csv` with constant memory and list-valued columns, `iter_rosbag` for ROS 1 and ROS 2 bags with the new extra `[ros]`, `merge_by_time`, `shuffle`); `dense_armor.utility.datasets` (`Casper` reader for a local copy of the CASPER UR3e dataset, `DriftStream` with sudden, gradual and incremental drift after Ksieniewicz and Zyblewski 2020, `SyntheticArm` labelled joint stream from a URDF with friction or payload faults); `dense_armor.utility.compose` (`Select`, `Discard`, `FuncTransformer`). `Protected` passes `t` only to models that accept it; `HampelScorer.threshold` exposes `n_sigmas` for `AnomalyGate`.

- Incremental trees and online forests (`dense_armor.utility.tree`): `HoeffdingTreeClassifier` (Hoeffding bound, `explain_one`), `HoeffdingAnytimeTreeClassifier` (statistics on every node, splits revisited and replaced or collapsed; Manapragada 2018), `HoeffdingAdaptiveTreeClassifier` (a drift detector on each node, alternate subtrees), `SGTRegressor` / `SGTClassifier` (stochastic gradient trees with the t-test of Gouk 2019, eq. 16 and 18), `MondrianForestRegressor` / `MondrianForestClassifier` (online Mondrian trees, Algorithms 2 and 4 of Lakshminarayanan 2014, predictive mixture of eq. 3 of the 2016 paper, variance growing away from the data).
- `OnlinePlattScaling`: while the base classifier has no opinion yet (empty `predict_proba_one`), `learn_one` trains the base only and leaves the calibration untouched; a base that knows only the negative class gives `p = 1 - p(False)` instead of 0.5. Phishing log-loss of the docstring example: 0.3499.

## [1.1.24] - 2026-10-08

- Patch-level anomaly detection in the vision package: `PatchFeatures` (oriented-gradient histogram, mean and contrast per patch, averaged over a 3×3 neighbourhood) and `PatchMemory` (bank of normal patches, image score = largest nearest-neighbour distance, PatchCore eq. 6), online with a bounded reservoir or reduced once with the greedy coreset of eq. 5 (Roth 2021); `benchmarks/vision_rad_patches.py`: mean image-level AUROC 0.748 on RAD at 320×240 (baseline 0.497).
- Robot checks call a `learn_one(x, t)` with the sample alone, so the timestamp is never set to the label placeholder.

- Online clustering (`dense_armor.utility.cluster`): `OnlineKMeans` (sequential, optional half-life for drifting streams), `DenStream` (micro-clusters with a damped window, outliers kept apart, macro-clusters on demand); `BagOfVisualWords` in the vision package (k-means vocabulary over patches, word histogram per frame).

- Preprocessing (`dense_armor.utility.preprocessing`): online scalers per feature and per joint (standard, exponentially weighted, robust, min-max); joint velocity, acceleration and jerk on real timestamps without lag, joint power; text tokenizer, bag of words, TF-IDF (Silajev 2026), signed feature hashing (Weinberger 2009); variance threshold and k-best selection; queue-based resampling for imbalanced streams (Malialis 2018).

- Native online vision (`dense_armor.utility.vision`): `CameraStream` (OpenCV, extra `[vision]`), `ImageFolderStream`, `ArrayStream`; `FrameFeatures` (oriented-gradient histograms, intensity moments, Lucas–Kanade flow); `RandomProjection` and `IncrementalPCA`.
- `benchmarks/vision_rad.py`: native vision on real robot images (RAD); mean image-level AUROC 0.497 (chance level), reported as a baseline.
- `FrameFeatures(flow_levels=n)`: coarse-to-fine Lucas–Kanade on a 2×2-average pyramid with bilinear warping (Ziani 2025, section 3.2, eq. 9); `gray=False` adds per-channel colour moments.
- Pipelines built with `|` now train every intermediate step (transform, then learn) and pass `t` only to steps that accept it.

## [1.1.23] - 2026-10-08
- Online metrics (`dense_armor.utility.metrics`) and evaluation (`dense_armor.utility.evaluate`): classification, regression, intervals, rolling windows and ROC-AUC, prequential with delayed labels, model selection, event metrics (range-based precision/recall, NAB, detection delay, false alarms per hour).
- `AdaptiveConformalRegressor` passes `t` only to models that accept it.
- Drift detectors `PageHinkley`, `ADWIN` (Hoeffding bound with online range) and `KSWIN`; `CUSUMDriftDetector(reference="fixed")` no longer stays silent after a NaN or flat start; `benchmarks/drift_comparison.py` compares the four on the CASPER stream.

### Changed
- **`utility/orca.Orca.protect_and_forward`**: vectorized the entry-shield stage -- the
  previous `for b in range(B)` Python loop called `_execute_4_phase_input_shield` once per
  row (once per element of the batch), each a separate JAX dispatch. Same dispatch-dominated
  pattern already found and fixed in `core/hybrid_engine.hybrid_shield` for Armatura. New
  `_execute_4_phase_input_shield_batch` processes all B rows in one call via a new
  `AdaptiveSignalStabilizer.filter_batch_scenarios_independent` (`core/engine.py`): each row
  gets its OWN calibration (`calibrate_macro_context_independent`), not one shared across the
  whole batch like the existing `filter_batch_scenarios` -- rows with different scale/noise no
  longer contaminate each other's threshold, matching the old loop's per-row-independent
  behavior exactly. Verified bit-for-bit (float64) against the loop version, including a test
  that specifically checks one row's huge injected spike does not change another row's output
  (`test/test_orca_vectorized_batch.py`). Real measured latency, synthetic B=500/F=10 batch
  (same shape as the APT29 + AI-shield experiment): median of 10 calls after JIT warmup,
  **750.6ms -> 2.3ms (~321x)**. `chunk_threshold` behavior unchanged for the one case this
  doesn't cover (a single row's own flattened size exceeding `chunk_threshold`, which falls
  back to the original per-row/per-chunk loop, untouched). No public API change.

### Changed
- **`core/hybrid_engine.hybrid_shield`**: vectorized with `jax.vmap` -- the previous
  `for i in range(2, n)` Python loop called `calculate_phi_ab`/`calculate_vettore_dinamico`/
  `evaluate_phi_trigger` once per point. Profiling on a real 500-point security-telemetry
  series (Sysmon EventID10, APT29) showed 87% of the time in JAX dispatch itself (~2000
  `jnp.array` constructions, 3.2M `isinstance` calls), not in the actual per-point math --
  pure Python-orchestration overhead, not a real compute limit. Verified bit-for-bit against
  the original loop on that same real series and on synthetic edge cases (isolated spike,
  sustained step, with/without an external reference, n=0..3): identical output and trigger
  in every case. Real measured latency on that 500-point series (median of 20 calls after JIT
  warmup): **292.4ms -> 29.8ms (~9.8x)**. No public API change; `baseline`/`scale`/`ipg` still
  read only `processed` (the sanitized raw series), never `out` from a previous step, per the
  no-self-contamination invariant this function has always documented.

## [1.1.19] - 2026-09-05

### Added
- **`dynamics/passivity_cbf_controller.py`** (`solve_control_qp`): task-space passivity +
  singularity-avoidance CBF controller (Kurtz, Wensing & Lin 2021, arXiv:2109.13349) for any
  robot `RigidBodyModel` can load. A small QP over joint acceleration, constrained by `Vdot<=0`
  (passivity) and an exponential CBF keeping the manipulability index above a declared floor.
  Two-step promotion from Dense-Evolution-Discovery, same as `RigidBodyModel`: Experiment 61
  built the same controller hardcoded to one Kinova Gen3; Experiment 63 replaced the hardcoded
  calls with `RigidBodyModel`'s API and re-validated on the same three robots (Kinova Gen3
  7-DoF, Kinova Gen3 6-DoF, Franka Emika Panda), each driven toward its own true singularity,
  manipulability held within 0.1-1.8% of the declared floor in every case. Found and fixed a
  real OSQP-infeasibility bug along the way (soft passivity constraint dropped in favor of the
  hard CBF one when the solver reports infeasible). New dependency: `osqp`.
- **Real per-joint position/velocity limits**: `RigidBodyModel` now parses each joint's real
  `<limit>` tag (`+/-inf` where the URDF declares none); `solve_control_qp` enforces them as an
  additional CBF (Kurtz et al.'s own "joint" constraint type), added only when a robot's URDF
  has a real finite limit somewhere. Real number, Franka Panda `joint4` (range `[-3.1416,
  0.0]`) at its bound with velocity driving past it: unconstrained command `qdd=-205.8` clamped
  to the real box `[-7.175, -4.999]`.
- **`dynamics/six_dof_pbc_cbf_controller.py`** (`solve_control_qp`): full 6-DoF (position +
  orientation) generalization of `passivity_cbf_controller.py`, tracking a link's full pose via
  `RigidBodyModel`'s new `link_pose`/`link_spatial_jacobian`. Attitude error uses Lee, Leok &
  McClamroch (2010)'s SO(3) formula instead of an RPY-based one, avoiding gimbal lock. Promoted
  from Dense-Evolution-Discovery Experiment 65: validated with an exact gravity-compensation
  check at zero position and orientation error (machine precision), and a real closed-loop run
  converging a 10cm/30-degree offset to near-zero (position 1e-6 m, orientation 1e-4) over 1000
  control ticks; also validated on the same 3-robot bar as `passivity_cbf_controller.py`, which
  surfaced a real second-level OSQP infeasibility (the 6-DoF manipulability measure can be well
  below the 3-DoF one at the same configuration, making CBF+box jointly infeasible) -- fixed
  with a third fallback level that drops the box and keeps only the CBF.
- **`.xacro` file support**: `RigidBodyModel` now accepts `.xacro` macro files directly,
  expanding them via the real `xacro` package before parsing. Promoted from Dense-Evolution-
  Discovery Experiment 66, which found and fixed a real inconsistency in the Franka Panda's own
  published macros (`clvrai/furniture`): a hand-attachment link commented out in the arm macro
  but required by the hand macro. New dependency: `xacro`.
- **`<mimic>` joint support**: a joint with a `<mimic joint="..." multiplier="..."
  offset="..."/>` tag no longer gets its own independent coordinate -- its motion is computed
  from its master's, chain-ruled onto the master's own column in the hand-built geometric
  Jacobian. Promoted from Dense-Evolution-Discovery Experiment 67, checked against a real
  central finite difference of `link_pose` (Franka Panda gripper fingers) to under 1e-5.

## [1.1.18] - 2026-09-05

### Added
- **`dynamics/urdf_dynamics.py`** (`RigidBodyModel`): Euler-Lagrange rigid-body dynamics --
  M(q), C(q,qdot)qdot, g(q) -- parsed from any URDF file (stdlib xml.etree, no new dependency),
  not one hardcoded robot. New `dynamics/` subpackage, separate from `utility/`, since it needs
  an actual physical model file and returns torque-level quantities. Two-step promotion from
  Dense-Evolution-Discovery: Experiment 61 built the same dynamics (JAX autodiff, not
  hand-derived Christoffel symbols) but hand-transcribed from one Kinova Gen3's URDF -- the
  reason it wasn't promoted then. Experiment 62 replaced the hardcoded tables with a parser and
  re-validated on three independent robots (Kinova Gen3 7-DoF, Kinova Gen3 6-DoF, Franka Emika
  Panda -- different manufacturer, prismatic joints): cross-checked to machine precision
  against Experiment 61's own numbers, then mass-matrix SPD and energy-conservation (correct
  4th-order RK4 convergence) verified fresh on all three.

## [1.1.17] - 2026-09-04

### Added
- **`utility/cbf_filter.py`** (`cbf_safety_filter_live`): single-control-tick counterpart to
  `cbf_filtered_trajectory` -- for a real-time loop reacting to one sensor callback at a time
  (a real `dt` per tick) instead of filtering a whole pre-recorded array offline. Promoted
  after a real live ROS2/Ignition safety loop (Dense-Evolution-Discovery Experiment 58)
  needed exactly this and had to reconstruct it by hand from `cbf_filtered_trajectory`.
  Reuses the same substep math already validated on SO-101/ALOHA -- not a new validation
  domain, an ergonomic single-tick wrapper around it. Verified directly: converges exactly
  to the declared safety boundary and never crosses it (max reached 2.199999999999889 for a
  2.2 boundary).
- **`utility/trajectory.py`** (`quintic_trajectory`): closed-form, minimum-jerk-continuous
  point-to-point trajectory generator for any number of joints -- fills the gap
  `rate_limiter`/`cbf_filter` don't cover (neither generates a reference to track).
  Deliberately scoped down from two real papers proposing much larger URDF/dynamics-aware
  optimizers (Lozer et al., Robotics and Autonomous Systems; Fried & Paternain,
  arXiv:2412.07859 -- both read in full before writing any code). Promoted from
  Dense-Evolution-Discovery after validation on two independent real physical domains
  (SO-101, ALOHA, 20 real joint excursions): quintic peak velocity always lower than the
  real recorded peak velocity for the same start/end/duration -- expected, since it is the
  smoothest possible path, not a bug.
- **`utility/kinematic_controller.py`** (`kinematic_tracking_controller`): closed-form
  feedforward-plus-proportional kinematic tracking controller -- turns a reference from
  `trajectory.py` into a velocity command for `rate_limiter`/`cbf_filter`. Searched for a
  real "universal controller" paper first (Wu & Tan 2025 -- best fit, paywalled; Scruggs --
  real, needs infinite-dimensional convex Youla optimization; Califano et al. -- real, needs
  Hamiltonian mechanics; classical PD+gravity-compensation -- needs a real dynamics model,
  contradicting this stack's single-integrator scope). What ships instead: `u = qd_ref +
  kp*(q_ref - q)`, with an exact closed-form exponential convergence guarantee, verified
  numerically. Promoted from Dense-Evolution-Discovery after validation on two independent
  real physical domains (SO-101, ALOHA, 20 real joint excursions), chained with
  `quintic_trajectory`: every real excursion recovers from a real disclosed nonzero initial
  tracking error and converges.

## [1.1.16] - 2026-09-04

### Fixed
- **`dense_armor/__init__.py`**: `__version__` was still hardcoded to `"1.1.14"` after the
  1.1.15 release -- it drives `dense_armor_health`'s `dense_armor_version` field in the MCP
  server, so the running package misreported its own version. Real regression found by
  checking `dense_armor.__version__` right after installing 1.1.15 from PyPI, not caught
  before that release. `pyproject.toml`'s `version` field was already correct; only the
  Python-level constant was stale.

## [1.1.15] - 2026-09-04

### Added
- **`utility/cbf_filter.py`** (`cbf_safety_filter`, `cbf_filtered_trajectory`): geometric
  Control Barrier Function safety filter -- never lets an applied command enter a
  forbidden region, complementing `rate_limiter.py`'s rate-of-change bound. Grounded in
  Ames et al. (2019), "Control Barrier Functions: Theory and Applications" (2019 ECC,
  arXiv:1903.11199) -- the same theory SAFER-Splat uses, applied to a known geometric
  obstacle instead of SAFER-Splat's GPU-bound Gaussian-Splatting perception. Real
  numerical finding disclosed, not hidden: the CBF's continuous-time guarantee needs
  discrete sub-stepping (20/sample default) to hold on real commands that can jump
  substantially between samples. Promoted from Dense-Evolution-Discovery after validation
  on two independent real physical domains (SO-101 6-DoF 30Hz, ALOHA bimanual 14-DoF
  50Hz): 100% invariance from safe starting conditions on both (17/17, 38/38), minimal
  invasiveness 99.9%+ exact on SO-101 and perfectly exact (0/18444) on ALOHA.
- **`utility/rate_limiter.py`** (`rate_limited_follower`): causal velocity+acceleration
  command limiter -- bounds how fast an applied command can physically change instead of
  classifying whether a deviation is real, after a causal neighbor-consensus classifier
  approach hit a structural dead end (1.7% win rate vs a trivial moving median, 120 real
  trials). Grounded in Berscheid & Kroger (2021), "Jerk-limited Real-time Trajectory
  Generation" (RSS 2021, arXiv:2105.04830) -- causal by construction, verified directly.
  Promoted from Dense-Evolution-Discovery after validation on two independent real physical
  domains (SO-101 6-DoF 30Hz, ALOHA bimanual 14-DoF 50Hz): wins the real safety metric (max
  instantaneous command jump) 400/400 real trials across both domains. Honest limit, not
  hidden: average tracking fidelity (RMSE vs the true signal) does not reliably beat a
  trivial moving median -- this is a safety bound, not a signal cleaner.
- **3 new stateful MCP tools** (`dense_armor_stream_start`/`_update`/`_end`,
  `dense_armor.mcp_server`): a real-time streaming session wrapping
  `MultiChannelStreamingDeviationDetector` -- for a live sensor feed (robot
  joints, IMU axes) where waiting to collect a full array first isn't an
  option, unlike the existing batch `dense_armor_detect_anomalies` tool.
  One `session_id`-keyed detector instance per session, held in server
  memory (in-process, same architecture as every other tool here -- see
  server.py's own docstring); a real caller must call `_stream_end` when a
  real stream ends, sessions are not garbage-collected automatically.
  Verified end-to-end against real LeRobot robot-arm joint data fed one
  point at a time through the real MCP call path (`mcp.call_tool`), not
  just the underlying Python function directly.

### Fixed
- **`utility/curvature.py`** (`curvature`): added a `scale` parameter (default `1.0`,
  reproduces the exact previous behavior -- `Orca`'s existing call site unaffected). Real
  finding, checked directly against real SO-101 joint data (Dense-Evolution-Discovery):
  the unscaled formula saturates to ~1.0 within ~5 raw units of any reference regardless
  of the caller's physical units, making it a near-binary near/far indicator rather than a
  graded proximity signal when used in real degrees (`elbow_flex`, whose real distance to
  its real physical limit spans 0.1-66.9 degrees in one real episode: correlation with raw
  distance only 0.235 at the old fixed scale, vs 0.932 with a physically meaningful
  `scale=15.0`). Not a promotion (no second-domain validation attempted -- this fixes an
  existing internal utility's own parameterization, not a new one).

## [1.1.14]

### Added
- **`utility/streaming.py`** (`StreamingDeviationDetector`,
  `MultiChannelStreamingDeviationDetector`, `classify_segments_
  multichannel`): porta a latenza zero solo la meta' causale di
  `classify_segments` (arbiter.py) -- il flag di deviazione per-punto,
  non l'etichetta finale spike/regime, che guarda `radius` punti avanti
  alla fine di una sequenza deviante e resta una domanda batch/offline
  per design, non una svista. Buffer semplice (deque, O(span)), non una
  struttura a due heap -- misurato a ~18.6kHz sostenibile per le
  finestre gia' usate in tutto il progetto (10-100 punti), oltre 180x
  il tasso reale di un anello di controllo robotico (30-100Hz). Il
  supporto multi-canale rimuove il bisogno di un ciclo manuale
  per-canale (giunti di un braccio robotico, assi di un IMU), ogni
  canale con la propria finestra di riferimento indipendente. Promosso
  da Dense-Evolution-Discovery (Esperimenti 48-49) dopo validazione su
  due domini fisici reali indipendenti (braccio robotico teleoperato
  SO-101, IMU umano reale UCI HAR) -- stessa disciplina di promozione
  gia' usata per `stable_frame_filter.py`. Corrispondenza bit-per-bit
  con `classify_segments` verificata direttamente, non assunta.
- **`utility/cusum.py`** (`one_sided_arl`, `two_sided_arl`, `detectability_report`): stima
  pre-flight, teoria classica Reynolds (1975)/Siegmund (1985), di quanti campioni servono
  per rilevare uno shift dato il rumore locale reale di `cusum_detector` -- senza dover
  lanciare un benchmark prima. Promossa da Dense-Evolution-Discovery dopo validazione su
  due domini fisici reali indipendenti: lidar (7/7 punti reali, latenza sempre sotto la
  stima) e accelerometro (5 punti reali, risultato genuinamente misto 2/5 -- non forzato a
  coincidere con il lidar). L'accelerometro ha anche fatto emergere un limite reale della
  formula grezza a SNR estremo (ARL frazionario, privo di senso fisico sotto 1 campione) --
  corretto qui con un floor a 1.0 su entrambi gli ARL.


## [1.1.13]

### Fixed
- **`utility/cusum.py` (`cusum_detector`)**: il default `h=5.0` ("classico
  da manuale", mai validato contro una lunghezza di serie pratica) dava
  un tasso di falso allarme a livello di STREAM del 100% (`adaptive`) /
  85% (`fixed`) su una serie stabile di 1000 campioni -- invisibile al
  test esistente, che controllava solo il tasso di falso allarme per
  PUNTO (molto più basso e fuorviante). Trovato come sottoprodotto
  diretto di un porting parallelo dello stesso algoritmo per
  the same problem first appeared in an existing online-learning library.
  Nuovo default `h=20.0`: 3.5% (`adaptive`) / 15.5% (`fixed`) sulla
  stessa serie di 1000 campioni. `fixed` resta strutturalmente più
  esposto (il suo riferimento non si aggiorna mai, quindi una stima di
  warmup sfortunata non si autocorregge) -- non un difetto di taratura,
  conseguenza diretta di cosa `fixed` è pensato per fare. Nuovo test di
  regressione (`test_default_h_keeps_stream_level_false_alarm_rate_low_
  on_a_practical_length`) per non ricadere silenziosamente nello stesso
  problema. Nessun benchmark preregistrato è stato toccato: tutti e
  quattro (`test_benchmark_v0_runtime_behavioral_drift.py`,
  `test_benchmark_v2_1_ablation.py`, `test_benchmark_v2_1_log_
  onesided.py`, `test_benchmark_v2_agent_runtime.py`) passano `h=5.0`
  esplicitamente nel proprio `CUSUM_KW` congelato, indipendentemente dal
  default della funzione.

### Added
- **`utility/stable_frame_filter.py` (`velocity_gated_stable_mask`)**: filtro
  che restringe l'analisi di un segnale ai punti dove un segnale di
  riferimento COMPANION indica uno stato stabile/lento -- evita che un
  detector interpreti come anomalia un transitorio spiegato da un
  confondimento noto e indipendente (es. il comando/leader di un braccio
  robotico che si muove veloce, causando un lag di tracking normale).
  Parametro `already_rate` per riferimenti gia' di tipo velocita' (es.
  magnitudine giroscopica) invece che posizione -- trovato necessario da
  un fallimento reale su un secondo caso, non aggiunto preventivamente.
  Promosso da Dense-Evolution-Discovery dopo validazione su due domini
  fisici reali indipendenti (braccio robotico teleoperato SO-101, IMU
  umano reale) -- stessa disciplina di promozione gia' usata per
  `one_sided_upper_filter`.

## [1.1.12]

### Added
- **`utility/cusum.py` (`cusum_detector`)**: canale complementare a
  `classify_segments` per il drift lento/sostenuto -- CUSUM a due code
  (Page 1954), modalita' `adaptive` (default, finestra causale scorrevole,
  dichiarata onestamente come variante non identica allo schema originale
  a riferimento fisso) e `fixed` (lo schema classico). Gestione NaN/Inf
  esplicita. Nato da un gap reale misurato in
  `test/test_benchmark_v0_runtime_behavioral_drift.py`:
  `classify_segments` rileva un glitch transitorio al 100% (latenza 0) ma
  solo il 9-27% di un drift graduale nella finestra di transizione.
- **`utility/one_sided.py` (`one_sided_upper_filter`)**: filtro a una
  coda componibile con `classify_segments`/`cusum_detector` -- per segnali
  dove solo un AUMENTO e' significativo (latenza, error rate). Riduce i
  falsi positivi misurati su telemetria reale di un agente Qwen2 1.8B
  (via Ollama) dal 22.5% al 12.5% (`classify_segments`) e dal 17.5% al
  10.0% (`cusum_detector`), e il falso rifiuto su un cambio di
  comportamento legittimo dal 24.0% al 4.0% su entrambi -- al costo reale
  di una sensibilita' ridotta al drift persistente (non nascosto: vedi
  `test/test_benchmark_v2_1_ablation.py`). Un log-transform provato in
  parallelo non ha portato benefici misurabili ed e' stato scartato.

### Investigated (percorso di validazione, non solo il risultato finale)
- Benchmark sintetico congelato (`test/test_benchmark_v0_runtime_
  behavioral_drift.py`): protocollo preregistrato, 4 condizioni (baseline,
  drift, glitch, test negativo), mai ritoccato dopo aver visto i
  risultati.
- Benchmark su agente reale (`test/agent_v2/`, `test/test_benchmark_v2_*`):
  prima validazione non sintetica -- loop minimale con Qwen2 1.8B via
  Ollama (tool-calling manuale, il modello non restituisce `tool_calls`
  strutturati nonostante il tag "tools"), 4 scenari con ground truth
  esatta. Trovato: la latenza reale ha una coda molto piu' pesante del
  rumore gaussiano sintetico, da cui il filtro one-sided sopra.

## [1.1.11]

### Investigated (nessun candidato promosso)
- **`compute_damping_gating_smooth` (Collatz continuo), mai benchmarkato,
  ora chiuso**: il suo stesso docstring rimandava a
  `test/test_collatz_smooth_experiment.py` per il confronto misurato --
  file mai esistito. Scritto ora, gira `Orca` per intero sui 7 scenari di
  `test/testKalman.py` una volta con il gate costante (default) e una con
  la variante smooth. Risultato: smooth vince 0/7, pareggia 6/7 (RMSE
  identico entro 1e-4), perde su "Stealth sub-soglia" (RMSE 0.0029 vs
  0.0012, oltre il doppio). Non promosso -- docstring aggiornato con il
  risultato reale al posto del rimando a un file inesistente.
- **`hybrid_shield` (phi_ab, gia' vendorizzato per Armatura) come possibile
  gate per Orca**: non e' uno scambio diretto (motore binario a ciclo su
  tutta la serie, non un gate continuo per elemento), quindi confrontato
  come pipeline completa (`test/test_hybrid_shield_vs_orca_experiment.py`)
  sugli stessi 7 scenari. Risultato misto, non un vincitore netto:
  vince 4/7 su impulsi isolati/collasso a zero/buco NaN (in 3 casi RMSE
  ~0.0000, quasi perfetto), ma perde nettamente su segnale sinusoidale
  strutturato e rumore Cauchy pervasivo (RMSE 2-3x peggiore di Orca) --
  ottimo per anomalie isolate, pessimo quando il cambiamento e' diffuso e
  genuino, non un'anomalia da respingere.
- **`pressure_valve` (JSD-adattivo, gia' in `robust_filters.py`, mai
  collegato a nessuna pipeline) come terzo candidato**: stesso confronto
  a 7 scenari (`test/test_pressure_valve_vs_orca_experiment.py`). Vince
  1/7 ma in modo netto: RMSE 0.0000 esatto sulla rottura di livello
  permanente (l'unico dei tre a riconoscere un vero cambio di regime),
  ma fallisce clamorosamente sugli impulsi isolati (RMSE 86.6 su
  "impulsi alternati", peggio di non fare nulla). Causa reale trovata
  (non un'ipotesi): con 3 picchi ravvicinati in una finestra di 21 punti,
  mediana/IQR e mediana/MAD restano ESATTAMENTE 0 (18/20 punti identici,
  matematicamente corretto per stimatori con breakdown ~25-50%), quindi
  vengono scartati e resta solo media/std -- non robusta, contaminata
  dagli stessi picchi che dovrebbe rilevare. Fix parziale applicato: la
  finestra locale ora esclude il punto stesso (come gia' faceva solo il
  sotto-passo sigma-clipping), corretto in linea di principio ma
  verificato non sufficiente quando PIU' outlier restano comunque vicini
  tra loro nella stessa finestra (nessun cambiamento misurabile sui 7
  scenari). Un M-estimator di Huber (IRLS, vedi Sung et al. 2019,
  arXiv:1912.04982, ricerca bibliografica reale via RAG locale) e' stato
  provato come alternativa piu' sofisticata e converge ANCH'ESSO a scala
  0, per lo stesso motivo di fondo (la sua scala e' a sua volta una MAD
  dei residui). Conclusione onesta: non e' un problema di scelta
  dell'estimatore, e' che dividere per una scala locale legittimamente
  zero (baseline quasi piatta) e' indefinito per costruzione, qualunque
  stimatore si usi. Non risolto -- vedi il "LIMITE NOTO" nel docstring di
  `pressure_valve` per la via di fix non ancora tentata (scala di
  fallback dalla finestra di riferimento piu' ampia).
- **Conclusione dei tre esperimenti**: nessuno dei tre candidati batte in
  modo netto il gate costante 0.85 di Orca su tutti gli scenari --
  ognuno (incluso il default) ha una nicchia precisa (impulsi isolati,
  segnale diffuso/rumore pervasivo, cambio di regime permanente). Il
  gate costante resta quello di default; nessuna promozione fatta.

### Fixed
- **`core/vector.py`, `run_parallel_scenarios` ricompilava ad ogni chiamata**: `jax.jit(jax.vmap(...))`
  veniva ricostruito DENTRO il metodo ad ogni invocazione -- una nuova closure Python ad ogni
  chiamata, quindi un nuovo oggetto `jax.jit` la cui cache di compilazione non poteva mai fare
  hit con la precedente. Il "warm run" restava bloccato a ~130ms invece di scendere ai pochi ms
  attesi (la diagnosi originale, un report Colab, puntava al cast `np.array` come colpevole --
  verificato e smentito: il vero costo era la ricompilazione ad ogni chiamata). Fix: motore
  jit+vmap spostato a livello di modulo, costruito una sola volta; `base_state` passato come
  argomento tracciato invece che chiuso in closure, cosi' valori diversi non forzano una nuova
  compilazione. Verificato: da un plateau di ~140ms/chiamata a ~2.4ms/chiamata (~55x) su chiamate
  ripetute al metodo pubblico.
- **`utility/orca.py`, riferimento pulito che attraversa lo zero esplodeva la compressione log10**:
  un valore pulito quasi-zero (anche solo un residuo di floating point, mai esattamente `0.0` --
  es. `sin(4π) ≈ -4.9e-16`) faceva esplodere il fattore di scala condiviso `fact_shared`. Siccome
  lo stesso fattore si applica anche al corrotto (che NON e' vicino a zero), il suo controvalore
  compresso schizzava a ordini di grandezza sopra i punti vicini (misurato: da `~1e-4` tipico a
  `~2e9` in un caso), sballando la memoria causale di `AdaptiveSignalStabilizer` per l'intero
  batch, non solo per quel punto. Fix a due livelli: (1) l'esponente non viene piu' amplificato
  oltre la scala target (`val_e`); (2) il valore compresso del corrotto resta comunque limitato a
  un multiplo ampio (20x) della scala target, indipendentemente dalla causa esatta del salto.
  Verificato: MSE su un riferimento sinusoidale che attraversa lo zero da ~9 (peggio della
  modalita' cieca sullo stesso segnale) a ~0.02.

### Changed
- **`core/engine.py`, naming e precisione non armonizzati col resto della libreria**: le variabili
  interne di `AdaptiveSignalStabilizer` (`_step_kernel` e la variante storica `_legacy_step`) si
  chiamavano `phi_ab`/`v_dyn`/`trigger` -- stessi nomi di `core/hybrid_engine.py` (motore di
  Armatura) per una formula matematicamente diversa. Rinominate in modo non ambiguo
  (`coherence_signal`/`coherence_mix` e `coherence_legacy`/`dynamic_shift_legacy`/
  `is_anomaly_legacy`). Il file usava anche `float32` esplicito ovunque, mentre il resto della
  libreria richiede `jax_enable_x64` e gira in `float64` -- i commenti giustificavano il
  `float32` con una "compatibilita' di esportazione binaria nativa" che non esiste da nessuna
  parte nel repo (verificato con una ricerca sull'intero codebase). Convertito tutto a `float64`,
  rimossi i commenti fuorvianti. Suite completa (189 test, inclusi i 5 file adversarial
  `test_bound*.py`) invariata prima e dopo entrambe le modifiche.

### Added
- **`Orca` usa ora `UniversalMemoryGuard` invece di un check RAM ad-hoc**: `_gc_se_ram_bassa()`
  duplicava a mano un check `psutil` gia' fatto meglio da `core/memory.py` (che copre anche la
  VRAM, mai controllata prima). Cambio di comportamento onesto: sotto la soglia critica ora
  solleva `MemoryPressureError` invece di continuare in silenzio -- fail-fast con un errore
  leggibile invece di un OOM qualche chunk piu' avanti.
- **`Orca` ricorda riferimenti puliti passati e li richiama via `apply_fast_resonance`**: in
  modalita' cieca (`x_reference=None`), prima di ricadere sulla stima locale (`_blind_reference`),
  Orca cerca nella propria memoria (per shape, fino a `reference_memory_size=32` per default) un
  riferimento gia' visto simile all'input corrotto corrente (soglia `reference_recall_min_score=
  0.90`, calibrata empiricamente: segnali correlati restano sopra 0.95 anche con rumore
  moderato, segnali scorrelati non superano ~0.75). Banca vuota o nessun match abbastanza simile
  -> comportamento identico a prima.
- **`chunk_threshold` di `Orca` auto-tarato su `AIHardwareProfiler`**: default cambiato da un
  intero fisso (`1_000_000`, uguale per qualunque host) a `None`, che auto-deriva il valore
  scalando proporzionalmente a `max_tensor_dim` dell'host corrente (RAM/GPU-TPU). Su un host
  "base" (RAM<12GB, nessuna GPU/TPU) il risultato e' identico al vecchio default fisso -- nessuna
  regressione silenziosa sul caso comune. Un valore esplicito bypassa l'auto-tuning senza nemmeno
  istanziare il profiler.
- **`Orca(use_arbiter=True)`, opzione nuova, default `False`**: instrada ogni
  punto verso il correttore giusto invece di forzarne uno solo su tutta la
  riga -- nato dai tre esperimenti sopra (nessun gate unico batteva gli
  altri due ovunque). Nuovo modulo `utility/arbiter.py`: classifica ogni
  punto come 'clean'/'spike'/'regime' confrontandolo con una finestra di
  riferimento larga e CAUSALE (solo punti prima di i, mai dopo -- stessa
  scala della "molla" JSD di `pressure_valve`, non la finestra locale
  stretta che si e' vista collassare sopra), poi la lunghezza e la
  coerenza interna della sequenza di punti devianti decide impulso
  isolato vs cambio di regime sostenuto. 'spike' -> rigetto duro (mediana
  della finestra larga); 'regime' -> valore grezzo (fiducia, stessa
  filosofia del gate costante); 'clean' -> quello che lo scudo standard a
  4 fasi ha gia' prodotto (il suo smorzamento morbido, non un
  pass-through, altrimenti un segnale continuo non anomalo come
  un'oscillazione strutturata verrebbe lasciato passare intatto).
  Verificato su tutti i 7 scenari di `test/testKalman.py`
  (`test/test_arbiter_orca_integration.py`): mai peggiore del default
  (`use_arbiter=False`, garanzia testata), migliore su 5/7 (impulsi
  isolati, sub-soglia, Cauchy, rottura di livello, buco NaN), pari sul
  collasso a zero, identico sul segnale strutturato (ricade correttamente
  sullo scudo standard, com'e' giusto: e' smorzamento continuo, non
  un'anomalia discreta). Popola anche `Orca.etichette_arbitro`/
  `incertezza_arbitro` (l'incertezza della classificazione, non la
  dimensione della correzione -- segnale complementare a
  `margine_ingresso`, non ridondante) e
  `Orca.tipi_corruzione_visti(slice_shape)`, un Counter dei tipi di
  corruzione gia' osservati per quella shape quando `x_reference` era
  noto -- memoria separata dalla banca dei riferimenti puliti gia'
  esistente, che ricordava solo la FORMA, non il TIPO di corruzione.
  CRONOLOGIA di un limite trovato e risolto nella stessa sessione: la
  prima versione usava una finestra di riferimento SIMMETRICA, che a
  cavallo di una rottura di livello permanente mescolava vecchio e nuovo
  livello nella propria scala, non rilevando affatto la transizione
  (deviazione sempre 0) -- un primo confronto isolato aveva scambiato
  questo per un successo (RMSE 0.0), ma era un artefatto: i dati grezzi
  di quello scenario di test coincidono GIA' col target, quindi "passa
  tutto grezzo" da' RMSE 0 indipendentemente dall'etichetta. Attraverso
  Orca la mancata rilevazione si vedeva davvero: nessun miglioramento
  (9.4007 identico con o senza l'arbitro). Passando alla finestra
  causale, la rilevazione e' diventata reale (indici 60-74 etichettati
  'regime', esattamente la transizione vera a 60) e l'RMSE reale scende
  a 6.0449 -- vedi `utility/arbiter.py` per il dettaglio completo.
- **`dense_armor.mcp_server`, nuovo (extra `[mcp]`)**: server MCP con 5
  tool (`dense_armor_health`, `dense_armor_clean_signal`,
  `dense_armor_detect_anomalies`, `dense_armor_robust_filter`,
  `dense_armor_heal_series`) cosi' un agente puo' ripulire una serie
  senza scrivere Python. Diretto e in-process, non un proxy HTTP verso
  un kernel separato come l'adattatore di Dense-Evolution -- Dense-Armor
  non ha una web UI da condividere. Vive apposta sotto
  `dense_armor.mcp_server`, non un semplice `mcp_server`: con
  Dense-Evolution installato nello stesso ambiente (il suo adattatore si
  chiama esattamente `mcp_server`), un nome non annidato e' una
  collisione VERA, osservata direttamente (`from mcp_server.server
  import mcp` risolveva ai 25 tool di Dense-Evolution invece dei 5 di
  qui, a seconda dell'ordine di installazione), non un'ipotesi --
  scoperta e corretta prima che questo finisse pubblicato. Nuovo script
  console `dense-armor-mcp`, separato da `dense-armor` (la CLI di
  Armatura, invariata). Un bug reale trovato e corretto nello sviluppo:
  lo schema Pydantic di `values` dichiarava supporto NaN nel docstring
  ma rifiutava silenziosamente `null` JSON dentro la lista (`List[float]`
  invece di `List[Optional[float]]`). Verificato end-to-end via
  `mcp.call_tool()` reale (non solo le funzioni Python dirette) in
  `test/test_mcp_server.py`, 222/222 test verdi in totale.

## [1.1.9]

### Fixed
- **`core/compiler.py` ricompilava ad ogni chiamata**: `run_dynamic_pipeline`,
  `run_pipeline_with_chunking` e `compute_gradients` definivano il proprio kernel `@jax.jit`
  DENTRO il metodo -- un nuovo oggetto `jax.jit` (quindi una cache di compilazione vuota) ad
  ogni singola chiamata, mai riuso della compilazione XLA anche a parita' di shape tra una
  chiamata e la successiva. Stesso bug gia' trovato e risolto nel motore di
  `AdaptiveSignalStabilizer` (vedi [1.0.3]), mai corretto qui. Trovato scrivendo un test che
  doveva dimostrare l'utilita' reale di `PipelineProfiler` (warm-up separato dal regime): con
  il bug, warm-up e regime erano praticamente identici. Fix: i tre kernel spostati a livello di
  modulo, compilati una sola volta. Verificato: 375ms -> 0.2ms a regime sulla stessa pipeline
  (>1700x), test di regressione aggiunto.
- **`utility/anwav.py` andava in crash su console Windows non-UTF-8**: i verdetti su volume
  "troppo spinto"/"troppo silenzioso"/picco a rischio usavano emoji (`❌`, `⚠️`) nei `print()` --
  su cp1252 (l'encoding di default della console Windows) sollevavano `UnicodeEncodeError`,
  quindi qualunque uso reale della funzione fuori da pytest (che forza UTF-8 in
  `test/conftest.py`) su un file rumoroso o silenzioso andava in crash. Rimosse le emoji dai
  messaggi.
- **Link rotto in questo stesso CHANGELOG bloccava il deploy del sito in silenzio**: la voce
  1.1.8 linkava `[Toolkit](api/toolkit.md)`, un percorso relativo valido solo dentro `docs/` --
  ma `CHANGELOG.md` vive nella root del repo e viene incluso cosi' com'e' in
  `docs/changelog.md`. `mkdocs build --strict` rifiutava giustamente la build, e i due push
  precedenti non erano mai arrivati sul sito pubblico (la sezione toolkit di
  `getting-started.md` e gli esempi in `docs/api/toolkit.md` erano gia' su GitHub ma non ancora
  online). Corretto con l'URL completo del sito; verificato dal vivo che il sito pubblico ora
  li mostra davvero.

### Changed
- **`docs/api/toolkit.md` non vende piu' ogni modulo allo stesso modo**: aggiunte note oneste
  dove l'utilita' e' effettivamente sottile (`TensorVault` sono matrici scrivibili in una riga,
  l'euristica RAM di `AIHardwareProfiler` e' arbitraria, `BitwisePermutationEngine` fa un solo
  swap controllato per chiamata non una permutazione generale, l'update di
  `ParametricScenarioSimulator` e' una EMA fissa non un modello configurabile, i formatter di
  logging sono wrapper sottili, `StochasticAdversarialNoise` si sovrappone alla suite
  adversarial reale invece di aggiungerci qualcosa) e rafforzate quelle con prove reali
  (`PipelineProfiler` ha scoperto il bug sopra; i preset producono filtraggio misurabilmente
  diverso; `kappa` in `apply_fast_resonance` modula davvero il punteggio, non e' cosine
  similarity travestita).

### Added
- **Test che dimostrano l'utilita' dichiarata dei moduli, non solo che "non crashano"**: preset
  diversi (`SIGNAL_STABILIZER_PRESETS`) producono un filtraggio misurabilmente diverso quando
  collegati a `AdaptiveSignalStabilizer`; `kappa` in `apply_fast_resonance` modula davvero il
  punteggio; il warm-up di `PipelineProfiler` e' davvero piu' lento del regime (il test che ha
  scoperto il bug di ricompilazione sopra).
- **`docs/api/toolkit.md`**: ogni modulo del toolkit ha ora un esempio di codice reale
  verificato (non solo la firma auto-generata dai docstring), preceduto da cosa fa e a cosa
  serve.

## [1.1.8]

### Fixed
- **`docs/getting-started.md` non spiegava come usare il toolkit**: la pagina aveva un
  quickstart per Orca, Armatura e robust_filters, ma per il toolkit (14 moduli, aggiunti in
  1.1.6-1.1.7) solo un link nudo nella home del sito, nessun esempio eseguibile. Aggiunta una
  sezione "Standalone toolkit" con due esempi reali verificati (`DynamicAICodegen`,
  `UniversalMemoryGuard`) e link a [Toolkit](https://tatopenn-cell.github.io/Dense-Armor/api/toolkit/)
  per il resto.

## [1.1.7]

Nessuna modifica alla matematica/logica di `Armatura`/`Orca` in questa versione. I 14 moduli
del toolkit aggiunti a 1.1.6 sono ora davvero pubblici (documentati sul sito, `show_source:
true`), quindi tutto quello che c'era di poco professionale nel codice sorgente -- non solo nel
README -- ora si vede.

### Added
- **Sito**: nuova pagina `docs/api/toolkit.md` (mkdocstrings, pesca dai docstring reali) per i
  14 moduli aggiunti in 1.1.6 -- prima invisibili sul sito, l'API Reference copriva solo i 5
  moduli dello scudo. Linkata da `mkdocs.yml`, `docs/index.md`, `docs/api/index.md`.
- **README**: la sezione `$ toolkit --standalone` (prima solo un blocco di import con commenti)
  riscritta con prosa reale, raggruppata per categoria (pipeline/chunking, hardware/profiling,
  tensori/configurazione, logging/provenance, audio/I/O dati, ricerca per similarità).

### Changed
- **Rimosso "Sentinel"/branding residuo dal codice sorgente** dei 14 moduli toolkit e dei 5
  moduli originali dello scudo (`armatura.py`, `core/damping_operator.py`, `utility/metro.py`,
  `utility/orca.py` -- mancati nel passaggio di de-branding di 1.1.5, che si era fermato al
  testo rivolto all'esterno README/pyproject/sito, senza scendere nei singoli docstring con
  `show_source: true`): banner "SENTINEL ENTERPRISE... BEAST MODE" e claim falsi di "logica
  quantistica" in `core/chunk.py`, blocco "Fix applicati/BILANCIAMENTO AUREO" (contenuto da
  changelog finito nel docstring) in `core/compiler.py`, riferimenti a file interni inesistenti
  (`simulator.py`, `NoiseModel`, `test21.py`/`main.py`) in `core/vector.py`/`core/noise.py`/
  `core/chunk.py`, intestazioni "Sentinel Metrology Framework" e vecchio percorso interno
  `shield_/...` nei 4 moduli originali.
- **Simboli pubblici rinominati** (uniche API realmente rotte da questa release, erano appena
  diventate pubbliche in 1.1.6): `execute_pipeline_beast_mode` -> `execute_pipeline_chunked`
  (`core/chunk.py`); `SENTINEL_PRESETS` -> `SIGNAL_STABILIZER_PRESETS` (`core/preset.py`);
  `get_enterprise_logger` -> `get_json_file_logger`, nome logger/file di default
  `sentinel*`/`sentinel_dashboard.log` -> `dense_armor*`/`dense_armor.log`, campo JSON
  `"framework": "Sentinel-TensorFlowEngine"` -> `"dense-armor"` (`core/logger.py`);
  `AIEngineVisualizer.ENGINE_SIGNATURE`/intestazione del trend report non dicono più
  `TensorFlowEngine-Sentinel` (`core/visualizer.py`).
- **`utility/orca.py`**: rimossa una `sys.path.insert(...)` morta (residuo di quando il modulo
  girava come script standalone fuori dal pacchetto -- gli import relativi non ne hanno
  bisogno, e nient'altro nel file usava `sys`/`os` dopo quella riga) e i relativi import
  inutilizzati.
- **`armatura.py`**: docstring e messaggi di uso CLI aggiornati da `python armatura.py`/`from
  armatura import Armatura` (invocazione da script standalone) a `python -m dense_armor`/`from
  dense_armor import Armatura` (pacchetto installato).

## [1.1.6]

Nessuna modifica alla matematica/logica di `Armatura`/`Orca` in questa versione. Partita
dalla richiesta di portare la coverage Codecov al 100%: 14 file in `core/`/`utility/`
risultavano a 0% di coverage. Prima di cancellarli come "codice morto" (la prima ipotesi),
trovato `dense_armor/README.md` (riferimento interno) che li documenta tutti come componenti
reali e intenzionali — indagine invertita, modulo per modulo, invece di procedere con la
cancellazione.

### Fixed
- **`core/__init__.py` vuoto**: il vero file di inizializzazione del pacchetto (import/export
  di 9 classi: `AIHardwareProfiler`, `StochasticAdversarialNoise`, `UniversalMemoryGuard`,
  `MemoryPressureError`, `ParametricScenarioSimulator`, `BitwisePermutationEngine`,
  `DynamicAICodegen`, `CMD_MAP`, `TensorVault`, `AIEngineVisualizer`, `PipelineProfiler`)
  esisteva ma era salvato come `core/init.py` (nome sbagliato, mai eseguito da Python) — il
  file col nome giusto era vuoto. Nessuna di queste classi era importabile da
  `dense_armor.core.*` nonostante il codice reale esistesse. Contenuto spostato nel file col
  nome corretto, `init.py` eliminato.
- **`core/preset.py`**: l'intero dizionario `SENTINEL_PRESETS` (4 profili calibrati) era
  definito due volte verbatim nello stesso file. Rimossa la duplicazione.
- **`utility/iodat.py`**: `lodat()` ritornava in silenzio un tensore di dati casuali finti se
  il file richiesto non esisteva (solo un log di warning, nessun errore) — ora solleva
  `FileNotFoundError`.

### Added
- **Test reali per tutti e 14 i moduli** precedentemente a 0% di coverage: `chunk.py`,
  `compiler.py`, `memory.py`, `preset.py`, `logger.py`, `tensor.py`, `vector.py`, `noise.py`,
  `profiler.py`, `visualizer.py` (`core/`); `anwav.py`, `diagnostic.py`, `iodat.py`
  (`utility/`, con file WAV/HDF5/NetCDF reali generati nei test, non mock) — più le righe
  ancora scoperte in `resonance_search.py`. Coverage totale del pacchetto: 54% -> 91%.
- **README**: nuova sezione `$ toolkit --standalone` che documenta questi moduli come seconda
  parte del pacchetto, indipendente dallo scudo anomalie di Armatura/Orca.

## [1.1.5]

Nessuna modifica alla matematica/logica di `Armatura`/`Orca`/`robust_filters` in questa
versione -- solo infrastruttura di test, documentazione e metadati.

### Added
- **Suite di test adversarial reale** (`test/test_boundA-E.py`): i test dietro la tabella
  "robustezza adversarial" del README (PGD/BIM/MI-FGSM, affine/elastica, Carlini-Wagner L2/Linf,
  DeepFool, Fourier a banda estesa) esistevano solo in un progetto precursore ("sentinel3", mai
  parte di questo repository) da cui `AdaptiveSignalStabilizer` era stato estratto. Riportati
  qui con una modifica sostanziale: l'import punta ora a
  `dense_armor.core.engine.AdaptiveSignalStabilizer` (il pacchetto reale) invece del modulo
  locale standalone del progetto precursore -- stessa classe, stessa API, ri-eseguiti e
  confermati riprodurre esattamente i numeri gia' nel README (PGD 0.0130, BIM 0.0625, MI-FGSM
  0.0784, C&W L2 78.96%, C&W Linf 64.39%, DeepFool 78.79%, Fourier 99.77%+): la tabella e' ora
  una misura reale e riproducibile, non piu' un dato non tracciato.
- **Sito di documentazione** (GitHub Pages, <https://tatopenn-cell.github.io/Dense-Armor/>):
  home, guida rapida, riferimento API generato dai docstring reali dei 5 moduli pubblici
  (`Armatura`, `Orca`, `hybrid_engine`, `engine`, `robust_filters`), changelog/licenza
  sorgentati da `CHANGELOG.md`/`LICENSE.md` -- stesso impianto mkdocs-material gia' in uso su
  Dense-Evolution.

### Fixed
- **Due assert instabili in CI**: `test_boundB.py`/`test_boundD.py` avevano un
  `assert t_elapsed >= 5.0` che non verificava la correttezza della difesa, solo che il calcolo
  fosse durato "abbastanza" sulla macchina di sviluppo originale -- falliva su ogni runner CI
  piu' veloce (verificato: 4.41s invece dei 5.0s richiesti su GitHub Actions, riprodotto su
  ubuntu/windows-latest, ogni versione Python). Rimosso: le assert reali (nessuna divergenza
  numerica) sono l'unica verifica che conta.

### Changed
- **Rimosso "Sentinel"** (nome interno originale del progetto) dal testo rivolto
  all'esterno: descrizione PyPI (`pyproject.toml`), README, home della documentazione,
  docstring di `__init__.py`, e l'etichetta nella definizione di Software in `LICENSE.md`
  (solo l'etichetta, nessuna modifica ai componenti elencati/coperti dalla licenza). Il
  `CHANGELOG.md` (storia reale del progetto) e i commenti interni nel codice restano
  invariati.

## [1.1.4]

### Added
- **`pressure_valve`**: orchestratore automatico dei quattro rilevatori
  aggiunti in 1.1.3. Non un voto (quanti metodi segnalano un punto) ma la
  combinazione classica a minima varianza (stimatore BLUE): ogni metodo
  produce una coppia (centro, scala) locale, i pesi della combinazione
  sono derivati con un moltiplicatore di Lagrange minimizzando la varianza
  della combinazione pesata sotto il vincolo che i pesi sommino a 1 --
  w_k proporzionale a 1/scala_k^2, un metodo la cui incertezza si gonfia
  (es. Chauvenet quando la finestra contiene gia' un outlier) viene
  automaticamente pesato meno, senza scartarlo esplicitamente. Decisione
  finale sempre binaria (marcato/sostituito con la mediana locale, o
  intatto). Soglia di default (`soglia_pressione=8.0`) calibrata
  empiricamente -- 1/300 falsi positivi su rumore gaussiano puro,
  rilevamento corretto di un outlier chiaro e di due outlier ravvicinati.
- **Soglia dinamica via JSD ("la molla")**: `soglia_pressione` non e' piu'
  fissa -- si confronta la finestra locale con una piu' ampia (Jensen-
  Shannon divergence) e si allarga proporzionalmente quando le due
  distribuzioni divergono (una vera transizione di regime in corso), senza
  mai smettere di decidere in binario. La soglia puo' solo allargarsi
  (JSD >= 0), mai restringersi sotto il valore base.
  - **Bug reale trovato e corretto durante la calibrazione**: la JSD
    "ingenua" (bin fissi, epsilon quasi-zero) risultava PIU' alta su
    rumore stazionario puro che vicino a una vera transizione -- il
    contrario esatto dello scopo del meccanismo, un artefatto da pochi
    campioni per bin. Corretto con binning adattivo (~6 campioni/bin) e
    smoothing di Laplace vero (pseudo-conteggio, non solo un epsilon per
    evitare log(0)) -- riverificato: JSD media vicino a una vera
    transizione 1.0->5.0 ora 2-3x piu' alta che su rumore stazionario alla
    stessa ampiezza, la direzione corretta.

## [1.1.3]

### Added
- **`dense_armor/utility/robust_filters.py`**: quattro rilevatori di anomalie
  "a basso costo" -- Chauvenet's criterion (1863), Tukey's fences/IQR, Hampel
  filter, sigma-clipping iterativo. Statistiche classiche, note e usate da
  decenni in telemetria/astronomia, nessun modello dinamico/stato -- solo
  aritmetica su una finestra locale centrata (pensate per pulizia offline/
  batch, non il ciclo causale real-time di `core/hybrid_engine.py`; adatte
  anche a un futuro porting embedded/Arduino, dove non ci si potra' appoggiare
  a dipendenze come numpy).
- **`pressure_valve`**: orchestratore automatico dei quattro metodi sopra.
  Non un voto (quanti metodi segnalano un punto) ma la combinazione classica
  a minima varianza (stimatore BLUE): ogni metodo produce una coppia
  (centro, scala) locale, e i pesi della combinazione sono derivati con un
  moltiplicatore di Lagrange minimizzando la varianza della combinazione
  pesata sotto il vincolo che i pesi sommino a 1 -- il risultato,
  w_k proporzionale a 1/scala_k^2, pesa automaticamente meno un metodo la
  cui incertezza si gonfia (es. Chauvenet quando la finestra contiene gia'
  un outlier), senza doverlo scartare esplicitamente. Decisione finale
  sempre binaria (marcato/sostituito con la mediana locale, o intatto) --
  nessuno stato intermedio, coerente col resto dell'ecosistema. Soglia di
  default (`soglia_pressione=8.0`) calibrata empiricamente (non "3 sigma":
  la scala combinata e' sempre piu' stretta di ogni scala singola) --
  verificato 1/300 falsi positivi su rumore gaussiano puro, rilevamento
  corretto di un outlier chiaro e di due outlier ravvicinati (dove
  Chauvenet da solo soffre), nessun tocco su un gradino genuino sostenuto.

## [1.1.2]

### Fixed
- **`hybrid_shield` restava bloccato su un equilibrio sbagliato dopo un
  cambiamento di livello genuino e sostenuto** (`core/hybrid_engine.py`):
  trovato testando `Armatura(livello_ia=0.0)` dal vivo su uno scenario base
  (salto da 1.0 a 5.0, poi 30 punti stabili al nuovo livello). La finestra
  di baseline locale era presa da `out` (già guarito) invece che da
  `processed` (grezzo) — una scelta deliberata per proteggere da uno spike
  isolato passato che sposta la baseline, ma con un effetto collaterale
  peggiore su un gradino vero: i primi 1-2 punti dopo la transizione
  vengono inevitabilmente respinti (nessun rilevatore causale distingue un
  gradino vero da uno spike al primo campione), quei punti respinti
  finiscono in `out`, e la finestra dei punti successivi li rimedia dentro
  il proprio calcolo — producendo una baseline "a metà strada" che respinge
  ANCHE i punti successivi genuini, in un loop che si autoalimenta. Il
  segnale restava bloccato per sempre su un valore intermedio sbagliato
  (~4.0 invece di 5.0), non solo per una manciata di passi transitori.
  Fix: la finestra torna a usare `processed` (grezzo), riallineata a
  `ia_utils.vector_healing.enhanced_dense_healing_hybrid` (il riferimento
  che questo modulo dichiara di generalizzare) — il valore di fallback è
  ora la MEDIANA della finestra grezza, non la media: la mediana ignora già
  da sola un singolo spike isolato nella finestra, quindi la protezione
  originale resta valida senza bisogno del trucco `out` (riverificato
  direttamente). Il test di regressione esistente
  (`test_gradino_sostenuto_non_viene_appiattito_ma_lo_spike_isolato_si`)
  non lo intercettava (tolleranza/finestra di controllo troppo larghe,
  passava anche col bug) — aggiunto un nuovo test più severo che fallisce
  sul codice vecchio e passa su quello corretto.

## [1.1.1]

### Fixed
- **NaN sanato con la media globale della serie invece di una stima locale**
  (`core/hybrid_engine.py`, `armatura.py`): trovato testando l'esempio esatto
  del README con un'installazione pulita da PyPI subito dopo la 1.1.0. Un
  `NaN` veniva sostituito con `np.nanmean` calcolata su TUTTA la serie —
  se uno spike enorme era presente ovunque nella serie (anche lontanissimo),
  contaminava il sostituto di ogni NaN, non solo di quelli vicini allo
  spike. Su serie brevi il sostituto risultava assurdo (2000.81 invece di
  ~1.28 sull'esempio del README, n=6); su serie più lunghe l'effetto si
  attenuava ma non spariva (51.24 invece di ~1.0 su n=200) — non era un
  caso limite di serie corte, era la logica stessa. Se il NaN cadeva anche
  entro il raggio locale di uno spike (fino a 20 passi), la volatilità
  locale gonfiata dallo spike impediva pure al trigger di ricorreggerlo
  verso la baseline. Fix: nuova funzione `_local_nan_fill` — ogni NaN è
  sostituito con la mediana dei vicini FINITI in una finestra locale
  (mediana, non media: robusta a uno spike che capiti nella stessa
  finestra), non con una statistica sull'intera serie. La marcatura come
  anomalia (`anomalie`) era già sempre corretta anche col bug — solo il
  valore restituito in `pulito` per quel punto era sbagliato.

## [1.1.0]

### Changed
- **Motore di `Armatura` sostituito**: `Armatura.analizza()` non usa più
  `AdaptiveSignalStabilizer` (Stadio 1) + `ABCollatz` (Stadio 2). Il gate
  ABCollatz era già documentato in [1.0.10] come matematicamente non
  discriminante (il radicale di una traiettoria di Collatz non ha relazione
  monotona con l'intensità del rumore — verificato con sweep numerico, e una
  sigmoide monotona alternativa, pur costruita e testata, peggiorava
  sistematicamente su 7 scenari reali). Sostituito con un motore a trigger
  binario (`dense_armor/core/hybrid_engine.py`), portato e adattato dalla
  logica phi_ab/vettore-dinamico/trigger già verificata in Dense-Evolution
  (`dense_evolution/healing.py`, oggi in produzione su PyPI dense-evolution
  >= 8.1.9) e in `ia_utils.vector_healing.enhanced_dense_healing_hybrid`.
  Tre adattamenti necessari per il nuovo dominio (serie scalari a scala
  libera, non vettori di embedding normalizzati), tutti verificati con test
  manuali mirati prima di essere accettati: (1) la baseline della finestra
  locale usa i valori già guariti (`out`), non quelli grezzi, altrimenti un
  outlier passato continua a spostare la baseline per `radius` passi dopo
  di sé; (2) l'IPG (gradiente istantaneo) resta sui valori grezzi, non su
  quelli guariti, altrimenti un punto respinto non lascia mai più traccia
  e nessun gradino reale può essere riconosciuto in seguito; (3) la
  distanza tra stati è normalizzata sulla volatilità locale del segnale
  (deviazione standard delle differenze successive nella finestra), non su
  una costante fissa né sulla grandezza assoluta dei valori — altrimenti
  o qualunque cambiamento sopra ~1.4 unità restava permanentemente
  appiattito, o (nel tentativo intermedio) anche uno spike isolato passava
  intatto perché "proporzionalmente coerente" con la propria grandezza.
  Contratto pubblico di `Armatura` invariato (`analizza`, `referto`,
  `referto_json`, `deriva`, stessa firma del costruttore); `K` è ora
  binario (0.0/1.0) invece di una sigmoide continua, coerente con la
  natura binaria del trigger sottostante.
- **`Orca` non è stata toccata in questa release**: continua a usare
  `AdaptiveSignalStabilizer` + `ABCollatz` + `apply_damping_blend`. I
  numeri di robustezza adversarial nel README (PGD/Carlini-Wagner/Fourier)
  restano validi e non ricalcolati, perché misurati su quel motore.

## [1.0.11]

### Fixed
- **`K_anomalous` in `apply_damping_blend` (`dense_armor/core/damping_operator.py`)**:
  la formula `c_anom/(c_anom+diff)` dava K_anomalous massimo a differenza
  quasi nulla (segnale già pulito, dovrebbe damparsi poco) e lo faceva
  decrescere fino al pavimento `_ALPHA=0.25` al crescere della differenza —
  l'opposto di quanto dichiarato nella docstring della funzione. Invertito
  il rapporto (`diff/(c_anom+diff)`). Testato su 30 seed × 4 livelli di
  rumore: il fix batte l'originale 30/30 a rumore medio/alto/molto alto
  (RMSE fino a -0.137), ma peggiora sistematicamente a rumore molto basso
  (limite noto, documentato e coperto da test di regressione dedicato).

### Added
- **`dense_armor/utility/healing.py` — `healing_filter`**: nuovo modulo,
  a sé stante (non integrato in `armatura.py`/`orca.py`), porting
  concettuale da `Dense-Evolution/dense_evolution/healing.py`. A differenza
  di ABCollatz e del damping Stadio 1 (giudicano un punto dal proprio
  residuo istantaneo), classifica ogni punto guardando quanti vicini
  condividono la stessa deviazione dalla baseline locale — deviazione
  isolata = rumore (sostituita con la mediana di una finestra più ampia),
  deviazione sostenuta dai vicini = cambiamento vero (lasciata passare).
  Batte una mediana mobile a raggio 2 su segnali con transizioni vere +
  spike (40/40 e 30/30 su più varianti testate); perde su denoising puro
  di segnali stazionari a basso/medio rumore (limite noto, non è il suo
  caso d'uso primario).

## [1.0.10]

### Fixed
- **Gate ABCollatz dello Stadio 2 (`compute_damping_gating`)**: era
  matematicamente saturo a ~0.85 sempre, indipendentemente dal rumore
  vero — causa: il radicale di un intero (derivato dalla traiettoria di
  Collatz) non ha nessuna relazione monotona con la sua grandezza
  (verificato con sweep numerico: rumore crescente 0->1000 dava
  discrepanze 0, 300, 220, 600, -8, 13920, 0, senza andamento). Non era
  un problema di scala/compressione: l'artefatto restava identico su
  dati grezzi.
- Costruita e testata anche una sigmoide monotona e scala-invariante sul
  rumore relativo, genuinamente discriminante — ma su 7 scenari reali
  (`test/testKalman.py`), sia in modalità cieca sia con riferimento vero
  esplicito, peggiora sistematicamente l'RMSE rispetto al fallback
  costante 0.85 (anche alzando il pavimento minimo fino a 0.84). Nel
  design attuale di Orca il riferimento (Stadio 1) è già una stima
  affidabile: correggere sempre con forza verso di esso batte qualunque
  discriminazione basata sul rumore locale. Ripristinato il fallback
  costante, ora dichiarato esplicitamente invece di emergere per
  accidente da una formula rotta.

Nessun cambio di comportamento a runtime per chi già usa il pacchetto
(RMSE e tempi verificati pressoché identici prima/dopo su tutti gli
scenari di test) — il fix è di correttezza/manutenibilità del codice, non
di funzionalità.

## [1.0.9]

Indagine approfondita partita dalla verifica di uno script esterno che
testava (male, con una reimplementazione a mano non fedele) il fix 1.0.8 —
ha portato a scoprire e risolvere 4 bug distinti in cascata nello scudo
entrata, ciascuno mascherato dal precedente.

### Fixed
- **Compressione log10 (`Orca._execute_4_phase_input_shield`)**: due fattori
  di scala indipendenti (uno da pulito, uno da corrotto) collassavano
  qualunque valore alla stessa magnitudine compressa, distruggendo ogni
  differenza relativa prima ancora che Stadio 1/Stadio 2 la vedessero
  (verificato: anche un pass-through totale senza nessuna vera protezione
  ricostruiva il pulito esatto, allo stesso modo). Ora un solo fattore
  condiviso, derivato dal pulito, applicato a entrambi.
- **Stadio 2 ingannato dal segnale già ammortizzato**: `compute_damping_gating`
  valutava `f1` (output dello Stadio 1) invece del segnale originale,
  sotto-stimando l'anomalia se già parzialmente corretta a monte.
- **Contaminazione post-shock dello Stadio 1**: il motore ricorsivo
  (`AdaptiveSignalStabilizer`) lasciava sempre passare almeno il 25%
  (`k_anom_min`) di un'anomalia enorme nel proprio stato interno, che poi
  decadeva lentamente contaminando per diversi passi anche campioni
  successivi perfettamente normali.
- **Costante `c_anom` fissa non in scala con i dati compressi**: era
  comparabile in grandezza al rumore in spazio compresso, impedendo alla
  soppressione naturale delle anomalie di funzionare anche con lo State
  Flush attivo.

### Added
- **Hard-clamp deterministico**: `raw_noise` (dati grezzi originali, prima
  di qualunque compressione) > 0.05 forza il gate finale a 0.99 (non più
  0.85, per non ereditare il pavimento pensato per i disturbi ordinari).
- **State Flush**: lo stesso segnale hard-clamp, già autorevole, passato
  anche allo Stadio 1 (nuovo parametro opzionale `hard_clamp_mask` su
  `filter_batch_scenarios`/`_process_single_scenario`/`_step_kernel`) —
  azzera il pavimento di guadagno minimo solo per il passo flaggato.
- **`c_anom` scala-adattiva**: proporzionale alla magnitudine locale
  corrente (`prev_filtered`, già nello stato ricorsivo) invece di una
  costante fissa assoluta — si adatta da sola sia a dati grezzi
  (`Armatura`, `filter_data_stream` diretto) sia a dati compressi (`Orca`),
  senza imporre l'assunzione di scala di un solo chiamante nella classe
  generica `AdaptiveSignalStabilizer`.

Verificato con numeri reali: un outlier da 9999 in una serie di valori
~1.3 è ora protetto a 3.61 (indistinguibile dagli altri campioni ~3.4-3.62
dopo il modello), tutti i vicini tornano normali, margine d'errore corretto.
Tutti i parametri nuovi sono opzionali con default che preservano il
comportamento esistente per chi non li usa.

## [1.0.8]

### Fixed
- `Orca._execute_4_phase_input_shield`: lo Stadio 2 (gating ABC/Collatz)
  valutava `f1`, l'output già ammortizzato dallo Stadio 1
  (`AdaptiveSignalStabilizer`), invece del segnale originale — se lo
  Stadio 1 riduceva parzialmente un outlier enorme, lo Stadio 2 poteva
  sotto-stimare quanto fosse anomalo l'input reale. Ora valuta il segnale
  compresso pre-Damping.

### Added
- Sbarramento deterministico (hard-clamping): la soglia di rumore critico
  (0.05) è calcolata sui dati **grezzi originali**, prima di qualunque
  compressione log10 (che rinormalizzerebbe ogni valore individualmente,
  facendo perdere l'intensità reale del rumore). Se superata, il gate
  finale viene forzato alla blindatura massima (0.85), bypassando il
  calcolo ABC/Collatz solo per le macro-anomalie; sotto soglia la
  pipeline sinergica originale resta invariata. Verificato: il leak
  residuo su una macro-anomalia crolla a zero. `@jax.jit`/`jax.vmap`
  intatti, nessuna nuova chiamata eager fuori dal kernel precompilato.

## [1.0.7]

### Fixed
- `evaluate_abc_discrepancy` arrotondava sempre `b` prima di calcolare il
  radicale, anche quando chiamata da `compute_damping_gating_smooth` — che
  genera apposta valori `collatz_wave` non arrotondati tramite
  `execute_collatz_step_smooth`. La variante smooth collassava così
  silenziosamente sulla stessa matematica di quella discreta, un livello
  più in basso di dove il docstring già avvertiva del rischio. Aggiunto
  `smooth_mode: bool = False` (gestito con `jnp.where`, compatibile con
  `jax.jit`/`vmap`); `compute_damping_gating_smooth` ora passa
  `smooth_mode=True`. La variante discreta di default non è cambiata.
  Verificato: prima del fix le due varianti davano risultati identici
  anche su input non interi; ora divergono, come previsto.

## [1.0.6]

Risolti i 3 punti lasciati aperti come Known Issues in 1.0.5.

### Changed
- **Logging** (comportamento cambiato, non solo interno): le ~14 chiamate
  `print()` che stampavano lo stato di `Orca.protect_and_forward` ad ogni
  inferenza (`[ORCA] Attivazione...`, `orca/utility/iodat.py`,
  `core/compiler.py` save/load pipeline) sono ora `logging.getLogger(__name__)`.
  **Di default sono silenziose** (nessun handler configurato dalla libreria,
  convenzione standard) — per vederle di nuovo: `logging.basicConfig(level=logging.INFO)`
  prima di usare la libreria. Le ~35 `print()` rimaste (in `armatura.py`
  metodo `referto()` e CLI `main()`, `utility/anwav.py`, `utility/diagnostic.py`)
  **non sono state toccate**: sono referti/output voluti quando l'utente
  chiama esplicitamente quelle funzioni, non rumore di background —
  convertirle avrebbe reso silenzioso per default uno strumento il cui
  scopo e' stampare un referto.

### Fixed
- **Test JIT flaky sotto pressione di RAM**: `Orca._gc_se_ram_bassa()`
  chiamava `jax.clear_caches()` (svuota la cache di compilazione JIT di
  TUTTO il processo) alla stessa soglia morbida di `gc.collect()` — se la
  RAM libera scendeva anche solo temporaneamente sotto quel margine
  preventivo, la precompilazione fatta in `__init__` veniva vanificata e
  ogni chiamata successiva ricompilava XLA da zero. Ora `jax.clear_caches()`
  scatta solo al limite duro (`min_free_ram`), non al margine preventivo.
  Il test di regressione (`test_orca_protect_and_forward_usa_la_cache_jit_
  non_ricompila_ogni_volta`) ora finge anche RAM abbondante via mock,
  rendendolo indipendente dallo stato reale della macchina/CI.
- **Copertura docstring/type hint**: dal 53.9%/58.8% (55/60 su 102
  funzioni) al 100% — tutte le funzioni/metodi pubblici e privati hanno
  ora una docstring one-line e annotazioni di tipo su argomenti/ritorno.
- **`except Exception:` generico**: ristretto a eccezioni specifiche dove
  identificabili — `core/memory.py` (`subprocess`/parsing di `nvidia-smi`
  a `(CalledProcessError, TimeoutExpired, FileNotFoundError, OSError,
  ValueError)`; query VRAM `jax.devices()` a `(RuntimeError, AttributeError)`),
  `core/noise.py` (`jax.default_backend()` a `RuntimeError`). Lasciati
  broad-by-design, ma documentati con un commento: `memory.py` (pulizia
  cache JIT best-effort, ora anche loggata a livello debug invece di
  `pass` silenzioso) e `utility/resonance_search.py::smoke_test` (per
  definizione uno smoke test deve catturare qualunque fallimento).

## [1.0.5]

### Fixed
- `pyproject.toml`: campo `license` come tabella TOML deprecata da
  setuptools (avviso ad ogni build, sarebbe diventato errore bloccante dal
  18 febbraio 2027) sostituito con stringa SPDX standard `"BUSL-1.1"` +
  `license-files = ["LICENSE.md"]`. Richiede `setuptools>=77` (bumpato in
  `build-system.requires`).
- Residuo del vecchio nome del progetto ("sentinel02") nei commenti di
  intestazione di `dense_armor/core/damping_operator.py` e
  `dense_armor/utility/metro.py` — aggiornati al percorso corrente
  (`shield_/...`), nessun impatto funzionale.

### Known Issues (non risolti in questa versione)
- **Logging**: 49 chiamate a `print()` sparse nel codice per i messaggi
  `[ORCA] ...`. Esiste gia' `dense_armor/core/logger.py` ma non e'
  importato da nessun altro modulo — chi integra la libreria non puo'
  abbassare la verbosita', reindirizzare su file o silenziare i log senza
  modificare il sorgente.
- **Copertura docstring/type hint**: su 102 funzioni/metodi totali, 55
  hanno una docstring (53.9%) e 60 hanno almeno un type hint tra
  argomenti/valore di ritorno (58.8%).
- **`except Exception:` generico**: 5 punti che catturano tutto
  silenziosamente — `core/memory.py` (righe 56, 72, 96), `core/noise.py`
  (riga 43), `utility/resonance_search.py` (riga 73). Da rivedere caso per
  caso: quali eccezioni sono davvero previste (es. file mancante) e quali
  nascondono bug che dovrebbero propagarsi.

## [1.0.4]

### Changed
- Nessuna modifica al codice rispetto a 1.0.3 — solo bump di versione,
  perche' 1.0.3 risultava gia' occupata su PyPI da un caricamento
  precedente.

## [1.0.3]

### Fixed
- `filter_data_stream` (motore di `AdaptiveSignalStabilizer`) ricompilava
  l'intero kernel JIT ad ogni chiamata invece di riusare quello
  precompilato — kernel 1D spostato in `__init__`, soglie/gain passati
  come argomenti jit invece che letti da `self.*` dentro la funzione.
- `jnp.insert` chiamato fuori da `jax.jit` subito dopo lo scan (stesso
  problema, punto diverso) — sostituito con `np.insert` post-conversione.
- `Orca._execute_4_phase_input_shield` / `_execute_4_phase_output_shield`
  eseguivano la loro pipeline elementwise (`jnp.where`/`jnp.isnan`/
  aritmetica) in modalita' eager, fuori da qualunque `jax.jit` — estratta
  in due kernel dedicati precompilati una sola volta in `__init__`.
  Eliminate anche chiamate ridondanti duplicate a
  `filter_batch_scenarios`/`compute_damping_gating`.
- Risultato: 3 chiamate a `protect_and_forward` passano da ~11s a ~0.03s
  (zero ricompilazioni XLA residue a regime, verificato con `cProfile`).
- CI: workflow lanciava ancora `pytest tests/` dopo la rinomina della
  cartella in `test/` — corretto in `.github/workflows/tests.yml` e
  `README.md`.

### Changed
- Cartella dei test rinominata da `tests/` a `test/`.

## [1.0.1]

### Fixed
- **Bug critico**: `evaluate_abc_discrepancy` (radicale ABCollatz)
  calcolava il radicale su `a*b*c` invece che sul valore generato dalla
  traiettoria di Collatz — la traiettoria era di fatto ignorata,
  producendo lo stesso risultato indipendentemente dal valore Collatz
  reale (verificato: b=7 e b=999999 davano discrepanze identiche prima
  del fix).
- Dipendenza core mancante `psutil` aggiunta a `pyproject.toml` (trovata
  testando l'installazione del wheel in un venv pulito).

### Added
- Suite di test pytest (`test/`) e CI GitHub Actions
  (`.github/workflows/tests.yml`), matrice Python 3.10/3.11/3.12 su
  ubuntu/windows + controllo build/metadata.
- Istruzioni per lanciare la suite di test in locale nel README.

## [1.0.0]

Prima pubblicazione.
