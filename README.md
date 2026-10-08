<p align="center">
  <img src="docs/assets/banner.jpg" alt="Dense Armor -- shield for AI and robot I/O" width="900">
</p>

<p align="center">
  <img alt="tests" src="https://github.com/tatopenn-cell/Dense-Armor/actions/workflows/tests.yml/badge.svg">
  <a href="https://codecov.io/gh/tatopenn-cell/Dense-Armor"><img alt="codecov" src="https://codecov.io/gh/tatopenn-cell/Dense-Armor/branch/master/graph/badge.svg"></a>
  <a href="https://pypi.org/project/dense-armor"><img alt="pypi" src="https://img.shields.io/pypi/v/dense-armor.svg"></a>
  <a href="https://doi.org/10.5281/zenodo.23247521"><img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.23247521.svg"></a>
  <img alt="license" src="https://img.shields.io/badge/license-BSL_1.1-blue.svg">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
  <img alt="backend" src="https://img.shields.io/badge/backend-JAX-orange.svg">
  <img alt="training" src="https://img.shields.io/badge/training%20required-no-brightgreen.svg">
  <img alt="nan" src="https://img.shields.io/badge/NaN--safe-yes-brightgreen.svg">
  <img alt="detectors" src="https://img.shields.io/badge/anomaly%20detectors-4%2B1-blueviolet.svg">
  <img alt="combination" src="https://img.shields.io/badge/combination-Lagrange%20(BLUE)-9cf.svg">
  <a href="https://tatopenn-cell.github.io/Dense-Armor/"><img alt="docs" src="https://img.shields.io/badge/docs-tatopenn--cell.github.io-00e5ff?style=flat-square"></a>
</p>

<p align="center"><strong>Online learning, runtime shielding and robot control in one JAX library: estimators that learn one sample at a time, with uncertainty, drift and anomaly detection, adaptive damping for AI signals and verified safety for robotic commands.</strong></p>

<p align="center">📖 <a href="https://tatopenn-cell.github.io/Dense-Armor/"><strong>Full documentation, API reference, quick guide →</strong></a></p>

---

## `$ what it does`

A sensor that sends lost readings (`NaN`) or spits out an absurd value (`1e6` instead of `1.2`) silently breaks any downstream pipeline. Dense-Armor steps in between the raw data and the model that consumes it:

```
  corrupted data ──► [ INPUT SHIELD ] ──► AI model ──► [ OUTPUT SHIELD ] ──► clean output
                     purifies vs             │             purifies vs
                     reference               │             response-to-reference
                     (or robust blind        │             (or self-consistency)
                     estimate)
```

- **Input**: cleans the corrupted data toward a clean reference, if you have one — or toward a robust estimate derived from the data itself, if you do not.
- **Output**: checks that the model's response is not itself corrupted, comparing it with the response the model would give to the clean reference.
- **Error margin**: for each corrected value, returns how much it was shifted to clean it. Small correction → trust it. Large correction → treat with caution.

It leaves the weights untouched. It runs at runtime. It works from 1D up to 11D, tested (see `test/test_orca2.py`).

**Same discipline, a second domain**: from `$ rate_limiter` onward the package also covers command and trajectory safety for real robots — velocity/acceleration limits, spatial control barrier functions, minimum-jerk trajectory generation, rigid dynamics from real URDF (including `.xacro`, including `<mimic>` joints), passivity+CBF controllers up to 6-DoF. JAX backend shared with `Armatura`/`Orca`, same verification discipline on real data/robots — a different application domain, not a different package.

**A third domain, online learning**: estimators that learn one sample at a time on a robot or next to an LLM, all on the same foundation (`dense_armor.roles`): `learn_one` / `predict_one` / `score_one`, timestamped `Signal` samples with units, a real-time contract (p99 latency within the control period), conformal uncertainty, `SafeEstimator` with health and checksummed checkpoints, unit checks and URDF joint limits. On top of it: online statistics and sketches, metrics with predict-then-learn evaluation and event metrics for robots (detection delay, false alarms per hour), and four drift detectors (CUSUM, Page-Hinkley, ADWIN, KSWIN). LLM side: embeddings as features and estimators exposed as JSON tools for agents. This layer is being built for robots from the ground up: it takes the best-established online-learning techniques from the original papers (streaming statistics, predict-then-learn evaluation, sequential change detection, conformal prediction) and extends them with time, units, real-time budgets, uncertainty and safety. Native vision has started: frame streams, frame features and online reduction.

---

## `$ install`

```bash
pip install dense-armor                 # core: numpy + jax
pip install "dense-armor[quantum]"      # + Dense-Evolution (NISQ simulator)
pip install "dense-armor[audio,data]"   # + WAV, HDF5, NetCDF
```

```python
# mandatory before every import — required by 64-bit gating
import jax
jax.config.update("jax_enable_x64", True)
```

```powershell
# equivalent from PowerShell, before launch
$env:JAX_ENABLE_X64="True"
```

To run the test suite locally (by cloning the repo; not needed if you only installed from pip):

```bash
pip install -e ".[dev]"
pytest test/ -v
```

---

## `$ quickstart`

```powershell
python -m dense_armor --json 1.2 1.3 9999 1.25 nan 1.3
```

```
> anomaly @ index 2 (peak 9999)
> anomaly @ index 4 (NaN)
> everything else: intact
```

Connected to a real model:

```python
from dense_armor.utility.protect.orca import Orca

orca = Orca()
protected_output = orca.protect_and_forward(
    my_model,                                 # callable: x -> output (JAX/NumPy)
    corrupted_data,                           # tensor from the sensor/pipeline
    x_reference=clean_reference_data,         # optional but recommended
)

orca.margine_ingresso_medio, orca.margine_uscita_medio   # how much to trust
```

1D series (training loss, metrics, token stream):

```python
from dense_armor import Armatura

a = Armatura(livello_ia=0.0)   # 0 = actively filters · 1 = only flags
clean, K, anomalies = a.analizza(series)
```

---

## `$ internals`

Two distinct engines, depending on where you enter:

```
Armatura (1D series: loss, metrics, sensors)   Orca (input+output shield for an entire AI model)
─────────────────────────────────────────      ─────────────────────────────────────────────────────
core/hybrid_engine.py                           STAGE 1  core/engine.py
binary-trigger engine (phi_ab/dynamic           adaptive stabilizer, dynamic threshold on
vector), verified in Dense-Evolution and        recent volatility
adapted to scalar signals at free scale         STAGE 2  utility/collatz.py
                                                  Collatz-based gating, decides HOW MUCH
                                                  to damp toward the clean reference
```

`Armatura.analizza()` decides point by point, with no intermediate degrees: a value is either a genuine change (it passes) or isolated noise/spike (replaced with the local baseline — the recent window if you have no reference, your explicit reference if you pass one).

`Orca.protect_and_forward()` (complete shield for a model) still uses the two original stages. Without a clean reference (blind mode), before falling back to the local estimate Orca searches its own **memory of past clean references** (by shape, via `apply_fast_resonance` — the same resonance described later in the toolkit) for a reference similar to the current corrupted input; if it does not find one, rejection of severe outliers via local median, then Stage 1 in causal version — it uses the entire history of the series, not only immediate neighbors, to estimate what that point "should" be. Every time a clean reference is passed explicitly, it stays in memory for subsequent blind calls (`reference_memory_size`, default 32 per shape).

Before every heavy chunk, Orca also checks available RAM/VRAM via `UniversalMemoryGuard` (see toolkit below): below the critical threshold it raises `MemoryPressureError` instead of risking a silent OOM a few chunks later. The chunk size itself (`chunk_threshold`) is auto-derived from `AIHardwareProfiler` — it scales with the host's RAM/backend instead of a fixed value equal for every machine — unless you pass an explicit one to the constructor.

Technical note: `AdaptiveSignalStabilizer.filter_batch_scenarios` (used e.g. by the adversarial test suite) accepts only 2D/3D/4D. `Orca`'s input shield is a different path, without that limit.

---

## `$ orca --use_arbiter`

`Orca.protect_and_forward(..., use_arbiter=True)` — optional, default `False` — routes each point to the right corrector instead of forcing a single one over the whole row:

```python
orca = Orca()
protected_output = orca.protect_and_forward(my_model, corrupted_data, use_arbiter=True)

orca.etichette_arbitro       # array 'clean'/'spike'/'regime', one per point
orca.incertezza_arbitro_media   # 0..1: how ambiguous the classification itself is
orca.tipi_corruzione_visti(corrupted_data.shape)   # Counter, populated when x_reference is known
```

`spike` (isolated impulse) → hard rejection toward the median of a wide reference window; `regime` (sustained level change, recognized by looking at length and internal consistency of the sequence of anomalous points) → raw value, full trust; `clean` → whatever the standard 4-phase shield has already produced, not the raw value — a continuous non-anomalous signal still needs the soft damping of `AdaptiveSignalStabilizer`.

Verified on the 7 scenarios of `test/testKalman.py` (`test/test_arbiter_orca_integration.py`): **never worse than the default, better on 5/7** (isolated impulses, below-threshold corruption, heavy-tailed noise, level break, NaN data gap), equal on one case, identical on the remaining one by correct design (continuous structured signal → falls back to the standard shield). Details, chronology of a real bug found and fixed (symmetric → causal reference window) in the [changelog](https://tatopenn-cell.github.io/Dense-Armor/changelog/) and in the docstring of `utility/arbiter.py`.

---

## `$ streaming --realtime`

A real robot runs at 30-100Hz and cannot wait for an already-recorded array. `StreamingDeviationDetector` (`dense_armor.utility.anomaly.streaming`) brings to zero latency only the causal half of `classify_segments` — the per-point deviation flag, not the final spike/regime label, which requires looking ahead in the sequence and remains a batch question by design:

```python
from dense_armor.utility.anomaly.streaming import StreamingDeviationDetector

det = StreamingDeviationDetector(radius=10, ref_mult=3, n_sigmas=3.0)
for x in sensor_stream:
    is_deviant = det.update(x)
```

`MultiChannelStreamingDeviationDetector` and `classify_segments_multichannel` apply the same already-validated logic to multiple independent channels (the joints of a robotic arm, the axes of an IMU) without requiring a manual loop — each channel keeps its own reference window. Promoted by Dense-Evolution-Discovery after validation on two independent real physical domains (SO-101 robotic arm, real human IMU) — the same discipline already used for `stable_frame_filter.py` and `velocity_gated_stable_mask`. Full documentation (auto-generated from the real docstrings) on the [site](https://tatopenn-cell.github.io/Dense-Armor/anomaly/streaming/).

---

## `$ cusum --detectability`

A drift too slow to exceed, point by point, the instantaneous threshold of `classify_segments` escapes Arbiter by design. `cusum_detector` (`dense_armor.utility.drift.cusum`) accumulates small deviations over time instead of judging each point in isolation:

```python
from dense_armor.utility.drift.cusum import cusum_detector, detectability_report

flagged, cusum = cusum_detector(x, radius=10, ref_mult=3, k=0.5, h=20.0)

report = detectability_report(local_noise_scale=local_mad, k=0.5, h=5.0, candidate_shift=10.0)
# {'false_alarm_arl': ..., 'detection_arl': ..., 'shift_in_sigma': ...}
```

`detectability_report` estimates *before* running a benchmark how many samples are needed to detect a given shift given the detector's real local noise -- Reynolds (1975)/Siegmund (1985) theory, promoted by Dense-Evolution-Discovery after validation on two independent real physical domains (lidar, accelerometer): on the lidar the real latency always beats the theoretical estimate; on the accelerometer the result is genuinely mixed -- documented as is, not forced to coincide. Full documentation (auto-generated from the real docstrings) on the [site](https://tatopenn-cell.github.io/Dense-Armor/drift/cusum/).

## `$ drift_detectors --online`

Four streaming drift detectors with one interface, `update(x)` and `drift_detected`: CUSUM with a fixed reference (safe on a NaN or flat start), Page-Hinkley (unit-independent), ADWIN (Hoeffding bound) and KSWIN (Kolmogorov–Smirnov), compared on the CASPER robot stream. Full documentation on the [site](https://tatopenn-cell.github.io/Dense-Armor/drift/detectors/).

## `$ vision --native`

A camera, a folder of images or an array of frames becomes a stream of timestamped frames; `FrameFeatures` turns each frame into oriented-gradient histograms, intensity moments and Lucas–Kanade flow, and `RandomProjection` / `IncrementalPCA` reduce them, all computed by the library, one frame at a time. Full documentation on the [site](https://tatopenn-cell.github.io/Dense-Armor/vision/).

## `$ vision --patches`

`PatchFeatures` cuts each frame into patches and `PatchMemory` keeps the patches of normal images, online (bounded reservoir) or reduced once to a coreset; an image is scored by its most unusual patch, and the per-patch distances show where the anomaly is (PatchCore with descriptors computed by the library). Full documentation on the [site](https://tatopenn-cell.github.io/Dense-Armor/vision/patches/).

## `$ preprocessing --online`

Online scalers per feature and per joint, joint velocity, acceleration, jerk and power on real timestamps, text features (tokens, TF-IDF, signed hashing), feature selection and queue resampling for rare events, one sample at a time. Full documentation on the [site](https://tatopenn-cell.github.io/Dense-Armor/preprocessing/).

## `$ cluster --online`

Online k-means (farthest-first start, optional half-life for moving groups), DenStream micro-clusters with a fading window and outliers kept apart, and a visual vocabulary over image patches, one sample at a time. Full documentation on the [site](https://tatopenn-cell.github.io/Dense-Armor/cluster/).

## `$ calibration --platt`

`OnlinePlattScaling` (`dense_armor.utility.learn.calibration`, `pip install dense-armor[river]`) wraps any river classifier and recalibrates its probabilities one sample at a time, following Algorithm 1 of Gupta and Ramdas (ICML 2023, arXiv:2305.00070):

```python
from dense_armor.utility.learn.calibration import OnlinePlattScaling

model = OnlinePlattScaling(tree.HoeffdingTreeClassifier())
```

On Phishing the log-loss of a Hoeffding tree drops from 0.4535 to 0.3502. Full documentation on the [site](https://tatopenn-cell.github.io/Dense-Armor/learn/calibration/).

## `$ metric_learning --knn`

`dense_armor.utility.learn.metric_learning` (`pip install dense-armor[river]`) learns a k-NN distance online with OASIS, LEGO or POLA, ported from their papers, and plugs it into river through `MetricKNNClassifier`:

```python
from dense_armor.utility.learn.metric_learning import MetricKNNClassifier, POLA

model = MetricKNNClassifier(POLA(), n_neighbors=5)
```

Full documentation on the [site](https://tatopenn-cell.github.io/Dense-Armor/learn/metric_learning/).

## `$ metrics --online`

Online metrics, predict-then-learn evaluation with delayed labels, and event metrics for robots (detection delay, false alarms per hour, range-based F1, NAB). Full documentation on the [site](https://tatopenn-cell.github.io/Dense-Armor/metrics/).

## `$ rate_limiter --damping`

A robotic arm cannot execute an unlimited instantaneous jump without risk -- `rate_limited_follower` (`dense_armor.utility.control.rate_limiter`) limits how fast an applied command can physically change (velocity + acceleration), instead of trying to classify whether a deviation is real:

```python
from dense_armor.utility.control.rate_limiter import rate_limited_follower

applied = rate_limited_follower(raw_command, max_vel=2.0, max_accel=1.0)
```

Based on Berscheid & Kroger (2021), "Jerk-limited Real-time Trajectory Generation" (RSS 2021, arXiv:2105.04830) — causal by construction, verified directly. Promoted by Dense-Evolution-Discovery after validation on two independent real physical domains (SO-101, 14-DOF bimanual ALOHA): it always wins (400/400 real trials) on the real safety metric (maximum instantaneous jump), but it is **not** a signal cleaner — on average fidelity (RMSE) the picture is genuinely mixed between the two domains, not hidden. Full documentation (auto-generated from the real docstrings) on the [site](https://tatopenn-cell.github.io/Dense-Armor/control/rate_limiter/).

## `$ cbf_filter --spatial`

A command that moves at a perfectly safe speed but straight toward an obstacle remains dangerous — `rate_limiter` limits HOW FAST, `cbf_filter` limits WHERE:

```python
from dense_armor.utility.control.cbf_filter import cbf_filtered_trajectory

applied = cbf_filtered_trajectory(raw_command, obstacle=5.0, safe_dist=2.0, alpha_gain=2.0)
```

Based on Ames et al. (2019), "Control Barrier Functions: Theory and Applications" (2019 ECC, arXiv:1903.11199) — same theory as SAFER-Splat, applied to a known geometric obstacle instead of GPU Gaussian-Splatting perception (not available on every machine). A real numerical problem found and solved along the way: the CBF guarantee is continuous in time, sub-steps are needed (20/sample, default) to hold in discrete time on real commands that can jump a lot between one sample and the next. Promoted by Dense-Evolution-Discovery after validation on two independent real physical domains (SO-101, ALOHA): 100% invariance from safe starts on both, minimal invasiveness practically exact (99.9%+ on SO-101, perfectly exact on ALOHA). `cbf_safety_filter_live` is the same mathematics for a real control loop that reacts to one sensor tick at a time (a real `dt`, not a pre-recorded array) — promoted after a real ROS2/Ignition live loop needed it and had to rebuild it by hand. Full documentation (auto-generated from the real docstrings) on the [site](https://tatopenn-cell.github.io/Dense-Armor/control/cbf_filter/).

## `$ trajectory --quintic`

`rate_limiter` limits HOW FAST, `cbf_filter` limits WHERE — but neither generates a reference to follow. `quintic_trajectory` covers exactly this: a smooth, minimum-jerk path between two points, for any number of joints in a single call:

```python
from dense_armor.utility.control.trajectory import quintic_trajectory

t, q, v, a = quintic_trajectory(q0=[0.0], qf=[10.0], T=2.0)
```

Deliberately reduced compared to two real papers that propose much larger optimizers (full dynamics, URDF, torques) — Lozer, Scalera, Boscariol & Gasparetto (*Robotics and Autonomous Systems*) and Fried & Paternain (arXiv:2412.07859), both read in full before writing code — to the simplest and most universal piece: no URDF, no dynamics, no connection to the robot. Promoted by Dense-Evolution-Discovery after validation on two independent real physical domains (SO-101, ALOHA, 20 real joint excursions): the quintic's peak velocity is always lower than the real one recorded for the same start/end/duration — expected, not a bug, since it is the smoothest possible path. Full documentation (auto-generated from the real docstrings) on the [site](https://tatopenn-cell.github.io/Dense-Armor/control/trajectory/).

## `$ kinematic_controller --tracking`

`trajectory` generates a smooth reference, but something must turn it into a real command — `kinematic_tracking_controller` does this, at the same single-integrator scale as `rate_limiter`/`cbf_filter`:

```python
from dense_armor.utility.control.kinematic_controller import kinematic_tracking_controller

u_des = kinematic_tracking_controller(q=[0.2], q_ref=[0.5], qd_ref=[1.0], kp=5.0)
```

`u = qd_ref + kp*(q_ref - q)` — for the system `qdot = u` this makes the tracking error exactly `edot = -kp*e`: exponential convergence in closed form, for any reference trajectory, verified numerically. It is not "passivity-based" in the sense of the papers that motivated this research (Wu & Tan 2025, the real target, behind a paywall with no open copy found; Scruggs, real but requires convex optimization in infinite dimension; Califano et al., real but requires Hamiltonian mechanics) — honest about this, it is simpler. Promoted by Dense-Evolution-Discovery after validation on two real physical domains (SO-101, ALOHA), chained with `quintic_trajectory`: every real excursion recovers from a declared real initial error and converges. Full documentation (auto-generated from the real docstrings) on the [site](https://tatopenn-cell.github.io/Dense-Armor/control/kinematic_controller/).

## `$ rigid_body --urdf`

`rate_limiter`/`cbf_filter`/`trajectory`/`kinematic_controller` all work at the single-integrator level: you give a joint velocity, you receive a safe joint velocity. `RigidBodyModel` (`dense_armor.dynamics.urdf_dynamics`) is different — it requires a real physical description of the robot (a real URDF, including `.xacro`) and returns true torque-level dynamics:

```python
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel
import jax.numpy as jnp

model = RigidBodyModel("panda.urdf")   # or a .xacro path, expanded automatically
q = jnp.zeros(model.n)
M = model.mass_matrix(q)
qdd = model.forward_dynamics(q, jnp.zeros(model.n), jnp.zeros(model.n))
```

`model.n` is the number of real, independent degrees of freedom (a joint with `<mimic>` does not count separately). `mass_matrix`, `gravity_forces`, `bias_forces` and `forward_dynamics` (solves `M(q)*qdd + C(q,qd)*qd + g(q) = tau`) use the standard Lagrangian construction via autodiff (`jax.grad`/`jax.jvp`), not hand-written Christoffel symbols. `link_position`/`link_jacobian`/`link_pose`/`link_spatial_jacobian` work for any link named in the URDF, not only the end effector.

Promoted by Dense-Evolution-Discovery (Experiment 62) after validation on three independent real robots (Kinova Gen3 7-DoF, Kinova Gen3 6-DoF, Franka Emika Panda — different manufacturer, prismatic joints): symmetric/positive-definite mass matrix and energy conservation with correct RK4 convergence on all three. Full documentation on the [site](https://tatopenn-cell.github.io/Dense-Armor/dynamics/urdf_dynamics/).

## `$ passivity_cbf --controller`

`RigidBodyModel` gives M(q), gravity and forward dynamics for any robot — `solve_control_qp` (`dense_armor.dynamics.passivity_cbf_controller`) uses them to drive that robot toward an operational-space target safely, guaranteeing passivity of the tracking error and distance from kinematic singularities, both as constraints in a small QP:

```python
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel
from dense_armor.dynamics.passivity_cbf_controller import solve_control_qp

model = RigidBodyModel("panda.urdf")
qdd, tau, mu, h = solve_control_qp(model, "panda_hand", q, qd, p_des, pd_des, pdd_des, eps=0.03)
```

`eps` is the minimum manipulability index that the controller maintains — `mu` (returned) never drops much below it, even when the commanded target would otherwise push the robot straight into a singularity. The real position/velocity limits of each joint (from the URDF `<limit>` tag) are a third constraint, added only where the robot actually declares them.

Based on Kurtz, Wensing & Lin (2021, arXiv:2109.13349). Promoted by Dense-Evolution-Discovery (Experiment 61→63) after validation on the same three robots as `RigidBodyModel`, each pushed toward its own real singularity: manipulability maintained within 0.1-1.8% of the declared threshold in every case. A real OSQP bug found and solved along the way (infeasibility of the passivity+CBF QP, solved by falling back to CBF alone). Full documentation on the [site](https://tatopenn-cell.github.io/Dense-Armor/dynamics/passivity_cbf_controller/).

## `$ six_dof_cbf --pose`

`passivity_cbf_controller` tracks only the position of a link. `six_dof_pbc_cbf_controller.solve_control_qp` extends the same QP to the full pose — position and orientation together — using the link's 6xN spatial Jacobian instead of only the translation Jacobian:

```python
from dense_armor.dynamics.six_dof_pbc_cbf_controller import solve_control_qp

qdd, tau, mu, h = solve_control_qp(model, "panda_hand", q, qd, p_des, pd_des, pdd_des,
                                    r_des, w_des, wd_des, eps=0.03)
```

`r_des` is the desired orientation (rotation matrix), `w_des`/`wd_des` the desired angular velocity/acceleration in world frame. The orientation error uses the SO(3) formula of Lee, Leok & McClamroch (2010) — smooth everywhere, without the real gimbal lock of a roll-pitch-yaw formulation.

Promoted by Dense-Evolution-Discovery (Experiment 65). Validated with exact gravity compensation at zero error (machine precision) and real closed-loop convergence (initial offset 10cm/30°, RK4 over 1000 ticks, final error 1e-6 m / 1e-4 rad) — then on the same three robots as `passivity_cbf_controller`, where a second real OSQP infeasibility emerged (6-DoF manipulability can be well below the 3-DoF one at the same configuration) solved with a third fallback level (CBF alone, without box). Full documentation on the [site](https://tatopenn-cell.github.io/Dense-Armor/dynamics/six_dof_pbc_cbf_controller/).

## `$ xacro --macros`

Manufacturers publish robot descriptions as `.xacro` macros (parameterized blocks, mathematical expressions, `xacro:include`), not as flat URDF. `RigidBodyModel` now accepts `.xacro` directly:

```python
model = RigidBodyModel("panda_arm_hand.urdf.xacro")
model.n   # 8 -- 7 arm joints + 1 independent gripper coordinate
```

The real `xacro` package (the same expander from the ROS ecosystem, no ROS installation required) does the expansion — nothing about macros/math/conditionals is reimplemented here.

Promoted by Dense-Evolution-Discovery (Experiment 66), which found and solved a real inconsistency in the published Franka Panda macros (`clvrai/furniture`): a hand attachment link commented out in the arm macro but required by the hand one. New dependency: `xacro`. Full documentation on the [site](https://tatopenn-cell.github.io/Dense-Armor/dynamics/xacro_support/).

## `$ mimic --joints`

The two fingers of a gripper move together: closing one also closes the other. URDF expresses this with a `<mimic joint="..." multiplier="..." offset="..."/>` tag — the slave joint angle is always `multiplier * master_angle + offset`, never a free variable. `RigidBodyModel` now respects this instead of giving the slave joint its own independent coordinate:

```python
model.n                                    # 8, not 9 -- the two fingers share a single real DOF
model.mimic_map["panda_finger_joint2"]     # (master_dof_idx, multiplier, offset)
```

Forward kinematics substitutes `q[master] * multiplier + offset` for the mimic joint angle; the hand-built geometric Jacobian scales the mimic joint's local column by `multiplier` and sums it into its master's column, instead of giving it its own column.

Promoted by Dense-Evolution-Discovery (Experiment 67), verified against a real central finite difference of `link_pose` (not just plausibility): moving the master by 0.02 moves both fingertips by exactly 0.02 in opposite directions, and the hand-written Jacobian matches the numerical derivative within 1e-5. Full documentation on the [site](https://tatopenn-cell.github.io/Dense-Armor/dynamics/mimic_joints/).

---

## `$ mcp --server`

An MCP server (`dense_armor.mcp_server`) exposes 8 tools — `dense_armor_health`, `dense_armor_clean_signal` (full Orca, with `use_arbiter`), `dense_armor_detect_anomalies` (classification only), `dense_armor_robust_filter`, `dense_armor_heal_series`, plus `dense_armor_stream_start`/`_update`/`_end` (stateful session for a real-time sensor stream, one multiple channel at a time) — so an agent (Claude Code, Claude Desktop, or any MCP client) can clean a series, or follow a live sensor stream, without writing Python. Direct and in-process (no separate HTTP kernel, unlike the Dense-Evolution adapter — Dense-Armor has no web UI to share):

```bash
pip install -e ".[mcp]"
claude mcp add dense_armor -- dense-armor-mcp
```

It intentionally lives under `dense_armor.mcp_server`, not a simple `mcp_server` — with Dense-Evolution installed in the same environment (its adapter is named exactly `mcp_server`), a non-nested name is a real, verified collision, not a hypothesis. See [`dense_armor/mcp_server/README.md`](dense_armor/mcp_server/README.md) for the complete tool list and `test/test_mcp_server.py` for the end-to-end verification of each one.

---

## `$ robust_filters --standalone`

Four classic anomaly detectors (`dense_armor.utility.anomaly.robust_filters`), independent of `Armatura`/`Orca` — no dynamic model, no state, only arithmetic on a centered local window (intended for offline/batch cleaning, not the real-time causal loop; also suitable for a future embedded port, where one will not be able to rely on numpy):

```python
from dense_armor.utility.anomaly.robust_filters import pressure_valve

clean, anomalies, pressure, effective_threshold = pressure_valve(series)
```

`pressure_valve` combines Chauvenet's criterion (1863), Tukey's fences/IQR, the Hampel filter and iterative sigma-clipping — not with a vote (how many of the 4 flag a point), but with the classic minimum-variance combination (BLUE estimator): each method produces a local (center, scale) pair, and the weights are derived with a Lagrange multiplier (minimizes the variance of the weighted combination, constraint Σw=1 → w_k ∝ 1/scale_k²) — a method whose uncertainty inflates (e.g. Chauvenet when the window already contains an outlier, its mean/std are not robust) is automatically weighted less, without discarding it by hand. Final decision always binary (flagged/replaced with the local median, or intact).

The threshold itself is not fixed: the local window is compared with a wider one via Jensen-Shannon divergence, and it widens when the two distributions diverge (a real regime transition, not noise) — never the opposite, `effective_threshold >= pressure_threshold` always.

The four individual methods also remain callable one by one (`chauvenet_criterion`, `tukey_fences`, `hampel_filter`, `sigma_clip`) if a single criterion is needed instead of the combination.

---

## `$ toolkit --standalone`

Under `core/`/`utility/` there is also a second part of the package, largely independent of Armatura/Orca — most of these modules do not participate in the anomaly shield, they are tools in their own right that only share the JAX/NumPy backend. Three exceptions: `UniversalMemoryGuard`, `apply_fast_resonance` and `AIHardwareProfiler`, also called by Orca (see `$ internals` above) — they remain usable standalone anyway. Full documentation (auto-generated from the real docstrings) on the [site](https://tatopenn-cell.github.io/Dense-Armor/toolkit/toolkit/); here is the summary.

**Pipeline and chunking** (`dense_armor.core`)

- `DynamicAICodegen` — compiles a list of operation names (`relu`, `sigmoid`, `tanh`, `scale`, `dropout`, `clip`, `l2_normalize`, `identity`) into a JAX JIT-compiled pipeline via `lax.switch`, with block execution for long lists and gradient via autodiff (`compute_gradients`).
- `ImageChunker` (`dense_armor.core.chunk`) — splits/recomposes a large batch into fixed-size blocks, both for data arrays and for compiled instruction lists.
- `UniversalMemoryGuard` — checks RAM (and VRAM, if there is an NVIDIA GPU) before a heavy allocation, computes the number of blocks needed to fit; raises `MemoryPressureError` below threshold. Also used by `Orca` before every heavy chunk.

**Hardware and profiling** (`dense_armor.core`)

- `AIHardwareProfiler` — detects available CPU/RAM/backend and computes a safe maximum tensor size for the current host. Also used by `Orca` to auto-derive `chunk_threshold` (scales proportionally to the host instead of a fixed value equal for every machine; passing an explicit value overrides it without even instantiating the profiler).
- `StochasticAdversarialNoise` — injects synthetic noise (bitflip, dropout, Gaussian blur) into a tensor, preserving its norm; useful for generating attack data when one wants to test a detector.
- `PipelineProfiler` — measures JIT latency in microseconds (compilation warm-up separated from steady-state time) of a `DynamicAICodegen` pipeline or of `AdaptiveSignalStabilizer`.

**Tensors and configurations** (`dense_armor.core`)

- `TensorVault` — library of static (`invert`, `identity`, `edge_detector`, `blend`) and parametric (`scale_project`, `amplify`, `bias_shift`) transformation matrices, auto-detected backend/precision.
- `ParametricScenarioSimulator` — parallel Monte Carlo simulations (`jax.vmap`) on a scalar state over time, plus a stochastic decision collapse conditioned by the distribution.
- `BitwisePermutationEngine` — permutes the elements of a combinatorial vector (2^n space) via target/control bit masks.
- `SIGNAL_STABILIZER_PRESETS` (`dense_armor.core.preset`) — 4 calibrated configurations (`balanced_v2`, `cifar10_best_v1`, `pure_1d_time_v1`, `cifar10_hardened_lyapunov`) for the parameters of `AdaptiveSignalStabilizer`.

**Logging and provenance** (`dense_armor.core`)

- `MinimalConsoleFormatter` / `CompactJsonFormatter` (`dense_armor.core.logger`) — two `logging.Formatter`s: one readable at the console, one compact JSON for files.
- `AIEngineVisualizer` — exports a SHA-256 signed provenance archive (parameters, execution environment, integrity hashes) and textual reports of raw/filtered variance.

**Audio and data I/O** (`dense_armor.utility`)

- `anwav(fpath)` — analyzes a WAV file: peak, RMS, estimated loudness (LUFS), crest factor, with a compliance verdict.
- `diag(iorig, ifilt)` — differential comparison between two audio signals (file paths or NumPy arrays): structural fidelity, removed energy, distortion peak.
- `lodat(fpath, dname)` (`dense_armor.utility.misc.iodat`) — reads a tensor from an HDF5 or NetCDF file.
- `apply_fast_resonance(matrix, query)` (`dense_armor.utility.anomaly.resonance_search`) — cosine similarity score between a query and the rows of a matrix, modulated by `apply_damping_blend` (the same operator used by Orca). Also used by `Orca` in blind mode to recall a similar clean reference already seen in the past (see `$ internals` above).

Each tested individually (`test/test_chunk.py`, `test_compiler.py`, `test_memory.py`, `test_preset.py`, `test_tensor.py`, `test_noise.py`, `test_vector.py`, `test_profiler.py`, `test_visualizer.py`, `test_logger.py`, `test_anwav.py`, `test_diagnostic.py`, `test_iodat.py`, `test_resonance_search.py`). Requires `pip install "dense-armor[audio,data]"` for `anwav`/`diagnostic` (scipy) and `iodat` (h5py/netCDF4).

```python
from dense_armor.core import DynamicAICodegen, UniversalMemoryGuard, TensorVault, AIHardwareProfiler
from dense_armor.core.chunk import ImageChunker
from dense_armor.core.preset import SIGNAL_STABILIZER_PRESETS
from dense_armor.utility.misc.anwav import anwav
from dense_armor.utility.misc.iodat import lodat
```

---

## `$ error margin`

```python
orca.margine_ingresso      orca.margine_ingresso_medio      orca.margine_ingresso_max
orca.margine_uscita        orca.margine_uscita_medio        orca.margine_uscita_max
```

`|received value − corrected value|` — how much the shield had to shift a datum to clean it. It is not a covariance calibrated in the strict statistical sense, but it correlates well in tests: low when the correction is reliable, high when the shield is guessing blindly.

Complementary signal, not redundant, with `use_arbiter=True`: `orca.incertezza_arbitro`/`incertezza_arbitro_media` tells how ambiguous the *classification* itself is (near the deviant/clean or spike/regime boundary), not how large the correction was — a small correction with high uncertainty is an ambiguous case that went well by chance, not a truly safe one. See `$ orca --use_arbiter` above.

---

## `$ vs kalman-filter --honest`

It does not replace a Kalman filter — they solve different problems, period.

Random walk, 15% missing data, 3% huge spikes:

| method | MSE |
|---|---|
| no protection | ~21000 |
| Kalman *without* anti-outlier gating (the common case) | ~7300 |
| Kalman *with* anti-outlier gating and known dynamics | **~0.12** |
| Dense-Armor, blind mode | ~0.23 |

```diff
+ against an unprotected Kalman (the most common scenario in practice): wins clearly
+ a single huge spike throws off the Kalman gain and drags it along behind itself
+ zero setup: no process model to know, estimate or calibrate (no Q/R)
+ also works where Kalman does not apply at all: images, embeddings, generic tensors
```

**The advantage is freedom, not specialization.** A well-designed Kalman filter, calibrated on a *known* dynamic process, remains more precise on that single use case — but it requires knowing the system model in advance and recalibrating it for every new type of data. Dense-Armor is a **general filter**: no customization, no a priori knowledge required, it applies as is to any tensor (temporal or not). The price of this freedom is a bit less precision in the specific case where a known and calibrated dynamic model already exists — a small loss (~0.23 vs ~0.12 MSE in our test) compared to the advantage of never having to configure anything.

---

## `$ adversarial robustness --tested`

9 shared engine tests run all the way through, no crash, no NaN escaped. No defense ever below 64%. Real code, not just reported numbers: [`test/test_boundA.py`](test/test_boundA.py)–[`test_boundE.py`](test/test_boundE.py) — the same attacks (PGD/BIM/MI-FGSM, affine/elastic, Carlini-Wagner, DeepFool, Fourier) run against `dense_armor.core.engine.AdaptiveSignalStabilizer`, not a separate engine for the benchmark.

| attack | type | defense |
|---|---|---|
| PGD / BIM / MI-FGSM | gradient, 1000 steps | mitigated, final V 0.013-0.078 |
| affine / elastic | geometric, 50k iter | contained, V_inf 0.05-0.14 |
| Fourier broadband | frequency domain, 50k FFT iter | **99.78%+** |
| Carlini-Wagner (L2) | optimization | 78.96% |
| Carlini-Wagner (L∞) | optimization | **64.39%** — the weakest point found so far |
| DeepFool | optimization | 78.79% |
| combined (all together) | 150,140 total steps | no exploding gradient |

Honest: **C&W in L∞ norm is the attack that breaks through the most** among those tested. It is not a failure — it remains real protection — but it is the crack closest to a failure among all the tests done, and it must be known before relying on it against that specific scenario.

---

## `$ limits --known`

```
1. semantics       distinguishes geometric deviations, not meanings
2. blind mode without reference: good for avoiding collapses/NaNs, not for
                    accurately reconstructing data that is truly lost
3. generality      zero calibration required, applies to any tensor --
                    at the cost of a pinch of precision where there already is a
                    known and calibrated dynamic model (e.g. Kalman on pure series)
4. slow drift      invisible point by point, requires reference=historical_baseline
5. C&W L-inf norm  the weakest defense measured so far (64%, against 79-83%
                    of the other attack variants tested) -- see table above
6. adaptive        attacks built specifically to mimic the coherence of the
   adversarial      clean signal (in addition to PGD/BIM/MI-FGSM/C&W/DeepFool/Fourier
                    already tested) not yet covered by the suite
```

**Real cause of point 5, not just the number**: investigated in depth, not yet solved. C&W in L-inf norm constructs a spatially smooth perturbation over the entire grid in one shot (global gradient optimization). No purely local coherence check (comparing a point with its immediate neighbors, what this engine uses) can distinguish a genuinely smooth spatial structure from one built specifically to look like it -- it is the exact same statistical signal. Three targeted interventions were tried and empirically verified (rate-limit on volatility, long-term coherence anchor, rigid leash on maximum drift): none moved the number, one even made it worse. It is not a tuning limit -- an external reference is needed (not only local spatial context) to truly solve it.

---

## `$ license`

Business Source License 1.1 — free non-commercial use, converts to Apache 2.0 on `2029-06-01`. See [LICENSE.md](LICENSE.md).

`© 2026 Salvatore Pennacchio <jtatopenn@libero.it>`

Twin project of [Dense-Evolution](https://github.com/tatopenn-cell/Dense-Evolution) (NISQ quantum circuit simulator).
