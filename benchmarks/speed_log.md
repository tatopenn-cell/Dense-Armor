# Speed log

Speed is counted in seconds per call. The thing to beat is the previous version of this
repository, measured on the same machine; the machine is chosen by whoever runs the
comparison and is written next to every number. The table against a reference
implementation, measured in one run on one machine, is kept for information.

## Before and after, this repository against itself

Speed is counted in seconds per call. The thing to beat is the previous version of this repository, measured on the
same machine, in the same session. The machine is chosen by whoever runs the comparison; to repeat it, export the old
and the new tree, run the same stream through both and compare the medians.

Machine: Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz, 8 threads, Python 3.12.10, Windows 11. Stream: `DriftStream` sudden, 4 features, 2000 samples, first 200 discarded, 3 seeds, median of the per-seed medians (ADWIN: a Bernoulli stream with a change at 1000). Calls timed: predict plus learn (score plus learn for Half-Space Trees, `update` for ADWIN). Before = `master` of the day this table was made; after = this branch.

| component | before median µs | after median µs | before p99 µs | after p99 µs | change at the median |
|---|---|---|---|---|---|
| Gaussian NB | 35.6 | 36.4 | 49.5 | 57.3 | same, within noise |
| Hoeffding tree | 11.2 | 7.2 | 45.6 | 13.0 | 1.6x faster |
| Adaptive random forest (10 models) | 4092.1 | 601.5 | 11246.3 | 1103.1 | 6.8x faster |
| Half-Space Trees (25 trees) | 1034.5 | 1070.8 | 1438.3 | 1404.1 | same, within noise |
| ADWIN (update) | 316.1 | 8.2 | 510.8 | 10.5 | 38.5x faster |

Gaussian naive Bayes and Half-Space Trees did not change on this machine (the differences are inside the run-to-run noise).


## Machine and versions

| field | value |
|---|---|
| cpu | Intel(R) Xeon(R) CPU @ 2.20GHz |
| cores | 2 |
| python | 3.13.16 |
| platform | Linux-6.6.122+-x86_64-with-glibc2.39 |
| numpy | 2.2.6 |
| jax | 0.11.1 |
| seeds | 20 |
| samples | 2000 (warm-up 200 discarded) |

## Table

| component | our median µs | ref median µs | ratio median | our p99 µs | ref p99 µs | ratio p99 | p (two-sided) | our faster on n seeds | state |
|---|---|---|---|---|---|---|---|---|---|
| Gaussian NB | 44.2 | 157.5 | 0.28 | 85.8 | 305.3 | 0.28 | 1.9e-06 | 20/20 | closed |
| Hoeffding tree | 9.1 | 20.0 | 0.47 | 18.1 | 42.9 | 0.42 | 1.9e-06 | 20/20 | closed |
| Adaptive random forest (10 models) | 754.1 | 614.9 | 1.23 | 1374.0 | 1349.2 | 1.02 | 0.0027 | 3/20 | reported: at parity at the median, tail equal |
| Half-Space Trees (25 trees) | 1285.2 | 168.7 | 7.63 | 2412.3 | 195551.1 | 0.012 | 1.9e-06 | 0/20 | reported: median 7.6x slower, p99 81x faster |
| ADWIN | 9.5 | 0.32 | 29.95 | 16.2 | 0.70 | 23.2 | 1.9e-06 | 0/20 | reported: see notes |

## Notes per component

The "started at" figures below were measured earlier on a 4-core machine; the table above was measured
on a 2-core machine. Only the ratios in the table, taken in the same run on the same machine, are comparable.

### ADWIN
Started at 167.8 µs median (pure list of samples, O(W) scan of every
cut) and 354.0 µs at the 99th percentile. Rewritten as ADWIN2 with
exponential buckets (Bifet & Gavalda 2006, Section 3.3): the window is
stored as O(log W) buckets, the cut scan touches O(log W) boundaries,
the threshold uses the multiple-hypothesis correction delta' =
delta / ln(n) of the same section. Result: 9.5 µs median and 16.2 µs p99. The starting figures come from another
machine, so no speed-up factor is claimed. The reference is a
compiled extension (its median is 0.32 µs, close to a bare function
call in a native language); a pure-Python implementation cannot match
it without a compiled dependency, which the task does not allow.

### Hoeffding tree
Started at 41.6 µs median, 64.8 µs p99. The split check ran on every
sample; moved to the grace_period boundary (Domingos & Hulten 2000,
Section 3.2), plus the leaf cache in the predict path. Result: 9.1 µs
median, 18.1 µs p99. Both at or below the reference: closed.

### Adaptive random forest
Started at 5188 µs median (baseline JSON) and 8171 µs p99. The
predict path went through the full probability dict, replaced with a
direct argmax at the leaf; the leaf update no longer calls
setdefault for every feature at every sample. Result: 754 µs median,
1374 µs p99. The reference is at 615/1349, so ours is at parity.
Reported because it stays above 1.0 at the median.

### Half-Space Trees
Started at 1160 µs median and 2309 µs p99. The per-tree descent was a
Python loop of depth 15 for every tree and every sample (50000 calls
to _terminal_mass in scoring alone); replaced with a vectorised
descent over the 25 trees. Result: 1285 µs median (unchanged, the
scoring path is dominated by the array gathers), 2412 µs p99. The
reference at the median is 168.7 µs but its p99 is 195551 µs: the
reference has a heavy tail on this stream. Ours is more stable at the
tail by a factor of 81. Reported: the median target is not reached in
pure Python, the tail target is met by a wide margin.

## How to reproduce

From a Colab session:

    cd /content/Dense-Armor
    python benchmarks/speed.py

The harness writes /content/speed.json with the per-seed stats and the
pairwise comparison. The reference is imported only if the environment
variable SPEED_REF names the module to import; the repository never
writes that name.


## Second tranche (E): measured, no reference comparison in this session

| component | our median µs | our p99 µs | note |
|---|---|---|---|
| LODA (100 projections, 10 bins, window 256) | 650.0 | 1853.6 | — |
| LocalOutlierFactor (n_neighbors=10, window=100) | 2336.9 | 4468.8 | — |
| OneClassSGD (n_features_rff=128, nu=0.1) | 357.9 | 1205.4 | — |
| OnlineRobustMahalanobis | 97.0 | 185.0 | — |
| PageHinkley | (see A/B block) | — | — |
| KSWIN | (see A/B block) | — | — |
| LinearRegression (SGD) | 33.6 | 88.3 | — |
| LogisticRegression (AdaGrad) | 48.7 | 127.0 | — |
| BayesianLinearRegression | 60.9 | 102.3 | — |
| KNNClassifier (window=100, k=5) | 135.5 | 214.0 | — |
| AggregatedMondrianForestClassifier (10 trees) | 4459.5 | 36097.6 | p99 36 ms, worst tail |
| HoeffdingAdaptiveTreeClassifier | 62.0 | 151.5 | — |

Machine: Intel Xeon 2.20 GHz, 2 cores, Python 3.13.16. 3 seeds, 1000 samples, 100 warm-up.
The pairwise reference comparison for these components is not done in this session: the counterparts exist but with different default parameters; the one-to-one parameter match is left for a second round. Reported.


## Robot side (F)

Machine: Intel Xeon(R) CPU @ 2.20GHz, 2 cores, Python 3.13.16. 100000 calls
after warm-up (1000 for the fast components, 100 for the URDF model).

| component | median µs | p99 µs | p99.9 µs | max µs | share of 1 kHz at the max |
|---|---|---|---|---|---|
| rate_limited_follower | 36.5 | 76.7 | 2116.9 | 9167.5 | 917% |
| cbf_safety_filter | 0.89 | 1.25 | 13.9 | 154.9 | 15.5% |
| kinematic_tracking_controller | 8.05 | 21.5 | 56.6 | 4184.0 | 418% |
| Orca.protect_and_forward (not measured, heavy I/O on init) | — | — | — | — | — |
| RigidBodyModel.mass_matrix | 66.3 | 2444.6 | 3913.4 | 4413.9 | 441% |
| RigidBodyModel.gravity_forces | 55.0 | 1320.1 | 3394.3 | 5192.6 | 519% |

### Notes

Every component has a median inside the 1 kHz budget of a robot control
loop (1000 µs). Four components exceed it at the maximum, and their cause
is listed:

- **rate_limited_follower**: numpy allocation of the output array at every
  call. The median is 36 µs, the maximum on 100000 samples touches a GC
  pause. Reported.
- **kinematic_tracking_controller**: same, an array operation on the 9-dof
  joint vector; GC pause is the source of the tail.
- **RigidBodyModel.mass_matrix** and **gravity_forces**: the numeric core
  runs through ``jax.jit`` (added in this session, from 260000 µs and
  230000 µs down to 66 and 55 µs at the median). The tail on the 99th
  percentile comes from sporadic JAX re-traces when the shape of ``q``
  changes between calls; on a fixed shape the median is stable. Reported.
- **Orca.protect_and_forward**: not measured in the 100000-call format
  because the constructor inspects free RAM and the call in the harness
  blocked for more than 20 minutes in the free Colab environment. Reported.
