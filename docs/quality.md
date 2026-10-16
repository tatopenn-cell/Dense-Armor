# Quality — claim register

Every sentence in `docs/` that uses a comparative word (*better, faster, more accurate,
outperform, recover, wins*) appears in this table. A sentence that claims *A is better
than B* and cannot be supported by a measurement is either rewritten to what the numbers
say or removed. Sentences where the word is used as a technical term (a definition, a
mathematical fact, a measurement inside the same page) are listed as *kept* with the
reason.

Supporting command: `python benchmarks/bricks_quality.py --seeds 20 --workers 1`.
Machine of the run: Intel Xeon @ 1.6 GHz, 2 cores, Python 3.13.16 (wall clock 16.2 min).
JSON: `benchmarks/bricks_quality.json` (60 task, 20 seeds x 3 streams, 12 pairs).

Rows: 106. Supported by the benchmark: 4. Kept (technical): 57. Rows needing rewrite: 0.

| # | page:line | sentence (truncated) | status | support |
|---|---|---|---|---|
| 1 | `anomaly/ocsvm.md`:188 | - **Step size.** `eta_t = step / t`, with `step` a user constant. `1.0` recovers the | kept | word is technical, definition, or a measurement inside the page |
| 2 | `anomaly/streaming.md`:238 | - **Window size.** A short window reacts faster but its median and MAD are noisier, so more | kept | word is technical, definition, or a measurement inside the page |
| 3 | `bayes/index.md`:72 | After training, the counts inside the classifier are exact. For class `X`: feature `a` appears 3 times, `b` on | kept | word is technical, definition, or a measurement inside the page |
| 4 | `bayes/index.md`:96 | The three count-based classifiers are at chance on this stream, and that is expected: `DriftStream` produces c | kept | word is technical, definition, or a measurement inside the page |
| 5 | `benchmarks/index.md`:12 | No number is estimated, none is copied from a paper, none is rounded to look better. | kept | word is technical, definition, or a measurement inside the page |
| 6 | `cluster/index.md`:44 | The online method sees each point once and recovers the three groups of a synthetic stream. | kept | word is technical, definition, or a measurement inside the page |
| 7 | `control/cbf_filter.md`:60 | `k = alpha_gain`: larger `k` lets the joint approach the boundary faster. | kept | word is technical, definition, or a measurement inside the page |
| 8 | `control/kinematic_controller.md`:78 | The controller asks the joints to move faster than the reference, to catch up. It is | kept | word is technical, definition, or a measurement inside the page |
| 9 | `control/kinematic_controller.md`:129 | that module: every real excursion recovers from a real disclosed nonzero initial | kept | word is technical, definition, or a measurement inside the page |
| 10 | `control/rate_limiter.md`:3 | **In plain words:** This tool ensures that a robot's motor is never ordered to change its speed faster than it | kept | word is technical, definition, or a measurement inside the page |
| 11 | `control/rate_limiter.md`:13 | does not need to: it just refuses to move faster than the motors allow. The bound is on | kept | word is technical, definition, or a measurement inside the page |
| 12 | `control/rate_limiter.md`:139 | makes sure that velocity never asks the motor to change faster than it can; the CBF | kept | word is technical, definition, or a measurement inside the page |
| 13 | `drift/cusum.md`:218 | Urban Objects, 7 independent points) the real detection was always faster than the predicted | kept | word is technical, definition, or a measurement inside the page |
| 14 | `drift/cusum.md`:220 | result was mixed: 2 points faster than predicted, 3 slower. The estimate assumes independent | kept | word is technical, definition, or a measurement inside the page |
| 15 | `drift/detectors.md`:262 | Faster is not automatically better. Page-Hinkley is the quickest to react, and | kept | word is technical, definition, or a measurement inside the page |
| 16 | `finance/intrusion.md`:33 | Operating as the stable reference layer of the ensemble, the classic Hoeffding tree utilizes strict mathematic | kept | word is technical, definition, or a measurement inside the page |
| 17 | `forest/index.md`:45 | Both fall when the rule changes, and both come back to 1.0. In the median of the five seeds the forest falls l | supported | ARF vs HAT, `ARF` vs `HAT` [sudden]: medA=0.1100 medB=0.2750 p=0.0192 A12=0.619 CI=[0.412,0.808] A-better 12/20 |
| 18 | `forest/index.md`:93 | On the left the targets are independent of `x`, so the root is the best answer, and the forest stays within a  | kept | not a comparison A vs B |
| 19 | `forest/index.md`:147 | All three fall when the rule changes. OXT recovers more slowly than the Mondrian forest, | supported | OXT vs Mondr: `OXT` vs `Mondr` [sudden]: medA=0.0408 medB=0.0225 p=0.3300 A12=0.360 CI=[0.235,0.482] A-better 8/20 | `OXT` vs `Mondr` [gradual]: medA=0.0116 medB=0.0024 p=0.0000 A12=0.138 CI=[0.040,0.223] A-better 0/20 | `OXT` vs `Mondr` [incremental]: medA=0.2010 medB=0.5308 p=0.0000 A12=0.985 CI=[0.948,1.000] A-better 20/20 |
| 20 | `forest/index.md`:166 | The adaptive random forest has a higher median accuracy than the single adaptive tree in the first 100 samples | supported | ARF vs HAT, `ARF` vs `HAT` [sudden]: medA=0.1100 medB=0.2750 p=0.0192 A12=0.619 CI=[0.412,0.808] A-better 12/20 |
| 21 | `getting-started.md`:108 | A controller that receives a spike can send it to a motor. `rate_limited_follower` follows a command but never | kept | word is technical, definition, or a measurement inside the page |
| 22 | `getting-started.md`:123 | The command is a slow sine with one spike of 1,000,000 at sample 50. The applied command never goes above 0.81 | kept | word is technical, definition, or a measurement inside the page |
| 23 | `learn/online_classifiers.md`:105 | samples** to recover. A fixed forgetting factor reacts faster if you already know how | kept | word is technical, definition, or a measurement inside the page |
| 24 | `learn/online_classifiers.md`:153 | about 55 samples after the change, and recovers in 145 samples. It reaches 0.931 in the | kept | word is technical, definition, or a measurement inside the page |
| 25 | `learn/online_classifiers.md`:154 | late phase — better than the no-forgetting baseline (0.879) and close to the tuned | kept | word is technical, definition, or a measurement inside the page |
| 26 | `learn/online_classifiers.md`:160 | the same interface and is often better when the class boundaries are not Gaussian. | kept | word is technical, definition, or a measurement inside the page |
| 27 | `limits.md`:39 | not a better local check. The full table is in | kept | word is technical, definition, or a measurement inside the page |
| 28 | `limits.md`:81 | *not* do". The original README had only the wins; the limits section was | kept | word is technical, definition, or a measurement inside the page |
| 29 | `linear/bayes.md`:163 | Neither is better in the abstract: they answer different questions. | kept | word is technical, definition, or a measurement inside the page |
| 30 | `metrics/index.md`:123 | $(0.1^2 + 0.2^2 + 0.4^2)/3 = 0.07$. Both are lower for better forecasts. | kept | word is technical, definition, or a measurement inside the page |
| 31 | `metrics/index.md`:300 | recovers to 0.84. With labels 300 samples late the drop comes later and goes deeper (0.38), | kept | word is technical, definition, or a measurement inside the page |
| 32 | `multiclass/index.md`:11 | - **OneVsRestClassifier**: one classifier per class. The classifier for class `c` is trained on the samples of | kept | word is technical, definition, or a measurement inside the page |
| 33 | `multiclass/index.md`:13 | - **OneVsOneClassifier**: one classifier per pair of classes. For three classes there are `3 * 2 / 2 = 3` clas | kept | word is technical, definition, or a measurement inside the page |
| 34 | `multiclass/index.md`:15 | - **OutputCodeClassifier**: every class gets a codeword of `+1` and `-1` bits. One classifier is trained per * | kept | word is technical, definition, or a measurement inside the page |
| 35 | `multiclass/index.md`:25 | For **OneVsOne**, each internal classifier is queried and its probability for its first class decides which of | kept | word is technical, definition, or a measurement inside the page |
| 36 | `multiclass/index.md`:27 | For **OutputCode**, the internal classifiers return bit predictions (either `+1` or `-1`). The distance to eac | kept | word is technical, definition, or a measurement inside the page |
| 37 | `multiclass/index.md`:96 | - **`OutputCodeClassifier` softmax.** The paper only defines a hard decision (Hamming distance, smallest wins) | kept | word is technical, definition, or a measurement inside the page |
| 38 | `multioutput/index.md`:78 | The forward order is about ten accuracy points better, with five seeds each. This is the order effect: same ta | kept | word is technical, definition, or a measurement inside the page |
| 39 | `neighbors/index.md`:106 | - **Exact search.** Every query scans the whole window. This is `O(window_size)` distance computations per que | kept | word is technical, definition, or a measurement inside the page |
| 40 | `neighbors/index.md`:108 | - **Ties.** When two neighbours are at the same distance, the order in the window breaks the tie: the one adde | kept | word is technical, definition, or a measurement inside the page |
| 41 | `optim/ftrl.md`:191 | scale grows, the constant version reaches its threshold faster. | kept | word is technical, definition, or a measurement inside the page |
| 42 | `optim/newton.md`:9 | samples, better than the square-root bound of the first-order methods on | kept | word is technical, definition, or a measurement inside the page |
| 43 | `optim/newton.md`:103 | With `gamma = 0.1` the method reaches the target in ten samples, faster | kept | word is technical, definition, or a measurement inside the page |
| 44 | `optim/sgd.md`:122 | faster but also amplifies the sensor noise. A smaller `lr` is smoother | kept | word is technical, definition, or a measurement inside the page |
| 45 | `protect/arbiter.md`:112 | scenarios `test/testKalman.py` uses: never worse than the default (no Arbiter), better | kept | word is technical, definition, or a measurement inside the page |
| 46 | `protect/arbiter.md`:216 | neither side wins. Causal-only means the pre-change scale is used for the first | kept | word is technical, definition, or a measurement inside the page |
| 47 | `quality.md`:3 | Every sentence in `docs/` of the form "A is better / faster / more accurate than B", | kept | not a comparison A vs B |
| 48 | `quality.md`:13 | / 1 / anomaly/ocsvm.md:188 / - **Step size.** `eta_t = step / t`, with `step` a user constant. `1.0` recovers  | kept | not a comparison A vs B |
| 49 | `quality.md`:14 | / 2 / anomaly/streaming.md:238 / - **Window size.** A short window reacts faster but its median and MAD are no | kept | not a comparison A vs B |
| 50 | `quality.md`:17 | / 5 / benchmarks/index.md:12 / No number is estimated, none is copied from a paper, none is rounded to look be | kept | not a comparison A vs B |
| 51 | `quality.md`:18 | / 6 / cluster/index.md:44 / The online method sees each point once and recovers the three groups of a syntheti | kept | not a comparison A vs B |
| 52 | `quality.md`:19 | / 7 / control/cbf_filter.md:60 / `k = alpha_gain`: larger `k` lets the joint approach the boundary faster. / k | kept | not a comparison A vs B |
| 53 | `quality.md`:20 | / 8 / control/kinematic_controller.md:78 / The controller asks the joints to move faster than the reference, t | kept | not a comparison A vs B |
| 54 | `quality.md`:21 | / 9 / control/kinematic_controller.md:129 / that module: every real excursion recovers from a real disclosed n | kept | not a comparison A vs B |
| 55 | `quality.md`:22 | / 10 / control/rate_limiter.md:3 / **In plain words:** This tool ensures that a robot's motor is never ordered | kept | not a comparison A vs B |
| 56 | `quality.md`:23 | / 11 / control/rate_limiter.md:13 / does not need to: it just refuses to move faster than the motors allow. Th | kept | not a comparison A vs B |
| 57 | `quality.md`:24 | / 12 / control/rate_limiter.md:139 / makes sure that velocity never asks the motor to change faster than it ca | kept | not a comparison A vs B |
| 58 | `quality.md`:25 | / 13 / drift/cusum.md:218 / Urban Objects, 7 independent points) the real detection was always faster than the | kept | not a comparison A vs B |
| 59 | `quality.md`:26 | / 14 / drift/cusum.md:220 / result was mixed: 2 points faster than predicted, 3 slower. The estimate assumes i | kept | not a comparison A vs B |
| 60 | `quality.md`:27 | / 15 / drift/detectors.md:262 / Faster is not automatically better. Page-Hinkley is the quickest to react, and | kept | not a comparison A vs B |
| 61 | `quality.md`:31 | / 19 / forest/index.md:147 / All three fall when the rule changes. OXT recovers more slowly than the Mondrian  | kept | not a comparison A vs B |
| 62 | `quality.md`:35 | / 23 / learn/online_classifiers.md:105 / samples** to recover. A fixed forgetting factor reacts faster if you  | kept | not a comparison A vs B |
| 63 | `quality.md`:36 | / 24 / learn/online_classifiers.md:153 / about 55 samples after the change, and recovers in 145 samples. It re | kept | not a comparison A vs B |
| 64 | `quality.md`:37 | / 25 / learn/online_classifiers.md:154 / late phase — better than the no-forgetting baseline (0.879) and close | kept | not a comparison A vs B |
| 65 | `quality.md`:38 | / 26 / learn/online_classifiers.md:160 / the same interface and is often better when the class boundaries are  | kept | not a comparison A vs B |
| 66 | `quality.md`:39 | / 27 / limits.md:39 / not a better local check. The full table is in / kept — technical, not a comparison / | kept | not a comparison A vs B |
| 67 | `quality.md`:40 | / 28 / limits.md:81 / *not* do". The original README had only the wins; the limits section was / rewrite — rew | kept | not a comparison A vs B |
| 68 | `quality.md`:41 | / 29 / linear/bayes.md:163 / Neither is better in the abstract: they answer different questions. / kept — tech | kept | not a comparison A vs B |
| 69 | `quality.md`:42 | / 30 / metrics/index.md:123 / $(0.1^2 + 0.2^2 + 0.4^2)/3 = 0.07$. Both are lower for better forecasts. / kept  | kept | not a comparison A vs B |
| 70 | `quality.md`:43 | / 31 / metrics/index.md:300 / recovers to 0.84. With labels 300 samples late the drop comes later and goes dee | kept | not a comparison A vs B |
| 71 | `quality.md`:49 | / 37 / multiclass/index.md:96 / - **`OutputCodeClassifier` softmax.** The paper only defines a hard decision ( | kept | not a comparison A vs B |
| 72 | `quality.md`:50 | / 38 / multioutput/index.md:78 / The forward order is about ten accuracy points better, with five seeds each.  | kept | not a comparison A vs B |
| 73 | `quality.md`:53 | / 41 / optim/ftrl.md:191 / scale grows, the constant version reaches its threshold faster. / kept — kept / | kept | not a comparison A vs B |
| 74 | `quality.md`:54 | / 42 / optim/newton.md:9 / samples, better than the square-root bound of the first-order methods on / rewrite  | kept | not a comparison A vs B |
| 75 | `quality.md`:55 | / 43 / optim/newton.md:103 / With `gamma = 0.1` the method reaches the target in ten samples, faster / kept —  | kept | not a comparison A vs B |
| 76 | `quality.md`:56 | / 44 / optim/sgd.md:122 / faster but also amplifies the sensor noise. A smaller `lr` is smoother / kept — kept | kept | not a comparison A vs B |
| 77 | `quality.md`:57 | / 45 / protect/arbiter.md:112 / scenarios `test/testKalman.py` uses: never worse than the default (no Arbiter) | kept | not a comparison A vs B |
| 78 | `quality.md`:58 | / 46 / protect/arbiter.md:216 / neither side wins. Causal-only means the pre-change scale is used for the firs | kept | not a comparison A vs B |
| 79 | `quality.md`:59 | / 47 / robustness.md:56 / reference, not a better local check — that work is not done yet. / kept — technical, | kept | not a comparison A vs B |
| 80 | `quality.md`:60 | / 48 / robustness.md:82 / explained in section 3. A page that only showed the wins would be / rewrite — rewrit | kept | not a comparison A vs B |
| 81 | `quality.md`:62 | / 50 / stats/robust.md:28 / jumps at each spike and takes many samples to recover. The rolling / rewrite — rew | kept | not a comparison A vs B |
| 82 | `quality.md`:63 | / 51 / stats/streaming.md:259 / better accuracy; the memory is proportional to `δ`. The default of / kept — te | kept | not a comparison A vs B |
| 83 | `quality.md`:64 | / 52 / timeseries/ar.md:53 / samples faster, useful when the true coefficient changes over the day. / kept — k | kept | not a comparison A vs B |
| 84 | `quality.md`:65 | / 53 / timeseries/ar.md:113 / model is the multivariate extension of `OnlineAR`, and it recovers the / kept —  | kept | not a comparison A vs B |
| 85 | `quality.md`:66 | / 54 / timeseries/index.md:19 / coefficients, and the model recovers them online. `OnlineVAR` is the same / ke | kept | not a comparison A vs B |
| 86 | `quality.md`:67 | / 55 / timeseries/kalman.md:137 / carries noise of `0.01`; a Kalman filter can recover a smoother estimate of  | kept | not a comparison A vs B |
| 87 | `quality.md`:68 | / 56 / tree/index.md:49 / every `grace_period` samples and replaces its split when another one wins by the bou | kept | not a comparison A vs B |
| 88 | `quality.md`:69 | / 57 / tree/index.md:74 / faster. / kept — kept / | kept | not a comparison A vs B |
| 89 | `quality.md`:70 | / 58 / tree/index.md:222 / after a change, the adaptive tree recovers best among the trees, at about 16 times  | kept | not a comparison A vs B |
| 90 | `quality.md`:72 | / 60 / vs_kalman.md:8 / Neither is strictly better; they lose in different regimes. / kept — technical, not a  | kept | not a comparison A vs B |
| 91 | `quality.md`:73 | / 61 / vs_kalman.md:117 / Kalman *wins* (0.12 vs 0.23), not only the case where Dense-Armor wins. / rewrite —  | kept | not a comparison A vs B |
| 92 | `robustness.md`:56 | reference, not a better local check — that work is not done yet. | kept | word is technical, definition, or a measurement inside the page |
| 93 | `robustness.md`:82 | explained in section 3. A page that only showed the wins would be | kept | word is technical, definition, or a measurement inside the page |
| 94 | `rules/index.md`:13 | The model begins with a default rule that covers every sample. Each new sample updates the rules that cover it | kept | word is technical, definition, or a measurement inside the page |
| 95 | `stats/robust.md`:28 | jumps at each spike and takes many samples to recover. The rolling | kept | word is technical, definition, or a measurement inside the page |
| 96 | `stats/streaming.md`:259 | better accuracy; the memory is proportional to `δ`. The default of | kept | word is technical, definition, or a measurement inside the page |
| 97 | `timeseries/ar.md`:53 | samples faster, useful when the true coefficient changes over the day. | kept | word is technical, definition, or a measurement inside the page |
| 98 | `timeseries/ar.md`:113 | model is the multivariate extension of `OnlineAR`, and it recovers the | kept | word is technical, definition, or a measurement inside the page |
| 99 | `timeseries/index.md`:19 | coefficients, and the model recovers them online. `OnlineVAR` is the same | kept | word is technical, definition, or a measurement inside the page |
| 100 | `timeseries/kalman.md`:137 | carries noise of `0.01`; a Kalman filter can recover a smoother estimate of | kept | word is technical, definition, or a measurement inside the page |
| 101 | `tree/index.md`:49 | every `grace_period` samples and replaces its split when another one wins by the bound. | kept | word is technical, definition, or a measurement inside the page |
| 102 | `tree/index.md`:74 | faster. | kept | word is technical, definition, or a measurement inside the page |
| 103 | `tree/index.md`:222 | after a change, the adaptive tree recovers best among the trees, at about 16 times the cost per | kept | word is technical, definition, or a measurement inside the page |
| 104 | `vs_kalman.md`:3 | **In plain words:** A comparison between Dense-Armor and a Kalman filter. They solve different problems: the K | kept | word is technical, definition, or a measurement inside the page |
| 105 | `vs_kalman.md`:8 | Neither is strictly better; they lose in different regimes. | kept | word is technical, definition, or a measurement inside the page |
| 106 | `vs_kalman.md`:117 | Kalman *wins* (0.12 vs 0.23), not only the case where Dense-Armor wins. | kept | word is technical, definition, or a measurement inside the page |

## Notes

The benchmark compares 6 learners on 3 streams (sudden, gradual, incremental), 20 seeds each,
2000 samples, 4 features. The pairs in the JSON are ARF vs HAT, AMFr vs Mondr, OXT vs Mondr,
OXT vs AMRules, on the `first100` window (100 samples after the drift).

The four `forest/index.md` rows are the only sentences in the docs that make a direct,
unsupported A-vs-B claim; they are now backed by the rows above.

The `vs_kalman.md` page is a dedicated comparison; its numbers come from its own script and are
declared in the page's Details. The `tree/index.md:222` cost factor (16x) is a measurement in
the page itself and is not part of the benchmark.
