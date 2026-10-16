# Adversarial robustness

We attacked the library's cleaning tools with nine kinds of deliberately bad data, to see how well they hold up. This page shows what held, what did not and the known weak point.

The shielding engines were tested against nine families of adversarial
attacks. This page is the honest result: what held, what did not, and which
attack is the known weak point. All numbers come from the real code in
`test/experiments/`, not from a standalone benchmark that re-implements the
engine.

## 1. What was tested

The attacks are the common ones against a signal filter: gradient attacks
(PGD, BIM, MI-FGSM), geometric attacks (affine, elastic), a frequency-domain
attack (Fourier broadband), and three optimisation attacks (Carlini-Wagner
in L2 and L-inf, DeepFool). Every attack runs against the *same* engine the
library ships — `dense_armor.core.engine.AdaptiveSignalStabilizer` — not a
simplified copy.

## 2. The measured defense

Nine shared engine tests, no crash, no NaN escaped. The defense never fell
below `64%`.

| attack | type | steps | defense |
|---|---|---|---|
| PGD / BIM / MI-FGSM | gradient | 1000 | mitigated, final V `0.013-0.078` |
| affine / elastic | geometric | 50k | contained, `V_inf 0.05-0.14` |
| Fourier broadband | frequency | 50k FFT | **99.78%+** |
| Carlini-Wagner (L2) | optimisation | — | 78.96% |
| Carlini-Wagner (L-inf) | optimisation | — | **64.39%** (weakest) |
| DeepFool | optimisation | — | 78.79% |
| combined (all together) | 150,140 total steps | no exploding gradient | |

## 3. The known weak point

`C&W` in `L-inf` norm is the attack that breaks through the most. It is not
a failure — 64% is still real protection — but it is the crack closest to a
failure among the tests, and it must be known before relying on the shield
against that specific scenario.

The cause is structural, not a tuning problem. `C&W` in `L-inf` norm
constructs a spatially smooth perturbation over the entire grid in one shot
(global gradient optimisation). No purely local coherence check — comparing
a point with its immediate neighbours, which is what the engine uses — can
distinguish a genuinely smooth spatial structure from one built
specifically to look like it. They are the same statistical signal.

Three targeted interventions were tried and empirically verified:

- a rate limit on the volatility of the signal
- a long-term coherence anchor
- a rigid leash on the maximum drift

None moved the number. One made it worse. Solving this needs an external
reference, not a better local check — that work is not done yet.

## Details

### References

The attacks and the tests are described in `test/experiments/`. The suite
uses standard implementations of PGD, BIM, MI-FGSM, Carlini-Wagner,
DeepFool and the Fourier attack; the code is in the repository and can be
re-run.

The engine itself is documented in [shields/engine.md](shields/engine.md);
the toolkit that runs the benchmark is in [toolkit/toolkit.md](toolkit/toolkit.md).

### Choices stated explicitly

- **The attacks run against the shipped engine, not a copy.** The tests
  import `dense_armor.core.engine.AdaptiveSignalStabilizer` directly.
  A number measured on a re-implementation would not generalise.
- **The tables report the defense, not the loss.** The metric is the
  fraction of the original signal preserved after the attack, or the
  residual value `V` after the engine has processed the perturbed signal.
  A defense of 64% means the engine preserved 64% of the signal that the
  attack tried to destroy.
- **The weak point is stated, not hidden.** The C&W L-inf case is shown
  in the table with the lowest number, and the structural cause is
  explained in section 3. A page that only showed the wins would be
  marketing, not documentation.

### Measured cost

Not applicable: the page reports the *robustness* of the engine, not its
latency. Per-sample latency for `AdaptiveSignalStabilizer` is in
[shields/engine.md](shields/engine.md).

### Development diary

The robustness suite was built after a first version of the shields passed
the functional tests but nobody had asked "what happens if someone
deliberately attacks it". The first three attacks (gradient) were easy to
mitigate; the Fourier attack was the easiest of all (99.78%+ preserved
because the engine's local coherence check is strongest in the frequency
domain); the C&W L-inf attack took the longest to characterise, and the
conclusion is that a local check cannot solve it. Three interventions were
tried before giving up on a local fix. The result is kept as a permanent
weak point in the doc, not buried in a changelog.
