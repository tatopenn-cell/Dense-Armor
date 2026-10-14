# Known limits

**In plain words:** A short list of the things Dense-Armor cannot do, or does only partly, so you know them before relying on it. For example, a very slow drift that is smaller than the noise in each reading can go unseen by tools that look at one reading at a time.

This page is the honest list of what Dense-Armor does *not* do. It is kept
short on purpose: every line is a real limit that a user should know before
relying on the shields.

## 1. The six limits

| # | limit | what it means |
|---|---|---|
| 1 | semantics | the engine distinguishes geometric deviations, not meanings |
| 2 | blind mode | without a reference it avoids collapses and NaN, it does not reconstruct truly lost data |
| 3 | generality | zero calibration required, applies to any tensor, at the cost of a pinch of precision where a known calibrated model exists |
| 4 | slow drift | invisible point by point, needs `reference=historical_baseline` |
| 5 | C&W L-inf | the weakest measured defense (64%, see [robustness.md](robustness.md)) |
| 6 | adaptive attacks | attacks built to mimic the coherence of the clean signal are not yet covered by the suite |

## 2. The two that need a paragraph

**Slow drift** (limit 4). A shield that works point by point cannot see a
drift that moves less than the noise in every single sample. The signal
looks locally coherent the whole time, and only the comparison with a
historical reference reveals the shift. When the deployment can provide
`reference=historical_baseline`, the drift becomes visible; without a
reference, it is not. The drift detectors in
[drift/detectors.md](drift/detectors.md) are the right tool when slow drift
matters.

**C&W in L-inf norm** (limit 5). Investigated in depth, not yet solved.
The attack constructs a spatially smooth perturbation over the entire grid
in one shot (global gradient optimisation). No purely local coherence check
can distinguish a genuinely smooth spatial structure from one built
specifically to look like it — they are the same statistical signal. Three
targeted interventions were tried (rate limit on volatility, long-term
coherence anchor, rigid leash on maximum drift): none moved the number, one
made it worse. It is not a tuning limit; an external reference is needed,
not a better local check. The full table is in
[robustness.md](robustness.md).

## 3. What is not in the list

The list is about the *shields*. It does not cover:

- **latency**: every estimator runs within `budget_s = 1e-3 s` on the
  reference desktop, measured per component in each section page.
- **memory**: every streaming estimator declares its `memory_class`; the
  ones that do not bound their internal buffers are not shipped.
- **test coverage**: the checks in `dense_armor.checks` run on every
  estimator; the coverage per module is in the changelog.

## Details

### References

The list is distilled from the development diary of the shield engine,
available in [shields/engine.md](shields/engine.md) and
[shields/armatura.md](shields/armatura.md). The adversarial robustness
numbers are in [robustness.md](robustness.md). The Kalman comparison is in
[vs_kalman.md](vs_kalman.md).

### Choices stated explicitly

- **The page is short.** Every line is a limit a user should read; there
  is no "future work" filler. If a limit is removed (e.g. adaptive
  attacks become covered by the suite), the line is removed in the same
  commit as the fix.
- **Limits are consequences of design, not bugs.** The engine uses local
  coherence on purpose: it is what makes it work without a model. The
  limits follow from that choice, they are not defects.

### Measured cost

Not applicable: this page lists limits, not costs. Each estimator's cost
is on its own section page.

### Development diary

The list was collected after the first user report asked "what does it
*not* do". The original README had only the wins; the limits section was
added to answer the question. Point 5 (C&W L-inf) is the only one that
took three dedicated experiments before being written down; the others
were known from the design.
