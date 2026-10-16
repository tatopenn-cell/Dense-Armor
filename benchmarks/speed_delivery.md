# Task V delivery — speed of the online estimators

Branch `taskV-speed`. One branch, one pull request.

## Top table

| item | state | command |
|---|---|---|
| A profile before touching | done for the 4 main components (ADWIN, HT, HST, ARF) | `python benchmarks/speed.py` |
| B order ADWIN -> HT -> HST -> ARF | done | `git log --oneline taskV-speed` |
| C rules (same behaviour, no new dependency, tail first) | respected | `pytest test/ -q` |
| D harness + log | done | `benchmarks/speed.py`; `benchmarks/speed_log.md` |
| E second tranche | **measured** (12 components), reference comparison **reported** for this session | `benchmarks/speed_log.md` |
| F robot side | **reported**, not measured | — |
| G public text | done for the 4 components | `benchmarks/speed_log.md` |

## Main results (20 seeds, 2000 samples, 200 warm-up)

| component | our median µs | ref median µs | ratio median | our p99 µs | ref p99 µs | state |
|---|---|---|---|---|---|---|
| Gaussian NB | 44.2 | 157.5 | 0.28 | 85.8 | 305.3 | closed |
| Hoeffding tree | 9.1 | 20.0 | 0.47 | 18.1 | 42.9 | closed |
| Adaptive random forest | 754.1 | 614.9 | 1.23 | 1374.0 | 1349.2 | reported (parity) |
| Half-Space Trees | 1285.2 | 168.7 | 7.63 | 2412.3 | 195551.1 | reported (p99 81x better) |
| ADWIN | 9.5 | 0.32 | 29.95 | 16.2 | 0.70 | reported |

## What changed

- `adwin.py`: ADWIN2 (Bifet & Gavalda 2006, Section 3.3), exponential buckets, correction `delta' = delta / ln(n)`. 167.8 -> 9.5 median, 354 -> 16.2 p99.
- `hoeffding.py`: split check at the grace_period boundary (Domingos & Hulten 2000, Section 3.2), leaf cache, direct argmax. 41.6 -> 9.1 median.
- `hst.py`: per-tree descent vectorised over the 25 trees (Tan et al. 2011). p99 9824 -> 2338.
- `arf.py`: predict path avoids the proba dict; leaf update no setdefault per feature. 5188 -> 754 median.
- `benchmarks/speed.py`: p99.9, max, first-call, incremental save, resume.
- `benchmarks/speed_log.md`: 20-seed table + second tranche.
- Neutral names in docs, benchmarks and tests (A+B of the cleanup).
- Bridge module renamed from `bridges/river.py` to `bridges/online.py`; extra renamed from `[river]` to `[online]` in `pyproject.toml` and CI.

## Reported

- ADWIN, Half-Space Trees: median target not reachable in pure Python vs a compiled extension.
- ARF: parity at the median.
- Item E: measurements done, reference comparison for these components left to a second round.
- Item F: not measured.
