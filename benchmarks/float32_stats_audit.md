# Audit — batch T3 float32 stats (item 0)

Script: `python tools/census_float_accumulators.py origin/master <repo>`
Risultato: 61 classi con accumulatore, 1 nomina float32 (`Multinomial`).
Batch T3 = 11 classi (`CLASSI.md`). Falsi positivi e falsi negativi sotto.

## Le 11 classi del batch T3

| # | Classe | Tiene mean/var/cov? | dtype oggi | Decisione |
|---|---|---|---|---|
| 1 | `stats/moments.py::RunningMoments` | mean, m2, m3, m4 (scalari float) | float64 (Python `float` + `np.float64` da `_block`) | **in**, kernel su `learn_many` |
| 2 | `stats/moments.py::EWStats` | mean, var (EMA) | float64 (Python `float`) | **in** |
| 3 | `stats/moments.py::RunningMomentsVector` | delega a N `RunningMoments` | come `RunningMoments` (float64) | **in** (float32 passa per `RunningMoments`) |
| 4 | `stats/dependence.py::RunningCovariance` | mean_x, mean_y, m2_x, m2_y, c | float64 (Python `float`) | **in** |
| 5 | `stats/dependence.py::Autocorrelation` | mean, m2, c (lag) | float64 | **in** |
| 6 | `stats/dependence.py::RunningCorrelation` | delega a `RunningCovariance` | float64 | **in** (delega) |
| 7 | `stats/dependence.py::RollingCorrelation` | delega a `RollingCovariance` | float64 | **in** (delega) |
| 8 | `stats/dependence.py::RollingCovariance` | buffer finestra + cov calcolata | float64 | **in** |
| 9 | `stats/mixture.py::OnlineGaussianMixture` | means_, variances_, weights_ | `np.float64` (`np.full`/`np.asarray` senza dtype) | **in**, `_init_from_first` va reso parametrizzabile |
| 10 | `stats/shrinkage.py::OnlineShrinkageCovariance` | cov per coppia (delega a `RunningCovariance`) | float64 | **in** (delega) |
| 11 | `stats/student_t.py::StudentT` | mu_, log_sigma_ (scalari) | float64 (Python `float`) | **in** |

## Falsi positivi del census (fuori batch)

| Classe | Perché è fuori |
|---|---|
| `mcp_server/models.py::RobustFilterInput` | 11 righe, nessun `self.*` =, nessun `np.mean/var/cov`. Modello di input, non accumulatore. |
| `core/visualizer.py::AIEngineVisualizer` | `np.var` solo **locale** in `export_trend_report_text` (righe 96–97), nessuno stato persistente. |
| `roles/uncertainty.py::Estimate` | Contenitore con `std()` e `__repr__`. Non tiene mean/var come stato. |
| `utility/active/strategies.py::VariableRandomizedUncertainty` | Solo `self.delta` e `should_label`. Nessun accumulatore. |

## Falsi negativi

La scansione larga (`np.mean/var/cov/std` senza dtype) su tutto `dense_armor/` (escluso `neural/`)
trova solo:

- wrapper `ml/` (`ml/stats.py`, `ml/proba.py`, `ml/linear.py`, `ml/preprocessing.py`, `ml/vision.py`, `ml/tree.py`, `ml/base.py`) → già nel **batch T7**, non T3;
- calcoli **locali** `float(np.mean(...))` su liste temporanee (`core/engine.py`, `core/profiler.py`, `ml/anomaly.py`, `utility/anomaly/filters.py`, `utility/learn/online_classifiers.py`, `utility/protect/orca.py`, `utility/protect/streaming_arbiter.py`, `utility/rules/amrules.py`, `utility/timeseries/cycle.py`, `utility/vision/*`, `dense_armor/armatura.py`) → nessuno stato persistente, non sono accumulatori.

**Nessun falso negativo da aggiungere al batch T3.** `RunningMomentsVector`, `StudentT` e `OnlineShrinkageCovariance` (i tre noti di `CLASSI.md`) erano già stati aggiunti a mano e sono confermati dentro.

## Non nel batch

- `stats/quantiles.py::DDSketch` e `TDigest`: accumulano quantili, non mean/var. **Fuori.**
- `stats/similarity.py`: già con `dtype=np.float32` opt-in (righe 136, 144–145, 271, 303, 360, 391). **Fuori** (già fatto).
- `stats/sketches.py`, `stats/robust.py`: interi/hash/ordering. **Fuori.**

## Stato del branch

- branch `feature/float32-stats-kernel` creato da `origin/master` (`68a6f79`).
- `dense_armor/utility/stats/stable.py`: **assente**, da creare in item A.
- `test/utility/stats/`: **assente**, da creare in item B.
- `docs/stats/stable_variance.md`: **assente**, da creare in item E.
- `mkdocs.yml`: nav `stats/` alle righe 183–190 (7 voci). Aggiungere `stats/stable_variance.md` accanto a `Moments`.
