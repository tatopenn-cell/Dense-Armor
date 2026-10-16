# Numbers on the pages edited this week

One row per sentence with a numerical claim that names a metric or a unit
(ROC-AUC, recall, precision, ms, %, VaR). The command is the script that
prints the number, or the code block on the page.

Rows: 274.

| # | page:line | sentence (truncated) | command |
|---|---|---|---|
| 1 | `docs/finance/credit_scoring.md`:12 | To benchmark this capability on historical data, the dual-tree streaming pipeline was run on **45,211 telemarketing interactions** | `docs/assets/finance/measure_market.py` |
| 2 | `docs/finance/credit_scoring.md`:51 | / ROC-AUC, MondrianForest / 0.859 / | `docs/assets/finance/measure_market.py` |
| 3 | `docs/finance/credit_scoring.md`:52 | / ROC-AUC, SGT / 0.833 / | `docs/assets/finance/measure_market.py` |
| 4 | `docs/finance/credit_scoring.md`:53 | / ROC-AUC, ensemble / **0.875** / | `docs/assets/finance/measure_market.py` |
| 5 | `docs/finance/credit_scoring.md`:59 | / Recall / 74.3% / | `docs/assets/finance/measure_market.py` |
| 6 | `docs/finance/credit_scoring.md`:60 | / Precision / 39.1% / | `docs/assets/finance/measure_market.py` |
| 7 | `docs/finance/credit_scoring.md`:61 | / F1 / 51.3% / | `docs/assets/finance/measure_market.py` |
| 8 | `docs/finance/credit_scoring.md`:62 | / Training time (Kaggle CPU) / 166 s / | `docs/assets/finance/measure_market.py` |
| 9 | `docs/finance/credit_scoring.md`:66 | Traditional model deployments rely mechanically on a generic 0.5 classification cutoff. On this dataset, a 0.5 threshold reaches o | `docs/assets/finance/measure_market.py` |
| 10 | `docs/finance/credit_scoring.md`:68 | Calibrating the operational threshold down to **0.25 pushes the Recall rate to 74.3%** with a Precision of 39.1%. Roughly **two ou | `docs/assets/finance/measure_market.py` |
| 11 | `docs/finance/execution.md`:11 | A desk in London. The automated strategy wants to buy 700 BTC in one shot. On a normal day it buys 5. If the order goes through, t | `docs/assets/finance/measure_market.py` |
| 12 | `docs/finance/execution.md`:34 | Orca costs about 3 milliseconds per call. On real BTC/USD trades only 11 orders out of 1,000 (1.1%) fell in this range, so the typ | `docs/assets/finance/measure_market.py` |
| 13 | `docs/finance/execution.md`:47 | One thousand trades from the Coinbase public API on 14 October 2026. Sizes from 0.00000001 to 0.7 BTC. | `docs/assets/finance/measure_market.py` |
| 14 | `docs/finance/execution.md`:51 | / Passed directly (no Orca) / 967 (96.7%) / | `docs/assets/finance/measure_market.py` |
| 15 | `docs/finance/execution.md`:52 | / Went through Orca / 11 (1.1%) / | `docs/assets/finance/measure_market.py` |
| 16 | `docs/finance/execution.md`:53 | / Refused (10x rule) / 22 (2.2%) / | `docs/assets/finance/measure_market.py` |
| 17 | `docs/finance/fraud.md`:35 | The SGD tree does the same job but each rectangle is chosen carefully, not randomly. It uses the gradient of the loss to decide wh | `docs/assets/finance/measure_market.py` |
| 18 | `docs/finance/fraud.md`:47 | The pipeline was trained on 40,000 real credit card transactions from the OpenML creditcard dataset. The dataset has 284,807 rows  | `docs/assets/finance/measure_market.py` |
| 19 | `docs/finance/fraud.md`:51 | / ROC-AUC, Mondrian / 0.969 / | `docs/assets/finance/measure_market.py` |
| 20 | `docs/finance/fraud.md`:52 | / ROC-AUC, SGT / 0.926 / | `docs/assets/finance/measure_market.py` |
| 21 | `docs/finance/fraud.md`:53 | / ROC-AUC, ensemble / 0.969 / | `docs/assets/finance/measure_market.py` |
| 22 | `docs/finance/fraud.md`:58 | / Recall on frauds / 67.6% / | `docs/assets/finance/measure_market.py` |
| 23 | `docs/finance/fraud.md`:59 | / Precision / 95.8% / | `docs/assets/finance/measure_market.py` |
| 24 | `docs/finance/fraud.md`:60 | / Training time (Kaggle CPU) / 189 s / | `docs/assets/finance/measure_market.py` |
| 25 | `docs/finance/index.md`:41 | All quantitative parameters, computational latencies, and accuracy matrices can be audited and reproduced locally by executing the | `docs/assets/finance/measure_market.py` |
| 26 | `docs/finance/intrusion.md`:12 | To benchmark this resilience on a public dataset, the online dual-tree pipeline was deployed on **125,973 labelled network flows** | `docs/assets/finance/measure_market.py` |
| 27 | `docs/finance/intrusion.md`:29 | Standard incremental decision trees lock in division attributes permanently, degrading rapidly when traffic distributions drift. T | `docs/assets/finance/measure_market.py` |
| 28 | `docs/finance/intrusion.md`:51 | / ROC-AUC, adaptive tree / 0.9553 / | `docs/assets/finance/measure_market.py` |
| 29 | `docs/finance/intrusion.md`:52 | / ROC-AUC, base tree / 0.9503 / | `docs/assets/finance/measure_market.py` |
| 30 | `docs/finance/intrusion.md`:53 | / ROC-AUC, ensemble / **0.9755** / | `docs/assets/finance/measure_market.py` |
| 31 | `docs/finance/intrusion.md`:59 | / Recall on attacks / 93.6% / | `docs/assets/finance/measure_market.py` |
| 32 | `docs/finance/intrusion.md`:60 | / Precision / 85.1% / | `docs/assets/finance/measure_market.py` |
| 33 | `docs/finance/intrusion.md`:61 | / Training time (Kaggle CPU) / 65 s / | `docs/assets/finance/measure_market.py` |
| 34 | `docs/finance/intrusion.md`:65 | Relying on standalone detection models creates an operational trade-off between false alarms and missed exploits. By orchestrating | `docs/assets/finance/measure_market.py` |
| 35 | `docs/finance/intrusion.md`:67 | For enterprise infrastructure, this translates into a **93.6% Recall rate**, so 93 out of every 100 malicious flows are caught, wh | `docs/assets/finance/measure_market.py` |
| 36 | `docs/finance/risk.md`:3 | Every morning, a bank or investment fund needs an estimate of how much money it could lose if the market turns bad. Instead of loa | `docs/assets/finance/measure_market.py` |
| 37 | `docs/finance/risk.md`:19 | / **DDSketch** / Bounded tail-risk tracking / Maintains a fully bounded summary of extreme asset distributions, calculating high-a | `docs/assets/finance/measure_market.py` |
| 38 | `docs/finance/risk.md`:25 | *Each bar is the loss of one day. The red line is the 99% Value-at-Risk at the end of the run: 14 days lose more.* | `docs/assets/finance/measure_market.py` |
| 39 | `docs/finance/risk.md`:29 | Quantifying exposure boundaries within severe market corrections requires tracking the exact shape of distribution tails. Traditio | `docs/assets/finance/measure_market.py` |
| 40 | `docs/finance/risk.md`:40 | 1. **Quantile Mapping:** The session return vector is fed into **DDSketch**, updating distribution statistics and emitting calibra | `docs/assets/finance/measure_market.py` |
| 41 | `docs/finance/risk.md`:51 | / VaR 99% / 2.93% / | `docs/assets/finance/measure_market.py` |
| 42 | `docs/finance/risk.md`:52 | / VaR 99.9% / 4.28% / | `docs/assets/finance/measure_market.py` |
| 43 | `docs/finance/risk.md`:53 | / Days with a loss above the final VaR 99% / 14 out of 1,253 (1.1%) / | `docs/assets/finance/measure_market.py` |
| 44 | `docs/finance/risk.md`:54 | / Days with a loss above the final VaR 99.9% / 3 out of 1,253 (0.24%) / | `docs/assets/finance/measure_market.py` |
| 45 | `docs/finance/risk.md`:55 | / Days above a 5% loss / 1 out of 1,253 (4 April 2025, −5.97%) / | `docs/assets/finance/measure_market.py` |
| 46 | `docs/finance/risk.md`:56 | / Days with a volatility-drift alarm / 49 out of 1,253 (3.9%) / | `docs/assets/finance/measure_market.py` |
| 47 | `docs/finance/risk.md`:61 | The final VaR 99% is the 99th percentile of these same 1,253 days, so about 1% of them fall above it: **1.1% of trading days** (14 | `docs/assets/finance/measure_market.py` |
| 48 | `docs/finance/single_ticker.md`:47 | / Time per day / 0.59 ms median (Kaggle CPU) / | `docs/assets/finance/measure_market.py` |
| 49 | `docs/finance/single_ticker.md`:49 | The 50 days include the crashes of April 3rd and 4th 2025 (−9.25% and −7.29%), the rebound of April 9th 2025 (+15.33%), the rise o | `docs/assets/finance/measure_market.py` |
| 50 | `docs/finance/static_vs_online.md`:9 | The simplest way to find an anomalous day is a fixed number: if the return is bigger than 3%, call it anomalous. It is one line of | `docs/assets/finance/measure_market.py` |
| 51 | `docs/finance/static_vs_online.md`:11 | Five years of S&P 500 data show the limit of that approach. On August 5th 2024 the index closed at −2.997%: three thousandths of a | `docs/assets/finance/measure_market.py` |
| 52 | `docs/finance/static_vs_online.md`:17 | / StaticAnalysis / Flag days above a fixed cutoff / 7 days above 3% after the warm-up / | `docs/assets/finance/measure_market.py` |
| 53 | `docs/finance/static_vs_online.md`:30 | / 2% / 74 / 36 / | `docs/assets/finance/measure_market.py` |
| 54 | `docs/finance/static_vs_online.md`:31 | / 3% / 17 / 7 / | `docs/assets/finance/measure_market.py` |
| 55 | `docs/finance/static_vs_online.md`:32 | / 4% / 6 / 4 / | `docs/assets/finance/measure_market.py` |
| 56 | `docs/finance/static_vs_online.md`:44 | Six days are caught by both, 2022-11-30 (+3.09%), 2025-05-12 (+3.26%) and four sessions of the April 2025 tariff weeks: | `docs/assets/finance/measure_market.py` |
| 57 | `docs/finance/static_vs_online.md`:48 | / 2025-04-03 / −4.84% / | `docs/assets/finance/measure_market.py` |
| 58 | `docs/finance/static_vs_online.md`:49 | / 2025-04-04 / −5.97% / | `docs/assets/finance/measure_market.py` |
| 59 | `docs/finance/static_vs_online.md`:50 | / 2025-04-09 / +9.52% / | `docs/assets/finance/measure_market.py` |
| 60 | `docs/finance/static_vs_online.md`:51 | / 2025-04-10 / −3.46% / | `docs/assets/finance/measure_market.py` |
| 61 | `docs/finance/static_vs_online.md`:53 | These are big price moves, above 3% by definition, so the fixed cutoff catches them too. | `docs/assets/finance/measure_market.py` |
| 62 | `docs/finance/static_vs_online.md`:57 | One day: **2022-11-10, +5.54%**. The online detectors did not flag it. The fixed cutoff flags every day above 3% whatever the rece | `docs/assets/finance/measure_market.py` |
| 63 | `docs/finance/static_vs_online.md`:67 | / 2022-12-16 / −1.11% / | `docs/assets/finance/measure_market.py` |
| 64 | `docs/finance/static_vs_online.md`:68 | / 2023-03-17 / −1.10% / | `docs/assets/finance/measure_market.py` |
| 65 | `docs/finance/static_vs_online.md`:69 | / 2023-06-16 / −0.37% / | `docs/assets/finance/measure_market.py` |
| 66 | `docs/finance/static_vs_online.md`:70 | / 2023-09-15 / −1.22% / | `docs/assets/finance/measure_market.py` |
| 67 | `docs/finance/static_vs_online.md`:71 | / 2023-12-15 / −0.01% / | `docs/assets/finance/measure_market.py` |
| 68 | `docs/finance/static_vs_online.md`:72 | / 2024-03-15 / −0.65% / | `docs/assets/finance/measure_market.py` |
| 69 | `docs/finance/static_vs_online.md`:73 | / 2024-06-21 / −0.16% / | `docs/assets/finance/measure_market.py` |
| 70 | `docs/finance/static_vs_online.md`:74 | / 2024-09-20 / −0.19% / | `docs/assets/finance/measure_market.py` |
| 71 | `docs/finance/static_vs_online.md`:75 | / 2024-12-20 / +1.09% / | `docs/assets/finance/measure_market.py` |
| 72 | `docs/finance/static_vs_online.md`:76 | / 2025-03-21 / +0.08% / | `docs/assets/finance/measure_market.py` |
| 73 | `docs/finance/static_vs_online.md`:77 | / 2025-06-20 / −0.22% / | `docs/assets/finance/measure_market.py` |
| 74 | `docs/finance/static_vs_online.md`:78 | / 2026-03-20 / −1.51% / | `docs/assets/finance/measure_market.py` |
| 75 | `docs/finance/static_vs_online.md`:80 | Each has a return smaller than 3%, so the fixed cutoff never sees them. The **ratio between price and volume** differed from a nor | `docs/assets/finance/measure_market.py` |
| 76 | `docs/finance/static_vs_online.md`:82 | **Other days.** 25 more, five of them with a move of 2% or more: | `docs/assets/finance/measure_market.py` |
| 77 | `docs/finance/static_vs_online.md`:86 | / 2024-08-05 / −3.00% (−2.997% before rounding) / | `docs/assets/finance/measure_market.py` |
| 78 | `docs/finance/static_vs_online.md`:87 | / 2024-11-06 / +2.53% (the day after the US election) / | `docs/assets/finance/measure_market.py` |
| 79 | `docs/finance/static_vs_online.md`:88 | / 2024-12-18 / −2.95% / | `docs/assets/finance/measure_market.py` |
| 80 | `docs/finance/static_vs_online.md`:89 | / 2025-03-10 / −2.70% / | `docs/assets/finance/measure_market.py` |
| 81 | `docs/finance/static_vs_online.md`:90 | / 2025-10-10 / −2.71% / | `docs/assets/finance/measure_market.py` |
| 82 | `docs/finance/static_vs_online.md`:92 | All five stayed under the 3% cutoff. The online detectors saw the price-and-volume pair shift and flagged them. | `docs/assets/finance/measure_market.py` |
| 83 | `docs/finance/surveillance.md`:14 | To benchmark this capability on historical data, the Dense-Armor multi-stage pipeline was deployed over **1,253 historical trading | `docs/assets/finance/measure_market.py` |
| 84 | `docs/finance/surveillance.md`:38 | While macroeconomic culls are immediately visible, slow and systemic modifications in asset distributions frequently escape conven | `docs/assets/finance/measure_market.py` |
| 85 | `docs/finance/surveillance.md`:42 | Calculating capital risk distribution under strict regulatory frameworks typically demands exhaustive retrospective database sweep | `docs/assets/finance/measure_market.py` |
| 86 | `docs/finance/surveillance.md`:49 | - **August 5th, 2024:** Institutional liquidation ("Black Monday"), S&P 500 contraction at −3.00% | `docs/assets/finance/measure_market.py` |
| 87 | `docs/finance/surveillance.md`:50 | - **November 6th, 2024:** Post-election macroeconomic readjustment, +2.53% | `docs/assets/finance/measure_market.py` |
| 88 | `docs/finance/surveillance.md`:51 | - **April 3rd and 4th, 2025:** Tariff implementation response, −4.84% and −5.97% | `docs/assets/finance/measure_market.py` |
| 89 | `docs/finance/surveillance.md`:52 | - **April 9th, 2025:** Operational pause rebound, +9.52% | `docs/assets/finance/measure_market.py` |
| 90 | `docs/finance/surveillance.md`:76 | - **Why it matters:** Payment and market fraud cost over **45 billion USD** in 2022 (Huang et al., 2025). On these five years a fi | `docs/assets/finance/measure_market.py` |
| 91 | `docs/finance/unsupervised_fraud.md`:26 | *Higher metric values denote superior ranking accuracy in isolating fraud instances. The dashed baseline represents a random class | `docs/assets/finance/measure_market.py` |
| 92 | `docs/finance/unsupervised_fraud.md`:54 | / ROC-AUC, LODA / 0.966 / | `docs/assets/finance/measure_market.py` |
| 93 | `docs/finance/unsupervised_fraud.md`:55 | / ROC-AUC, Half-Space Trees / 0.935 / | `docs/assets/finance/measure_market.py` |
| 94 | `docs/finance/unsupervised_fraud.md`:56 | / ROC-AUC, Mahalanobis / 0.957 / | `docs/assets/finance/measure_market.py` |
| 95 | `docs/finance/unsupervised_fraud.md`:57 | / ROC-AUC, ensemble / **0.959** / | `docs/assets/finance/measure_market.py` |
| 96 | `docs/finance/unsupervised_fraud.md`:59 | / Real frauds among the alerts / 33 (precision 39.8%) / | `docs/assets/finance/measure_market.py` |
| 97 | `docs/finance/unsupervised_fraud.md`:60 | / Recall on frauds / 32.4% / | `docs/assets/finance/measure_market.py` |
| 98 | `docs/finance/unsupervised_fraud.md`:61 | / Training time (Kaggle CPU, no labels) / 28 s / | `docs/assets/finance/measure_market.py` |
| 99 | `docs/finance/unsupervised_fraud.md`:65 | For comparative context, a fully supervised detection engine—which requires manual classification labels for every single historic | `docs/assets/finance/measure_market.py` |
| 100 | `docs/finance/unsupervised_fraud.md`:67 | On ranking quality the unsupervised ensemble reaches **99% of the supervised ROC-AUC** (0.959 against 0.969). At the 0.99 cutoff i | `docs/assets/finance/measure_market.py` |
| 101 | `docs/forest/index.md`:3 | A single decision tree relies on a single rule, which breaks the moment a robot's conditions change. A forest combines multiple tr | `benchmarks/bricks_quality.py` |
| 102 | `docs/forest/index.md`:16 | Every tree sees every sample, but with a random weight: a sample counts as many times as a Poisson draw with mean 6, so no two tre | `benchmarks/bricks_quality.py` |
| 103 | `docs/forest/index.md`:23 | f = AdaptiveRandomForestClassifier(n_models=5, seed=1) | `benchmarks/bricks_quality.py` |
| 104 | `docs/forest/index.md`:26 | y = int(x > 2.5) if i < 750 else int(x <= 2.5) | `benchmarks/bricks_quality.py` |
| 105 | `docs/forest/index.md`:37 | The stream gives class 0 to small `x` for the first 750 samples and then swaps the two labels. After the swap the forest answers 1 | `benchmarks/bricks_quality.py` |
| 106 | `docs/forest/index.md`:41 | A longer test makes the effect visible. `DriftStream` produces four numbers per sample and changes its rule at sample 1000. The fi | `benchmarks/bricks_quality.py` |
| 107 | `docs/forest/index.md`:53 | The forest costs about 4.3 ms per sample for ten trees against 0.24 ms for the single tree, roughly eighteen times more. | `benchmarks/bricks_quality.py` |
| 108 | `docs/forest/index.md`:66 | f = AggregatedMondrianForestRegressor(n_trees=10, seed=0) | `benchmarks/bricks_quality.py` |
| 109 | `docs/forest/index.md`:77 | The target is -1 on the left half and +1 on the right half; the forest answers 0.99 and -0.96 at `x = 0.9` and `x = 0.1`. | `benchmarks/bricks_quality.py` |
| 110 | `docs/forest/index.md`:89 | The paper bounds how much worse the aggregate can be than any single pruning. For the quadratic loss and targets in `[-B, B]`, wit | `benchmarks/bricks_quality.py` |
| 111 | `docs/forest/index.md`:103 | first `s = 5` observations per candidate feature, takes the minimum and the maximum of | `benchmarks/bricks_quality.py` |
| 112 | `docs/forest/index.md`:110 | every sample with probability `p = 0.5` (sub-bagging, without replacement). Every tree | `benchmarks/bricks_quality.py` |
| 113 | `docs/forest/index.md`:120 | f = OnlineExtraTreesRegressor(n_models=5, seed=0) | `benchmarks/bricks_quality.py` |
| 114 | `docs/forest/index.md`:166 | The adaptive random forest has a higher median accuracy than the single adaptive tree in the first 100 samples after the change, b | `benchmarks/bricks_quality.py` |
| 115 | `docs/forest/index.md`:168 | On the same stream OXT scores slightly above the plain Mondrian forest in every window and costs about five times more per sample  | `benchmarks/bricks_quality.py` |
| 116 | `docs/forest/index.md`:180 | - **Detector thresholds.** The paper compares a slow setting (`delta_w = 1e-4`, `delta_d = 1e-5`) and a fast one (`delta_w = 0.01` | `benchmarks/bricks_quality.py` |
| 117 | `docs/forest/index.md`:184 | - **Parameter names of the aggregated forest.** `step` is the learning rate `eta` of the paper and `dirichlet` is the parameter `a | `benchmarks/bricks_quality.py` |
| 118 | `docs/forest/index.md`:199 | - On a noiseless step function the aggregated prediction with `step = 1e4` equals the leaf prediction. On a noisy curve the two di | `benchmarks/bricks_quality.py` |
| 119 | `docs/forest/index.md`:205 | p99 of one `learn_one` plus one `predict_one` over 3000 samples with four features, on a Kaggle CPU: 9.4 ms for the adaptive rando | `benchmarks/bricks_quality.py` |
| 120 | `docs/forest/index.md`:213 | The first test of the adaptive forest expected a drift after a plain label swap. None came: the trees relearn from the weighted sa | `benchmarks/bricks_quality.py` |
| 121 | `docs/learn/calibration.md`:3 | A model can be great at ranking outcomes but still report the wrong confidence numbers—stating it is 99% sure when it is only righ | `code block on the page` |
| 122 | `docs/learn/calibration.md`:7 | A decision tree that has 80 % accuracy on some region will happily output 0.99 on every | `code block on the page` |
| 123 | `docs/learn/calibration.md`:9 | control policy reads those numbers: a 0.99 that means "80 %" makes the policy act far | `code block on the page` |
| 124 | `docs/learn/calibration.md`:30 | - `σ(z) = 1 / (1 + e^{−z})` — the sigmoid, from `ℝ` back to `(0, 1)`. | `code block on the page` |
| 125 | `docs/learn/calibration.md`:39 | A ← A + g gᵀ                 (start A = ρ I, ρ = 100) | `code block on the page` |
| 126 | `docs/learn/calibration.md`:40 | θ ← Proj(θ − A⁻¹ g / γ)      (γ = 0.1, projection onto ‖θ‖ ≤ 100) | `code block on the page` |
| 127 | `docs/learn/calibration.md`:50 | Base model says `p = 0.8` on a sample whose true label is `y = 0`. Start | `code block on the page` |
| 128 | `docs/learn/calibration.md`:55 | σ(θ · u) = σ(1.3863) = 0.8 | `code block on the page` |
| 129 | `docs/learn/calibration.md`:59 | Assume `A = 100 · I` (start of the run, `A` has not accumulated much yet). Then: | `code block on the page` |
| 130 | `docs/learn/calibration.md`:69 | σ(0.889 · 1.3863 + (−0.08)) = σ(1.152) = 0.760 | `code block on the page` |
| 131 | `docs/learn/calibration.md`:112 | / Newton step / scaled by `1/gamma`, `gamma = 0.1` / `gamma` / | `code block on the page` |
| 132 | `docs/learn/calibration.md`:113 | / Initial Hessian / `rho · I`, `rho = 100` / `rho` / | `code block on the page` |
| 133 | `docs/learn/metric_learning.md`:74 | Start `W = I`, `C = 0.1`, `xa = (1, 0)`, `xp = (1, 0.2)`, `xn = (0.8, 0)`. | `code block on the page` |
| 134 | `docs/learn/metric_learning.md`:77 | s(xa, xp) = xaᵀ W xp = 1 | `code block on the page` |
| 135 | `docs/learn/metric_learning.md`:78 | s(xa, xn) = xaᵀ W xn = 0.8 | `code block on the page` |
| 136 | `docs/learn/metric_learning.md`:79 | ℓ = max(0, 1 − 1 + 0.8) = 0.8 | `code block on the page` |
| 137 | `docs/learn/metric_learning.md`:82 | ‖V‖² = 0.08 | `code block on the page` |
| 138 | `docs/learn/metric_learning.md`:83 | τ = min(0.1, 0.8 / 0.08) = min(0.1, 10) = 0.1 | `code block on the page` |
| 139 | `docs/learn/metric_learning.md`:93 | learner = OASIS(C=0.1) | `code block on the page` |
| 140 | `docs/learn/metric_learning.md`:124 | Start `A = I`, `z = (1, 0)`, target `y = 0.5`, learning rate `η = 0.1`. | `code block on the page` |
| 141 | `docs/learn/metric_learning.md`:127 | ŷ = 1 | `code block on the page` |
| 142 | `docs/learn/metric_learning.md`:128 | a = 0.1 · 0.5 · 1 − 1 = −0.95 | `code block on the page` |
| 143 | `docs/learn/metric_learning.md`:129 | ȳ = (−0.95 + √(0.9025 + 0.4)) / 0.2 = 0.9564 | `code block on the page` |
| 144 | `docs/learn/metric_learning.md`:144 | learner = LEGO(eta=0.1) | `code block on the page` |
| 145 | `docs/learn/metric_learning.md`:178 | Start `A = I`, `b = 1`, `z = (1, 1)`, label `y = +1`. | `code block on the page` |
| 146 | `docs/learn/metric_learning.md`:181 | d = [1, 1] · I · [1, 1]ᵀ = 2 | `code block on the page` |
| 147 | `docs/learn/metric_learning.md`:182 | ℓ = max(0, 1 · (2 − 1) + 1) = 2 | `code block on the page` |
| 148 | `docs/learn/metric_learning.md`:183 | ‖z‖⁴ = (√2)⁴ = 4 | `code block on the page` |
| 149 | `docs/learn/metric_learning.md`:184 | α = 2 / (1 + 4) = 0.4 | `code block on the page` |
| 150 | `docs/learn/metric_learning.md`:186 | b ← 1 + 0.4 = 1.4 | `code block on the page` |
| 151 | `docs/learn/metric_learning.md`:219 | knn = MetricKNNClassifier(OASIS(C=0.1), n_neighbors=1, window_size=1000) | `code block on the page` |
| 152 | `docs/learn/metric_learning.md`:220 | hits = 0 | `code block on the page` |
| 153 | `docs/learn/metric_learning.md`:232 | classifies, and gets 93.75 % of the 400 samples right, each predicted before it is learned. | `code block on the page` |
| 154 | `docs/learn/metric_learning.md`:246 | `progressive_val_score` (predict, then learn), `StandardScaler` in front, k = 5, | `code block on the page` |
| 155 | `docs/learn/metric_learning.md`:247 | `window_size=1000`, metric learned online with the default parameters: | `code block on the page` |
| 156 | `docs/learn/online_classifiers.md`:43 | v = 2: n = 1, m = 2, M2 = 0 | `code block on the page` |
| 157 | `docs/learn/online_classifiers.md`:44 | v = 4: n = 2, m = 3, M2 = 2 | `code block on the page` |
| 158 | `docs/learn/online_classifiers.md`:45 | v = 6: n = 3, m = 4, M2 = 8 | `code block on the page` |
| 159 | `docs/learn/online_classifiers.md`:48 | Mean 4, variance `8 / 3 = 2.667`. | `code block on the page` |
| 160 | `docs/learn/online_classifiers.md`:56 | With `alpha = 0`, the statistics equal the batch MLE of a Gaussian NB on the observed | `code block on the page` |
| 161 | `docs/learn/online_classifiers.md`:57 | samples; the wrapper matches a batch Gaussian NB to numerical precision (max difference | `code block on the page` |
| 162 | `docs/learn/online_classifiers.md`:72 | `X[3000:][s[3000:] == 1] += [-2, -2, 2, 2]` | `code block on the page` |
| 163 | `docs/learn/online_classifiers.md`:83 | Prequential accuracy (predict first, then learn) before the change, right after it, and late. | `code block on the page` |
| 164 | `docs/learn/online_classifiers.md`:89 | wrong. Prequential accuracy over `[500, 3000)` (pre), `[3000, 3200)` (post), and | `code block on the page` |
| 165 | `docs/learn/online_classifiers.md`:94 | / `OnlineGaussianNB(alpha=0)` / 0.952 / 0.480 / 0.879 / 617 / | `code block on the page` |
| 166 | `docs/learn/online_classifiers.md`:95 | / `OnlineGaussianNB(alpha=0.01)` / 0.952 / 0.790 / 0.941 / 59 / | `code block on the page` |
| 167 | `docs/learn/online_classifiers.md`:96 | / `OnlineGaussianNB(alpha=0.1)` / 0.950 / 0.925 / 0.913 / 0 / | `code block on the page` |
| 168 | `docs/learn/online_classifiers.md`:97 | / `OnlineSoftmaxRegression(eta=0.1, l2=1e-4)` / 0.934 / 0.565 / 0.919 / 213 / | `code block on the page` |
| 169 | `docs/learn/online_classifiers.md`:98 | / `DriftAdaptiveClassifier(OnlineGaussianNB(alpha=0))` / 0.952 / 0.480 / 0.931 / 145 / | `code block on the page` |
| 170 | `docs/learn/online_classifiers.md`:101 | "Recovery" is the number of samples after 3000 until the 200-sample accuracy is back | `code block on the page` |
| 171 | `docs/learn/online_classifiers.md`:106 | fast the world changes — `alpha = 0.1` never loses accuracy, at the cost of a noisier | `code block on the page` |
| 172 | `docs/learn/online_classifiers.md`:121 | 5. When the background's accuracy over the last 50 samples beats the main classifier's | `code block on the page` |
| 173 | `docs/learn/online_classifiers.md`:140 | `X[3000:][s[3000:] == 1] += [-2, -2, 2, 2]` | `code block on the page` |
| 174 | `docs/learn/online_classifiers.md`:141 | det = CUSUMDriftDetector(reference="fixed", radius=20, ref_mult=5, two_sided=False) | `code block on the page` |
| 175 | `docs/learn/online_classifiers.md`:142 | clf, hit = DriftAdaptiveClassifier(OnlineGaussianNB(), det, window=50, smooth_window=50), [] | `code block on the page` |
| 176 | `docs/learn/online_classifiers.md`:152 | On the running data the wrapper leaves its pre-drift accuracy untouched (0.952), fires | `code block on the page` |
| 177 | `docs/learn/online_classifiers.md`:155 | `alpha = 0.1` (0.913). | `code block on the page` |
| 178 | `docs/learn/online_classifiers.md`:188 | At 100 Hz the loop has 10 ms per cycle: all three fit with three orders of magnitude | `code block on the page` |
| 179 | `docs/learn/online_dynamics.md`:29 | real = RigidBodyModel(write_minimal_urdf(payload_mass=0.5)) | `code block on the page` |
| 180 | `docs/learn/online_dynamics.md`:42 | Two copies of the same 2-link arm, one with a 0.5 kg payload in the gripper (the "real" robot) | `code block on the page` |
| 181 | `docs/learn/online_dynamics.md`:44 | from the payload-free URDF, predicts torques of 1.422 and 0.478 N·m where the real arm needs 1.496 | `code block on the page` |
| 182 | `docs/learn/online_dynamics.md`:90 | `P₀ = δ · I` and `w₀ = 0`. `λ` is the **forgetting factor**: `λ = 1` weights all past | `code block on the page` |
| 183 | `docs/learn/online_dynamics.md`:99 | Start `δ = 1e6` (so `P₀ = 1e6 · I`, essentially uninformative), `λ = 1`, one sample | `code block on the page` |
| 184 | `docs/learn/online_dynamics.md`:100 | `x = (1, 2)`, `y = 5`: | `code block on the page` |
| 185 | `docs/learn/online_dynamics.md`:103 | P₀ x = 1e6 · (1, 2) = (1e6, 2e6) | `code block on the page` |
| 186 | `docs/learn/online_dynamics.md`:104 | xᵀ P₀ x = 1·1e6 + 2·2e6 = 5e6 | `code block on the page` |
| 187 | `docs/learn/online_dynamics.md`:106 | w₀ = (0, 0), so xᵀ w₀ = 0, and the error is y − 0 = 5 | `code block on the page` |
| 188 | `docs/learn/online_dynamics.md`:113 | rls = RecursiveLeastSquares(lam=1.0, delta=1e6) | `code block on the page` |
| 189 | `docs/learn/online_dynamics.md`:122 | After one sample the weights are exactly `(1, 2)`: `w · x = 1·1 + 2·2 = 5`. That is the | `code block on the page` |
| 190 | `docs/learn/online_dynamics.md`:127 | at `1 / (1 − λ)`. `λ = 0.99` means roughly 100 samples of memory; `λ = 0.9` means | `code block on the page` |
| 191 | `docs/learn/online_dynamics.md`:141 | nom, real = RigidBodyModel(write_minimal_urdf()), RigidBodyModel(write_minimal_urdf(payload_mass=0.5)) | `code block on the page` |
| 192 | `docs/learn/online_dynamics.md`:164 | If the payload changes at sample 200 of 400, a `λ = 1` learner (batch least squares) | `code block on the page` |
| 193 | `docs/learn/online_dynamics.md`:165 | keeps the old weights and gets the new regime wrong. A `λ = 0.99` learner forgets the | `code block on the page` |
| 194 | `docs/learn/online_dynamics.md`:173 | fast = ResidualDynamicsLearner(model, lam=0.99, delta=1e6) | `code block on the page` |
| 195 | `docs/learn/online_dynamics.md`:174 | aware = DriftAwareResidualDynamicsLearner(model, lam=0.99) | `code block on the page` |
| 196 | `docs/learn/online_dynamics.md`:197 | 400, `lam=0.99`): | `code block on the page` |
| 197 | `docs/learn/online_dynamics.md`:203 | - On the CUSUM path the trigger either never fires (`h = 20`, `h = 10`) or fires as a | `code block on the page` |
| 198 | `docs/learn/online_dynamics.md`:204 | false positive (`h = 5`, events at 151 and 207). | `code block on the page` |
| 199 | `docs/learn/online_dynamics.md`:233 | tens of microseconds. For a 2-DoF arm the difference is roughly 1.9 ms vs 2.7 ms per | `code block on the page` |
| 200 | `docs/learn/online_dynamics.md`:239 | - `lam` (default 1.0): forgetting factor in `(0, 1]`. `lam = 1.0` weights all past | `code block on the page` |
| 201 | `docs/linear/glm.md`:23 | y_hat = 1 / (1 + exp(-z))        for LogisticRegression | `code block on the page` |
| 202 | `docs/linear/glm.md`:32 | A small example with `LinearRegression`, `SGD(lr=0.5)`, no intercept, three | `code block on the page` |
| 203 | `docs/linear/glm.md`:33 | samples of the line `y = 3 * x`: | `code block on the page` |
| 204 | `docs/linear/glm.md`:35 | reg = LinearRegression(optimizer=SGD(lr=0.5), intercept=False) | `code block on the page` |
| 205 | `docs/linear/glm.md`:46 | clf = LogisticRegression(optimizer=Adam(lr=0.1), intercept=True) | `code block on the page` |
| 206 | `docs/linear/glm.md`:69 | reg = LinearRegression(optimizer=SGD(lr=0.1), intercept=False, | `code block on the page` |
| 207 | `docs/linear/glm.md`:92 | problem, target weight `w* = 2`, each with `SGD` at a different `lr`: | `code block on the page` |
| 208 | `docs/linear/glm.md`:103 | `lr = 0.5` reaches the target in a single step, because for a one-feature | `code block on the page` |
| 209 | `docs/linear/glm.md`:105 | when `lr = 0.5`. `lr = 2.0` diverges: the same formula has a factor | `code block on the page` |
| 210 | `docs/linear/glm.md`:126 | for opt, lr in [(SGD(lr=1e-3), 1e-3), (AdaGrad(lr=10.0), 10.0), | `code block on the page` |
| 211 | `docs/linear/glm.md`:127 | (Adam(lr=0.5, beta1=0.99), 0.5)]: | `code block on the page` |
| 212 | `docs/linear/glm.md`:129 | period_s=1.0, rate_hz=50.0, n_cycles=2, seed=0) | `code block on the page` |
| 213 | `docs/linear/glm.md`:148 | `beta1=0.99`, `Adam` converges to `-2.273`. The choice of optimizer is | `code block on the page` |
| 214 | `docs/linear/glm.md`:187 | median of 1000 samples after a warm-up of 200 is `0.0139 ms` per sample, | `code block on the page` |
| 215 | `docs/linear/bayes.md`:26 | with `A_0 = alpha * I`, `b_0 = 0`. `alpha` is the prior precision on the | `code block on the page` |
| 216 | `docs/linear/bayes.md`:27 | weights, `beta` the precision of the noise on the targets. The prediction on | `code block on the page` |
| 217 | `docs/linear/bayes.md`:32 | `y = 2 x` and the current `m` is roughly `[2.0]`, `S` is small. On a new | `code block on the page` |
| 218 | `docs/linear/bayes.md`:33 | sample `x = 1.0` the prediction is around `2.0` with a narrow interval. On | `code block on the page` |
| 219 | `docs/linear/bayes.md`:34 | a sample `x = 50.0`, far outside the observed range, the same `m` predicts | `code block on the page` |
| 220 | `docs/linear/bayes.md`:51 | model = BayesianLinearRegression(alpha=1.0, beta=1.0, | `code block on the page` |
| 221 | `docs/linear/bayes.md`:70 | numerical precision (`max/dm/ ~ 1e-16`, `max/dS/ ~ 1e-19` on a 20-sample | `code block on the page` |
| 222 | `docs/linear/bayes.md`:95 | model = BayesianLinearRegression(alpha=1.0, beta=1.0, | `code block on the page` |
| 223 | `docs/linear/bayes.md`:108 | prediction on `x = 3` after five samples with `alpha = beta = 1`. The | `code block on the page` |
| 224 | `docs/linear/bayes.md`:128 | period_s=1.0, rate_hz=50.0, n_cycles=2, seed=0) | `code block on the page` |
| 225 | `docs/linear/bayes.md`:129 | model = BayesianLinearRegression(alpha=1.0, beta=100.0, | `code block on the page` |
| 226 | `docs/linear/bayes.md`:140 | The value `8.041` is the posterior mean on `qd = 0.5`. It is not close to | `code block on the page` |
| 227 | `docs/linear/bayes.md`:141 | the `-2.158` of `SGD(lr=1e-3)` on the same stream, and the gap is not a | `code block on the page` |
| 228 | `docs/linear/bayes.md`:144 | SGD with `lr=1e-3` has not converged after 100 samples: its weight `w = | `code block on the page` |
| 229 | `docs/linear/bayes.md`:149 | term, and it is very close to the target mean `mean(tau_0) = 8.19` of | `code block on the page` |
| 230 | `docs/linear/bayes.md`:193 | the code symmetrises it: `S = 0.5 (S + S^T)`. Without this, numerical | `code block on the page` |
| 231 | `docs/linear/bayes.md`:209 | problem with `d = 100`, the update is the most expensive of the models in | `code block on the page` |
| 232 | `docs/linear/bayes.md`:211 | `0.0178 ms` per sample, close to AdaGrad. | `code block on the page` |
| 233 | `docs/linear/payload.md`:43 | factor, in `(0, 1]`. `lam = 1` treats all past samples with the same | `code block on the page` |
| 234 | `docs/linear/payload.md`:46 | so `lam = 0.99` means a memory of about 100 samples (2 s at 50 Hz); the nine | `code block on the page` |
| 235 | `docs/linear/payload.md`:65 | est = PayloadEstimator(model, link="panda_link7", lam=0.99) | `code block on the page` |
| 236 | `docs/linear/payload.md`:68 | period_s=4.0, rate_hz=50.0, n_cycles=1, | `code block on the page` |
| 237 | `docs/linear/payload.md`:69 | fault_at_s=2.0, fault="payload", payload_mass=5.0, | `code block on the page` |
| 238 | `docs/linear/payload.md`:70 | payload_link="panda_link7", seed=0) | `code block on the page` |
| 239 | `docs/linear/payload.md`:98 | of `5.0 kg` appearing at `t = 2.0 s`: | `code block on the page` |
| 240 | `docs/linear/payload.md`:102 | With `lam = 0.99` the estimate is `2.88 kg` at `t = 3.5` and enters the band | `code block on the page` |
| 241 | `docs/linear/payload.md`:103 | of 10 % around `5.0 kg` `4.26 s` after the fault (it is `4.87 kg` at the end | `code block on the page` |
| 242 | `docs/linear/payload.md`:104 | of the 8 s stream). With `lam = 0.95` (a memory of about 20 samples) it enters | `code block on the page` |
| 243 | `docs/linear/payload.md`:105 | the band `0.96 s` after the fault and ends at `5.00 kg`; with `lam = 0.999` it | `code block on the page` |
| 244 | `docs/linear/payload.md`:106 | has not entered it at the end (`3.94 kg`). Before the fault the largest value | `code block on the page` |
| 245 | `docs/linear/payload.md`:107 | is `0.054 kg`, and at `t = 1.5` it is `0.0000 kg`, which is the correct | `code block on the page` |
| 246 | `docs/linear/payload.md`:111 | appears, disappears, or changes. Without it, `lam = 1`, the estimate | `code block on the page` |
| 247 | `docs/linear/payload.md`:118 | 50 Hz. The payload appears at `t = 2.0 s` as a fault injected by the | `code block on the page` |
| 248 | `docs/linear/payload.md`:119 | dataset, with mass `5.0 kg` on `panda_link7`. The estimator follows: | `code block on the page` |
| 249 | `docs/linear/payload.md`:131 | ds = SyntheticArm(URDF, period_s=4.0, rate_hz=50.0, n_cycles=2, | `code block on the page` |
| 250 | `docs/linear/payload.md`:132 | fault_at_s=2.0, fault="payload", payload_mass=5.0, | `code block on the page` |
| 251 | `docs/linear/payload.md`:133 | payload_link="panda_link7", seed=0) | `code block on the page` |
| 252 | `docs/linear/payload.md`:134 | est = PayloadEstimator(model, link="panda_link7", lam=0.99) | `code block on the page` |
| 253 | `docs/linear/payload.md`:146 | / `0.50` / `+0.0000 kg` / | `code block on the page` |
| 254 | `docs/linear/payload.md`:147 | / `1.50` / `+0.0000 kg` / | `code block on the page` |
| 255 | `docs/linear/payload.md`:148 | / `2.50` / `+1.4057 kg` / | `code block on the page` |
| 256 | `docs/linear/payload.md`:149 | / `3.50` / `+2.8762 kg` / | `code block on the page` |
| 257 | `docs/linear/payload.md`:150 | / `5.00` / `+3.8170 kg` / | `code block on the page` |
| 258 | `docs/linear/payload.md`:151 | / `7.00` / `+4.7582 kg` / | `code block on the page` |
| 259 | `docs/linear/payload.md`:155 | estimate enters the band of 10 % around the true mass `5.0 kg` `4.26 s` after | `code block on the page` |
| 260 | `docs/linear/payload.md`:218 | section 3 it enters the 10 % band `4.26 s` after the fault, while | `code block on the page` |
| 261 | `docs/linear/payload.md`:219 | `lam = 0.95` takes `0.96 s`. | `code block on the page` |
| 262 | `docs/linear/payload.md`:229 | which made the forgetting `lam^9` per sample (`0.91` for `lam = 0.99`, a | `code block on the page` |
| 263 | `docs/linear/payload.md`:232 | `dt <= 0`, a sample with a NaN in `q`, `qd` or `tau`, and a sample without | `code block on the page` |
| 264 | `docs/linear/payload.md`:240 | the full stream of `n_cycles=2` (400 samples) takes of the order of a | `code block on the page` |
| 265 | `docs/linear/payload.md`:255 | a single run of `make_figure_payload.py` with `n_cycles=1` after an | `code block on the page` |
| 266 | `docs/linear/payload.md`:256 | earlier attempt with three forgetting factors on `n_cycles=2` took more | `code block on the page` |
| 267 | `docs/getting-started.md`:22 | ds = DriftStream(n_samples=2000, n_features=5, n_drifts=1, kind="sudden", seed=0) | `code block on the page` |
| 268 | `docs/getting-started.md`:34 | Accuracy(n=1999, value=0.9989994997498749) | `code block on the page` |
| 269 | `docs/getting-started.md`:37 | `DriftStream` makes 2,000 samples with 5 features and one sudden change of the rule that links them to the label. The model predic | `code block on the page` |
| 270 | `docs/getting-started.md`:94 | pulito, K, anomalie = Armatura(livello_ia=0.0).analizza(serie) | `code block on the page` |
| 271 | `docs/getting-started.md`:104 | The names are Italian: `analizza` is "analyse", `livello_ia=0.0` means the shield filters actively (1.0 only marks, and the 9999 s | `code block on the page` |
| 272 | `docs/getting-started.md`:114 | raw[50] = 1e6 | `code block on the page` |
| 273 | `docs/getting-started.md`:115 | safe = rate_limited_follower(raw, max_vel=2.0, max_accel=1.0, dt=0.01) | `code block on the page` |
| 274 | `docs/getting-started.md`:140 | Every printed result was produced by running the code above with `numpy.random.default_rng(0)` or `DriftStream(seed=0)`. The earli | `code block on the page` |
