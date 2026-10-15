# Intrusion detection: drift-adaptive streaming tree architectures for real-time network security

A data center handles millions of network packets every day, but cybercriminals continuously change the shape of their attacks to bypass traditional security. Instead of using a static filter that becomes blind to new threats over time, this tool learns from every single packet in real time, sample by sample, deploying embedded drift detectors to dynamically grow new classification branches as soon as a new mutation in traffic behavior is identified.


## The scene

When an institutional data center undergoes a cyberattack, adversaries continuously alter payload signatures and traffic configurations to systematically bypass legacy intrusion detection barriers. A conventional classification engine trained on static historical data remains functionally blind to subsequent, mutated adversarial variants. 

To maintain perimeter integrity, network security systems must process incoming streams dynamically, adapting to shifting malicious footprints without relying on full model retraining intervals.

To benchmark this resilience on a public dataset, the online dual-tree pipeline was deployed on **125,973 labelled network flows** from the industry-standard NSL-KDD dataset. Malicious traffic accounts for 46% of the continuous stream, establishing a balanced statistical framework that demands genuine multidimensional separation between benign operations and active exploit variants. The training stream took 65 seconds on a Kaggle CPU.


## Core Capabilities

| Tool | Strategic Function | Operational Impact |
|---|---|---|
| **HoeffdingAdaptiveTree** | Dynamic structural adaptation | Deploys embedded statistical drift detectors across all internal nodes, automatically spawning and swapping subtrees to intercept mutated attack variants. |
| **HoeffdingTree** | High-speed baseline classification | Enforces rigorous statistical bounds to lock in stable execution splits, providing a robust, high-throughput reference layer for structural comparison. |


![Intrusion detection on 25,195 test network flows: 10,954 of 11,704 attacks caught, 10,954 of 12,875 alerts real](../assets/finance/intrusion.png)

*Green: right. Red: wrong. Top bar: what happened to the 11,704 attacks. Bottom bar: what the 12,875 alerts were.*


## 1. Dynamic structural branch adaptation (HoeffdingAdaptiveTree)
Standard incremental decision trees lock in division attributes permanently, degrading rapidly when traffic distributions drift. The adaptive architecture integrates an active ADWIN detector within every internal node. Upon detecting statistical variance shifts in recent traffic inputs, the node initiates a parallel background subtree. If this new structural branch demonstrates superior predictive precision on current operational data, it seamlessly replaces the obsolete branch while preserving the rest of the established tree topology.


## 2. High-throughput statistical reference (HoeffdingTree)
Operating as the stable reference layer of the ensemble, the classic Hoeffding tree utilizes strict mathematical bounds to control structural growth. Attribute splits are validated and executed only when the leading information gain parameter outperforms the secondary attribute by a margin exceeding the formal Hoeffding probability bound. This keeps the splits to the ones the data supports and protects the pipeline against localized noise.


## Execution Workflow

For every incoming network packet, the streaming security pipeline executes a parallel verification sequence:

1. **Parallel Stream Ingestion:** The 41 quantitative network features of the packet are routed simultaneously to both decision layers.
2. **Multi-Model Anomaly Scoring:** **HoeffdingAdaptiveTree** generates a risk score adapted to current traffic regimes, while **HoeffdingTree** outputs an independent baseline evaluation.
3. **Consensus Voting Optimization:** The final threat probability is computed as the arithmetic mean of both scores. Vetting boundaries exceeding the 0.5 classification threshold raise an alert.


## Empirical Performance

The unified ensemble was rigorously evaluated against a chronological testing stream of 25,195 network flows containing 11,704 verified security breaches.

| Metric | Value |
|---|---|
| ROC-AUC, adaptive tree | 0.9553 |
| ROC-AUC, base tree | 0.9503 |
| ROC-AUC, ensemble | **0.9755** |
| Threshold | 0.5 |
| Attacks predicted (out of 25,195) | 12,875 |
| True positives | 10,954 |
| False positives | 1,921 |
| False negatives | 750 |
| Recall on attacks | 93.6% |
| Precision | 85.1% |
| Training time (Kaggle CPU) | 65 s |

### Strategic Business Takeaway

Relying on standalone detection models creates an operational trade-off between false alarms and missed exploits. By orchestrating a parallel voting consensus, this pipeline achieves a **ROC-AUC of 0.9755**, outperforming either constituent architecture. 

For enterprise infrastructure, this translates into a **93.6% Recall rate**, so 93 out of every 100 malicious flows are caught, while the **85.1% Precision** means that 15 alerts out of 100 are false (1,921 of 12,875).


## Target Enterprise Use Cases

This streaming security architecture is optimized for high-volume network environments:

- **Live Security Operations Centers (SOC):** Provides continuous, packet-by-packet traffic profiling for live network operations monitoring high-throughput infrastructure.
- **Financial & Payment Processing Perimeters:** Scores each flow that reaches a banking or payment gateway.
- **Changing attack patterns:** The adaptive tree replaces a branch when the traffic drifts, with no manual rule updates.


## Details

### Implementation

The pipeline is in `pipelines/finance/intrusion.py`.
Primitives chained, each with its own page:

- [Hoeffding Adaptive Tree](../tree/index.md#3-following-a-sudden-change-the-adaptive-tree)
- [Hoeffding Tree](../tree/index.md#1-a-tree-that-learns-a-robot-state)

### References

- Domingos, P., Hulten, G. (2000). *Mining High-Speed Data Streams.* KDD.
- Bifet, A., Gavalda, R. (2009). *Adaptive Learning from Evolving Data Streams.* IDA.
- Tavallaee, M., Bagheri, E., Lu, W., Ghorbani, A. (2009). *A Detailed Analysis of the KDD CUP 99 Data Set.* IEEE CISDA. NSL-KDD is the cleaned version.
- Dataset: NSL-KDD, KDDTrain+ file. 125,973 labelled network flows.

*The numbers on this page come from a single run on historical data. Not a forecast, not financial advice.*
