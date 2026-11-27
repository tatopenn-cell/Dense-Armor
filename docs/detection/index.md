# Detection

## The map of detection

Click an area to open its page, or one of its pages on the right, or the button at the top to go back to the [map of the library](../index.md#the-library-folder-by-folder).

<object type="image/svg+xml" data="../assets/maps/detection.svg" style="width:100%"></object>

What is strange, what has changed, what to keep out of the model.

## The three branches

| Branch | The question | Classes |
|---|---|---|
| [Anomaly detection](../anomaly/index.md) | Each tool learns from the recent readings what is usual, and gives every new reading a score: the higher the score, the less it looks like the recent past. | `streamDD`, `MultistreamDD`, `streamDS`, `HampelScorer`, `TukeyScorer`, `ChauvenetScorer`, `SigmaClipScorer`, `HampelFilter`, `OnlineRobustMahalanobis`, `LODA`, `HalfSpaceTrees`, `OneClassSGD`, `OnlineIsolationForest`, `LocalOutlierFactor` |
| [Drift detection](../drift/index.md) | Anomaly detection flags samples. Drift detection flags a change of the average: something that happens over many samples, each of which looks plausible on its own. | `ADWIN`, `BayesianOnlineChangepoint`, `CUSUMDriftDetector`, `KSWIN`, `PageHinkley` |
| [Protection](../protect/index.md) | Once a sample looks wrong, what do you do with it? | `Orca`, `StreamingArbiter`, `StreamingHealing` |

## The pages of each branch

### Anomaly detection

| Page | What it shows |
|---|---|
| [Streaming](../anomaly/streaming/index.md) | per-sample deviation scores, single and multi-channel, plus causal Hampel, Tukey, Chauvenet, sigma clipping and online robust Mahalanobis |
| [Robust filters](../anomaly/robust_filters.md) | the four classic detectors on whole series, and the pressure-valve orchestrator |
| [Curvature](../anomaly/curvature.md) | a bounded proximity-to-reference score in [0, 1) |
| [Online Robust Mahalanobis](../anomaly/mahalanobis.md) | distance from a robust median under the MCM |
| [LODA](../anomaly/loda.md) | rare bins in random 1D projections |
| [Half-Space Trees](../anomaly/hst.md) | few reference samples in the region |
| [One-Class SGD (SONAR)](../anomaly/ocsvm.md) | distance from a learned kernel boundary |
| [Online Isolation Forest](../anomaly/oiforest.md) | depth to isolate a point |
| [Local Outlier Factor](../anomaly/lof.md) | density of the point against its nearest neighbours |
| [Offline algorithms](../anomaly/offline.md) | the offline side: an anomaly detector marks the values that do not look like their neighbours |

### Drift detection

| Page | What it shows |
|---|---|
| [CUSUM and ARL theory](../drift/cusum/index.md) | how to detect slow, gradual changes, like a mechanical part wearing out bit by bit: CUSUM accumulates tiny anomalies and triggers an alert when the total deviation confirms a real trend |
| [Offline algorithm](../drift/offline.md) | the offline side: a change detector reads a series of values and says where its level shifts |
| [Drift detectors](../drift/detectors/index.md) | how a system notices slow, gradual changes before an outright failure: four tools that pinpoint when a shift has happened, each trading off reaction delay against false alarms |

### Protection

| Page | What it shows |
|---|---|
| [Arbiter](../protect/arbiter.md) | per-point routing: it decides, reading by reading, whether a glitch should be replaced or a real change kept, and treats it accordingly |
| [Shields](../protect/shields.md) | a shield cleans a series of numbers or a model |
| [Orca (online)](../protect/orca.md) | protects a whole model: it cleans the input, runs the model, and checks the answer |
| [Adaptive engine (online)](../protect/engine.md) | follows a signal that drifts slowly, so that a slow drift is not mistaken for a glitch |
| [Armor (offline)](../protect/armor.md) | cleans one series of numbers: it finds the readings that are obviously broken and replaces them |
| [Hybrid engine (offline)](../protect/hybrid_engine.md) | the part inside Armor that decides whether a value is genuine or broken |

## Details

- Detection has an online side, in the pages of the three branches, and an offline side, in the tab Machine learning, whose classes read a whole table at once; the two offline pages are in their branch, with the pages that were in the tab Machine learning.

- The map is drawn by `docs/assets/maps/make_detection_map.py` from the menu of `mkdocs.yml` and from the catalog of the library, so a page added to one of the three branches appears in it when the script is run again.
