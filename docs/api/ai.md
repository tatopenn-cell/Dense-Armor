# API — IA

The Dense-Armor APIs, read from the AI side. A logical flow from raw
input (images, text, embeddings) through feature extraction, learned
representations and models, to the decision and protection layer.

## IA map

Rotate with the mouse, hover a node to see its level and the guide.

<iframe src="../assets/api/ai_3d.html" style="width:100%;height:700px;border:0" loading="lazy"></iframe>

*The levels are: input (raw data), feature (perception), repr
(compression), model (learners), decision (gate + protection).*

## APIs by level

### Input

- [`frame`](../api/general.md)
- [`text`](../api/general.md)
- [`embedding`](../api/general.md)

### Feature extraction

- [`FrameFeatures`](../api/general.md)
- [`PatchFeatures`](../api/general.md)
- [`Tokenizer`](../api/general.md)
- [`TFIDF`](../api/general.md)
- [`FeatureHasher`](../api/general.md)

### Representation

- [`RandomProjection`](../api/general.md)
- [`IncrementalPCA`](../api/general.md)
- [`BagOfVisualWords`](../api/general.md)

### Models

- [`HoeffdingTreeClassifier`](../api/general.md)
- [`OnlineRobustMahalanobis`](../anomaly/mahalanobis.md)
- [`PatchMemory`](../api/general.md)
- [`OnlineIsolationForest`](../anomaly/oiforest.md)
- [`LLMEmbeddingClassifier`](../api/general.md)

### Decision & protection

- [`AnomalyGate`](../api/general.md)
- [`SafeEstimator`](../api/robot_roles.md)
- [`schema`](../api/general.md)
- [`call_json`](../api/general.md)
