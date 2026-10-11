# API — IA

The Dense-Armor APIs, read from the AI side. A logical flow from raw
input (images, text, embeddings) through feature extraction, learned
representations and models, to the decision and protection layer.

## Interactive map

Clicca una zona dell'immagine per leggere cosa fa e quali API la riguardano. Il pannello a destra non copre l'immagine: si aggiorna a ogni click.

<iframe src="../assets/api/ai_interactive.html" style="width:100%;height:820px;border:0" loading="lazy"></iframe>


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

---

## Details

<sub>

**Note on the concept image.**
The prompt that generated the concept image on this page was built by a
pipeline of proprietary algorithms written by the author. The rendering
step called the Gemini image-generation API from inside the author's own
application. The stored file declares the asset as
`TrainedAlgorithmicMedia` in its IPTC metadata; it carries no C2PA
manifest and no detectable SynthID watermark in this version, most
likely because the image passed through an intermediate processing step
(resize, re-encode) inside the app before being saved here. The
attribution stays at the level of what the file can prove, not at the
level of what the author remembers.

</sub>
