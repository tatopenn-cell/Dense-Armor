# API — IA

**In plain words:** A picture map of the library seen from the AI side, from raw inputs such as images and text to the final decision. Click a part of the picture to read what it does and which parts of the library are involved.

The Dense-Armor APIs, read from the AI side. A logical flow from raw
input (images, text, embeddings) through feature extraction, learned
representations and models, to the decision and protection layer.

## Interactive map

Click a zone of the image to read what it does and which APIs concern it. The side panel does not overlap the image: it updates on each click.

<iframe src="../../assets/api/ai_interactive.html" style="width:100%;height:640px;border:0" loading="lazy"></iframe>


## IA map

Rotate with the mouse, hover a node to see its level and the guide.

<iframe src="../../assets/api/ai_3d.html" style="width:100%;height:640px;border:0" loading="lazy"></iframe>


**What this model shows.** The solid below is a reconstruction of the same
concept head that opens this page, generated from the 1024x1024 source image.
Every pixel of the source has been lifted to a height and the surface has been
closed into a volume. The result can be rotated and inspected from any angle,
including the back, which does not exist in the flat image above.

**How it was made.** The geometry was reconstructed from a single image by
Meshy AI, a feed-forward image-to-3D model. The mesh has roughly 107 500
vertices and 215 800 triangles, about the density of a game-ready hero asset.
The reconstruction is monochrome: the source image was not projected onto the
surface, so the model is shown as a single material rather than a textured
one. This is deliberate. The coloured map at the top of this page keeps the
colours and the API hotspots; the 3D model keeps the shape and the volume.

**How to read it.** The colour of the surface carries no information. What
matters is the silhouette: where the surface is convex, where it is concave,
how thick the neck is, how far the eyes are recessed. Compare this solid
against the flat image at the top of the page: the same shape, but the
volume is now visible.

**How to interact.**
- Drag with the mouse to rotate.
- Use the arrow keys (left, right, up, down) to rotate.
- Use + and - to zoom in and out.
- Auto-rotation is on until you interact.

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
