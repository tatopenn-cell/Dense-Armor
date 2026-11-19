# Preprocessing

Raw sensor numbers come in different sizes and units. These tools put them on a common scale and turn them into the numbers that matter (such as speed or power), one reading at a time, so that a model can learn from them.

## The map of preprocessing

Click a page to open it, or the job on the left (or the button at the top) to go back to the [map of the library](../index.md#the-library-folder-by-folder).

<object type="image/svg+xml" data="../assets/maps/preprocessing.svg" style="width:100%"></object>

Raw sensor numbers come in different units and sizes — a joint position in radians, a torque in
newton-metres, a word in a sentence. Before a model can learn from them they are put on a common
scale, turned into the quantities that matter (speed, power, word weights), filtered, and
balanced. Everything here works one sample at a time: the estimator sees a sample, updates its
own state, and is ready for the next one, without keeping the whole history in memory.

Nine steps follow. The first four are about numbers, the next three about text, the last two
about which features to keep and how to handle rare events.

- [Scaling features and joints](scaling.md) (steps 1-3)
- [Speed, acceleration, jerk and power](signals.md) (steps 4-5)
- [Text: words, counts, weights and hashing](text.md) (steps 6-7)
- [Choosing features and rare events](features.md) (steps 8-9)

## Classes documented on other pages

These classes of the folder have no guide page of this group; they are in the [API reference](#api-reference) below.

- `EWScaler`: per-feature standardisation with exponentially weighted statistics.
- `MinMaxScaler`: rescales each feature to `[0, 1]` with the running minimum and maximum.

## API reference

::: dense_armor.core.signals.preprocessing.scale
::: dense_armor.core.signals.preprocessing.joints
::: dense_armor.core.signals.preprocessing.text
::: dense_armor.core.signals.preprocessing.select
::: dense_armor.core.signals.preprocessing.imbalance

---

## Details

- Scalers keep one statistic per feature and per joint; a feature never seen, or with zero
  spread, passes unchanged.
- `TFIDF(smooth_idf=True)` uses $\ln(1 + N/\text{DF})$; this smoothing is not in Silajev
  (2026) and is off by default.
- `JointDerivatives` keeps the last `order + 1` valid samples; missing samples (NaN) are
  skipped and counted in `n_missing`.
- Sources: Silajev, I. (2026), "TF-IDF and BM25 are exact KL divergences", arXiv:2609.14016.
  Weinberger, K. et al. (2009), "Feature hashing for large scale multitask learning", ICML,
  arXiv:0902.2206. Malialis, K., Panayiotou, C., Polycarpou, M. (2018), "Queue-based
  resampling for online class imbalance learning", ICANN, arXiv:1809.10388.
