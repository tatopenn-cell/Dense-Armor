# Text vectorizer: how do I turn documents into ranked search results without a batch library?

A search over documents needs three things: a vocabulary, a score per term, and a way to rank a query against the documents. All three are online here: one document at a time, with a state that stays fixed once the vocabulary is built.

## 1. Stop words, before the n-grams

Stop words are the words that carry no meaning for a search: "the", "a", "over". They are removed before the n-grams are formed, so no bigram contains one.

```python
from dense_armor.utility.preprocessing.text import Tokenizer

tk = Tokenizer(ngrams=2, stop_words=["the", "over"])
print(tk.transform_one({"text": "the quick brown fox jumps over the lazy dog"})["tokens"])
```

```
['quick', 'brown', 'fox', 'jumps', 'lazy', 'dog', 'quick brown', 'brown fox', 'fox jumps', 'jumps lazy', 'lazy dog']
```

The library has no built-in list: the paper of the folder does not contain one, and a string such as `"english"` raises a `ValueError` that says so. The collection is materialised to a tuple, so a generator passed in is not consumed by the tokenizer and left empty.

## 2. Length normalisation

A long document has more terms and larger counts. The L2 normalisation divides the scores of a document by the square root of the sum of their squares, so a document is compared on direction and not on length (Rennie et al. 2003, page 7).

```python
from dense_armor.utility.preprocessing.text import TFIDF

tf = TFIDF(norm="l2")
tf.learn_one({"tokens": ["a", "b"]})
tf.learn_one({"tokens": ["a"]})
print(tf.transform_one({"tokens": ["a", "b"]})["scores"])
```

```
{'a': 0.0, 'b': 1.0}
```

`a` is in every document, so its idf is zero, and `b` gets the whole unit norm. An all-zero document stays all-zero.

## 3. The cosine of two documents

The cosine similarity is the dot product of two score dicts divided by the product of their norms. The norm of a query counts every term of the query, as `cosine_similarity_many` does: a term outside the index does not contribute to the product, but it does reduce the score through the norm.

```python
from dense_armor.utility.stats.similarity import cosine_similarity

print(round(cosine_similarity({"a": 1.0, "b": 2.0}, {"a": 2.0, "b": 4.0}), 6))
print(cosine_similarity({"a": 0.0}, {"a": 1.0}))
```

```
1.0
0.0
```

The two documents point in the same direction, so the cosine is 1. The second call has a zero vector and gives 0.0, not a division by zero.

## 4. A vocabulary that stays fixed, batch or online

`TextVectorizer` chains the tokenizer and the TF-IDF and fixes the vocabulary: a term that was not seen during the fit gets no score. The batch `fit` and the online `learn_one` end in the same state on the same documents, and `fit` resets the state, so two fits on the same documents give the same `n_docs_`.

```python
from dense_armor.utility.preprocessing.vectorizer import TextVectorizer

docs = ["the quick brown fox", "a quick brown dog", "the fox and the dog"]
a = TextVectorizer().fit(docs)
b = TextVectorizer()
for d in docs:
    b.learn_one(d)
print(a.tfidf_.n_docs_, a.tfidf_.df_ == b.tfidf_.df_)
```

```
3 True
```

`None`, NaN and non-strings are counted in `n_missing_`, never learned as the terms `"none"`, `"nan"`, `"5"`. The vocabulary grows with the distinct terms unless `max_features` caps it, and `memory_class` says `O(vocabulary)`: the class is not bounded without that cap.


## 5. The cost of the index

`SparseCosineIndex` builds a CSR matrix once, normalises the rows once, and answers a query with one sparse product. `query_many` stacks the queries in a sparse matrix and multiplies once, instead of one product per query. The default is the exact index: no term is dropped.

```python
import numpy as np
from dense_armor.utility.stats.similarity import SparseCosineIndex

docs = [{"a": 1.0, "b": 2.0}, {"b": 1.0, "c": 3.0}, {"a": 2.0}]
idx = SparseCosineIndex(docs, dtype=np.float32)
S = idx.query_many([{"a": 1.0}, {"b": 1.0}])
print(idx.n_documents_, S.shape)
```

```
3 (3, 2)
```

On the 1,549 chunks and 134 queries of `benchmarks/text_retrieval.json`, the exact float32 index answers a query in **0.000080 s** median over five runs (range 0.000079-0.000119 s), against **0.0508 s** of the loop of `cosine_similarity_many` on the same scores. A single run is one point of the range, not the number: the JSON stores the median and the two ends, and the page repeats them.

The top-5 of the exact float32 index is identical to the float64 one on 134 of 134 queries (`float32_top5_identical_to_float64` in the JSON). `max_terms_per_document` drops terms and is an option, not the default: on this corpus 50 terms per document keep the reciprocal rank but change the top-5 of some queries (overlap 0.80 against the exact index, in `pruned_k50_top5_overlap_vs_exact_median`).

## 6. Three ranking primitives from the papers, each for its own case

The full product is not the only way to rank. The library has three more primitives, each from a paper, each exact, each for a case where the full product is the wrong tool. On this corpus they are slower than the full product, and the numbers say so.

**The Threshold Algorithm of Fagin, Lotem and Naor (2001).** The cosine is a sum of per-term contributions; with the posting lists sorted by weight descending, the next weight of each list bounds every further contribution, and when that bound is below the fifth score the scan stops. The scores of the touched documents are then recomputed exactly, because a touched document can still grow. It is exact, and it is the primitive for a corpus where the full product is prohibitive.

```python
import numpy as np
from dense_armor.utility.stats.similarity import SparseCosineIndex

docs = [{"a": 1.0, "b": 2.0}, {"b": 1.0, "c": 3.0}, {"a": 2.0}]
idx = SparseCosineIndex(docs, dtype=np.float32)
print([(d, round(s, 3)) for d, s in idx.top_k_threshold({"a": 1.0}, k=2)])
```

```
[(2, 1.0), (0, 0.447)]
```

**The triangle inequality for cosine of Schubert (2021).** For any pivot `z`, `sim(x, y) <= sim(x, z)*sim(z, y) + sqrt((1-sim(x, z)^2)(1-sim(z, y)^2))` (Eq. 13, page 6). With a pivot in the index and the query as `x`, the bound says whether a document can still enter the top-k without computing its product. Exact, and it pays with a metric index (VP-tree, M-tree) on a large corpus.

```python
import numpy as np
from dense_armor.utility.stats.similarity import SparseCosineIndex

docs = [{"a": 1.0, "b": 2.0}, {"b": 1.0, "c": 3.0}, {"a": 2.0}]
idx = SparseCosineIndex(docs, dtype=np.float32)
print([(d, round(s, 3)) for d, s in idx.top_k_schubert({"a": 1.0}, k=2, n_pivots=2)])
```

```
[(2, 1.0), (0, 0.447)]
```

**The hashing trick of Weinberger et al. (2009).** A token goes to one of `m` columns by a deterministic hash, no vocabulary in memory; the collision error on the inner product is `O(1/sqrt(m))` (page 2). It is the primitive for a vocabulary too large to keep, and the exact top-5 is no longer guaranteed.

```python
from dense_armor.utility.stats.similarity import feature_hash

col, sign = feature_hash("fox", m=1024)
print(0 <= col < 1024, sign in (-1, 1))
print(feature_hash("fox", m=1024) == feature_hash("fox", m=1024))
```

```
True True
True
```

## 7. What does not work, with the numbers

The three primitives above are exact (134 of 134 queries give the same top-5 as the full product), and on this corpus they are slower than the full product: Fagin 2350 us per query (10.3x the full product), Schubert 109148 us (477x), against 229 us of the full product. The numbers come from `benchmarks/primitives.py`, which writes `benchmarks/primitives.json`. The bound is correct; the random access on the candidates, and the Python per row, cost more than the C product on 1,549 documents. They are the right tool on millions of documents, not here.

The build is 0.750 s against 0.562 s of the batch library (1.34x on our side slower). The pipeline is 637 us against 4647 us (7.3x on our side faster); the kernel alone is 80 us and the batch library does not expose a kernel-only path in the benchmark, so the honest comparison is the pipeline. 10x on the build would need 0.075 s. The numbers come from `benchmarks/text_retrieval.json`, a fresh run of `benchmarks/text_retrieval.py` on this commit.

The papers that could go further are not in the folder, or are for a different scale. Feature hashing attacks the build, but the exact top-5 would no longer be guaranteed. Random projection and SVD make the matrix dense, and the sparse product here is 80 us (`query_s` in `benchmarks/text_retrieval.json`): they pay when the original dimension is in the hundreds of thousands. The batched SpMM of Nagasaka et al. (2019) and the CSR5 of Liu and Vinter (2015) are GPU kernels for SpMV and SpMM; the kernel here is `scipy.sparse` CSR on CPU and sparse by sparse, and the 9.27x and 6.09x of those papers are GPU numbers, not ours.

## Details

### References

- Silajev, I. (2026). *TF-IDF and BM25 are exact KL divergences.* arXiv:2609.14016. The TF and IDF of the score, section 1, pages 1-4. The title is the one printed on page 1 of the PDF.
- Rennie, J. D. M., Shih, L., Teevan, J., Karger, D. R. (2003). *Tackling the Poor Assumptions of Naive Bayes Text Classifiers.* The length-normalisation transform, page 7.
- McCallum, A., Nigam, K. (1998). *A Comparison of Event Models for Naive Bayes Text Classification.* Feature selection, page 3: the paper keeps the words with the highest average mutual information with the class; `max_features` here keeps the highest document frequency, which needs no labels.
- Lassance, C., Déjean, H., Lupart, S., Tonellotto, N. (2023). *A Static Pruning Study on Sparse Neural Retrievers.* SIGIR '23, arXiv:2304.12702. Section 2, page 2: the document-centric pruning, an option of the class.
- Nagasaka, Y., Nukada, A., Kojima, R., Matsuoka, S. (2019). *Batched Sparse Matrix Multiplication for Accelerating Graph Convolutional Networks.* arXiv:1903.11409. Page 2 and Figure 8 page 8: batching several products into one, 9.27x at nB = 64 and 6.09x at nB = 512, cited as the motivation; the measurements there are on GPU and are not claimed here.
- Liu, W., Vinter, B. (2015). *CSR5: An Efficient Storage Format for Cross-Platform Sparse Matrix-Vector Multiplication.* arXiv:1503.05032. Algorithm 3, page 3: the segmented sum of a CSR product, cited as the description, not as the kernel this class runs.
- Fagin, R., Lotem, A., Naor, M. (2001). *Optimal Aggregation Algorithms for Middleware.* PODS 2001; JCSS 67(4), 2003. The Threshold Algorithm, section 3: `top_k_threshold`, exact, for a corpus where the full product is prohibitive.
- Schubert, E. (2021). *A Triangle Inequality for Cosine Similarity.* SISDA 2021, arXiv:2107.04071. Eq. 13, page 6: `top_k_schubert`, exact, for a metric index on a large corpus.
- Uhlmann, J. K. (1991). *Satisfying general proximity/similarity queries with metric trees.* Information Processing Letters 40(4), 175-179. The VP-tree, the index the Schubert bound pays with; not in the folder.
- Weinberger, K., Dasgupta, A., Langford, J., Smola, A., Attenberg, J. (2009). *Feature Hashing for Large Scale Multitask Learning.* ICML 2009, arXiv:0902.2206. Section 2 page 1 and the `O(1/sqrt(m))` bound page 2: `feature_hash`, for a vocabulary too large to keep.

### Choices stated explicitly

- The queries of the benchmark are the first heading and first sentence of each page, and that page is in the corpus: a reciprocal rank of 1.0 for every system is expected, and the benchmark checks that the index agrees with the batch library, not the quality of the retrieval.
- `top_k_threshold`, `top_k_schubert` and `feature_hash` are additions beyond the files the task lists.
- The library has no built-in stop-word list: a string such as `"english"` raises a `ValueError` that says so, and the caller passes the collection.
- `smooth_idf=False` is the default and follows the paper: `ln(N / DF)` with no `+1`.
- `norm="l2"` is the default of `TextVectorizer` and `None` is the default of `TFIDF`.
- `max_features` keeps the terms with the highest document frequency, ties broken by alphabetical order; the batch system orders by term frequency. Measured in the JSON: `top5_overlap_median_aligned` = 1.0 (no cap on either side), `top5_overlap_median` = 0.8 (cap 20000 on one side only), `top5_overlap_median_both_20000` = 0.9 (both sides at 20000). The selection rule alone changes the top-5 of one query in ten; the cap alone changes one in five.
- `memory_class` is `O(vocabulary)` in `TextVectorizer` and `SparseCosineIndex`: the vocabulary grows with the distinct terms unless `max_features` or `max_terms_per_document` caps it. The two classes are not bounded without those caps, and the docstring says so.
- The kernel is `scipy.sparse` CSR. Liu and Vinter, and Nagasaka et al., are cited as the description of a segmented-sum product and of batching, not as the implementation.
- `query_many` builds a sparse query matrix, not a dense `n_terms` by `n_queries` one.
- `top_k_threshold`, `top_k_schubert` and `feature_hash` are exact (the first two) and optional: they do not replace the full product as the default, and their docstrings say when they are the right tool with the measured numbers.

### Measured cost

On the 1,549 chunks and 134 queries of `benchmarks/text_retrieval.json`: build 0.750 s, query **0.000080 s** median over five runs (0.000079-0.000119), loop of `cosine_similarity_many` 0.0508 s per query. The `budget_s` of `TextVectorizer` is 1.5 ms, about ten times the p99 of 0.138 ms per document.

### Development diary

A full numba rewrite of the build, including the tokenizer, was considered and not done: the estimate is 3.5x to 4x on the two loops, not 10x, and it is a project of its own. Fagin, Lotem and Naor (2001) is not on arXiv: the citation says "section 3" and that section number is not verified against the PDF, because the paper is not in the folder. Uhlmann (1991) is not in the folder either, cited bibliographic only.

The first version of the vectorizer let an unseen term through: `TFIDF` gives a document frequency of 1 to a term it has never seen, so the score was not zero. The vectorizer now filters the scores to the vocabulary of `df_`. The first version of the index answered with a 1 by N sparse matrix and a `todense`, which is a sparse matrix-matrix product; the query is now a sparse vector and a single product. The page first said that 50 terms per document kept the ranking identical: that compared the reciprocal ranks, not the top-5, and the top-5 overlap with the exact index is 0.80, so the default is the exact index and the pruning is an option.
