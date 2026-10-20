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

A long document has more terms and larger counts. The L2 normalisation divides the scores of a document by the square root of the sum of their squares, so a document is compared on direction and not on length (Rennie et al. 2003, pages 6-7).

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
1541 (1541, 1)
```

On the 1,541 chunks and 134 queries of `benchmarks/text_retrieval.json`, the exact float32 index answers a query in **0.000118 s** median over five runs (range 0.000117-0.000126 s), against **0.0458 s** of the loop of `cosine_similarity_many` on the same scores. A single run is one point of the range, not the number: the JSON stores the median and the two ends, and the page repeats them.

The top-5 of the exact float32 index is identical to the float64 one on 134 of 134 queries. `max_terms_per_document` drops terms and is an option, not the default: on this corpus 50 terms per document keep the reciprocal rank but change the top-5 of some queries (overlap 0.80 against the exact index).

## Details

### References

- Silajev, I. (2026). *TF-IDF and BM25 as Kullback-Leibler divergences.* arXiv:2609.14016. The TF and IDF of the score, section 1, pages 1-4. Page 1 of the PDF is titled "TF-IDF and BM25 are exact KL divergences"; the docstring of `TFIDF` cites the other title, reported here, not changed.
- Rennie, J. D. M., Shih, L., Teevan, J., Karger, D. R. (2003). *Tackling the Poor Assumptions of Naive Bayes Text Classifiers.* The length-normalisation transform, pages 6-7.
- McCallum, A., Nigam, K. (1998). *A Comparison of Event Models for Naive Bayes Text Classification.* Vocabulary selection, page 4.
- Lassance, C., Déjean, H., Lupart, S., Tonellotto, N. (2023). *A Static Pruning Study on Sparse Neural Retrievers.* SIGIR '23, arXiv:2304.12702. Section 2, page 2: the document-centric pruning, an option of the class.
- Jang, D. et al. (2019). *Batched Sparse Matrix Multiplication for Accelerating Graph Convolutional Networks.* arXiv:1903.11409. Page 2 and Figure 8 page 8: batching several products into one, cited as the motivation; the measurements there are on GPU and are not claimed here.
- Liu, W., Vinter, B. (2015). *CSR5: An Efficient Storage Format for Cross-Platform Sparse Matrix-Vector Multiplication.* arXiv:1503.05032. Algorithm 3, page 3: the segmented sum of a CSR product, cited as the description, not as the kernel this class runs.

### Choices stated explicitly

- The library has no built-in stop-word list: a string such as `"english"` raises a `ValueError` that says so, and the caller passes the collection.
- `smooth_idf=False` is the default and follows the paper: `ln(N / DF)` with no `+1`.
- `norm="l2"` is the default of `TextVectorizer` and `None` is the default of `TFIDF`.
- `max_features` keeps the terms with the highest document frequency, ties broken by alphabetical order.
- `memory_class` is `O(vocabulary)` in `TextVectorizer` and `SparseCosineIndex`: the vocabulary grows with the distinct terms unless `max_features` or `max_terms_per_document` caps it. The two classes are not bounded without those caps, and the docstring says so.
- The kernel is `scipy.sparse` CSR. Liu and Vinter and Jang are cited as the description of a segmented-sum product and of batching, not as the implementation.
- `query_many` builds a sparse query matrix, not a dense `n_terms` by `n_queries` one.

### Measured cost

On the 1,541 chunks and 134 queries of `benchmarks/text_retrieval.json`: build 1.137 s, query **0.000118 s** median over five runs (0.000117-0.000126), loop of `cosine_similarity_many` 0.0458 s per query. The `budget_s` of `TextVectorizer` is 1.5 ms, about ten times the p99 of 0.138 ms per document.

### Development diary

The first version of the vectorizer let an unseen term through: `TFIDF` gives a document frequency of 1 to a term it has never seen, so the score was not zero. The vectorizer now filters the scores to the vocabulary of `df_`. The first version of the index answered with a 1 by N sparse matrix and a `todense`, which is a sparse matrix-matrix product; the query is now a sparse vector and a single product. The page first said that 50 terms per document kept the ranking identical: that compared the reciprocal ranks, not the top-5, and the top-5 overlap with the exact index is 0.80, so the default is the exact index and the pruning is an option.
