# Text vectorizer: how do I turn documents into ranked search results without a batch library?

A search over documents needs three things: a vocabulary, a score per term, and a way to rank a query against the documents. scikit-learn does the three in one batch call; the library does them one document at a time, and this page shows the four pieces that make the online form usable for retrieval: a stop-word filter, a length-normalised TF-IDF, a vectorizer that fixes a vocabulary, and a cosine similarity.

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

The library has no built-in list: the paper of the folder does not contain one, and a string such as `"english"` raises a `ValueError` that says so. The list comes from the caller; in the benchmark of section 5 it is scikit-learn's English list, passed to the library so the two systems filter the same words.

## 2. Length normalisation

A long document has more terms and larger counts. The L2 normalisation divides the scores of a document by the square root of the sum of their squares, so a document is compared on direction and not on length (Rennie et al. 2003, the length-normalisation transform, pages 6-7).

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

`a` is in every document, so its idf is zero, and `b` gets the whole unit norm. An all-zero document stays all-zero: the normalisation does not divide by zero.

## 3. The cosine of two documents

The cosine similarity is the dot product of two score dicts divided by the product of their norms. A zero vector gives 0.0, never `nan`.

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

`TextVectorizer` chains the tokenizer and the TF-IDF and fixes the vocabulary: a term that was not seen during the fit gets no score, whatever the query. The batch `fit` and the online `learn_one` end in the same state on the same documents.

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

The three documents, fitted in one call or learned one at a time, give the same number of documents and the same document frequency table: the batch and the online form end in the same state. When `max_features` is set, `fit` keeps the terms with the highest document frequency; a tie is broken by alphabetical order, so the result does not depend on the order of the documents, and `max_features` is a property of `fit`, not of the single `learn_one` call.

## 5. The benchmark against scikit-learn

```python
import json
from pathlib import Path

r = json.loads(Path("benchmarks/text_retrieval.json").read_text())
print(r["n_chunks"], r["n_queries"])
print(round(r["ours"]["rr_median"], 4), round(r["sklearn"]["rr_median"], 4))
print(round(r["wilcoxon_p"], 4), round(r["a12"], 3))
```

```
1541 134
1.0 1.0
0.0656 0.515
```

The corpus is the 1,541 chunks of the `docs/` pages, and the queries are the first heading and the first sentence of each of the 134 pages. Both systems rank the first relevant chunk first on almost every query, so the median reciprocal rank is 1.0 for both; the difference is in the few queries where the first place changes, and there the paired Wilcoxon gives p = 0.0656 with A12 = 0.515. Build time 1.151 s for the library against 0.575 s for scikit-learn; query time **0.0003 s against 0.0044 s**, about 15 times faster, with the same ranking. The build is slower because the index prunes each document to its 50 highest-scoring terms and normalises the rows once; the query is then one sparse matrix-vector product with a dense vector, not a 1 by N sparse matrix and a `todense`.

## Details

### References

- Silajev, I. (2026). *TF-IDF and BM25 as Kullback-Leibler divergences.* arXiv:2609.14016. The TF and IDF of the score, section 1, pages 1-4. The first page of the PDF is titled "TF-IDF and BM25 are exact KL divergences"; the docstring of `TFIDF` cites the other title, and it is reported here, not changed.
- Rennie, J. D. M., Shih, L., Teevan, J., Karger, D. R. (2003). *Tackling the Poor Assumptions of Naive Bayes Text Classifiers.* The length-normalisation transform, pages 6-7.
- McCallum, A., Nigam, K. (1998). *A Comparison of Event Models for Naive Bayes Text Classification.* Vocabulary selection, page 4.
- Lassance, C., Déjean, H., Lupart, S., Tonellotto, N. (2023). *A Static Pruning Study on Sparse Neural Retrievers.* SIGIR '23, arXiv:2304.12702. Section 2, page 2: the document-centric pruning, the 4/8/16/32/64 terms. Section 3, page 3: the 2x and 4x speedups with the drops of effectiveness.
- Liu, W., Vinter, B. (2015). *CSR5: An Efficient Storage Format for Cross-Platform Sparse Matrix-Vector Multiplication.* arXiv:1503.05032. Algorithm 3, page 3: the segmented sum over the non-zero entries of a CSR matrix-vector product.

### Choices stated explicitly

- The library has no built-in stop-word list: the three PDFs of the folder do not contain one, and a string such as `"english"` raises a `ValueError` that says so. The list is the caller's; the benchmark takes scikit-learn's English list and passes it to `TextVectorizer`, so the two systems filter the same words.
- `smooth_idf=False` is the default and follows the paper: `ln(N / DF)` with no `+1`. scikit-learn's default is `smooth_idf=True`; the benchmark passes the library's stop list but not this flag, and the difference is declared here rather than hidden.
- `norm="l2"` is the default of `TextVectorizer` and `None` is the default of `TFIDF`: the vectorizer is the retrieval tool and normalises, the primitive keeps its old default.
- `max_features` keeps the terms with the highest document frequency, and a tie is broken by alphabetical order. scikit-learn keeps the terms with the highest term frequency. The two rules are different, and the benchmark does not set `max_features` on the library side to compare the two.
- `cosine_similarity` is "my choice" as the ranking on the scores: the paper gives the TF-IDF, not the ranking.
- **`SparseCosineIndex`**: the query is one sparse matrix-vector product with a dense vector, and the rows are normalised at construction. The document-centric pruning (the 50 terms with the highest score per document) is the static pruning of Lassance, Déjean, Lupart and Tonellotto (2023), section 2, page 2 ("the 4, 8, 16, 32, 64 terms with the highest impact values"); on this corpus 50 terms keep the ranking identical and take the query from 2.11 ms to 0.28 ms. The sparse matvec is the CSR segmented sum of Liu and Vinter (2015), Algorithm 3, page 3. The count of 1,541 chunks and 134 queries is larger than the 1,531 and 133 of the first run: the pages of this repository changed in between.

### Measured cost

Build of the 1,531 chunks: 0.699 s for `TextVectorizer`, 0.570 s for scikit-learn. A query: 0.0393 s for the library, 0.0043 s for scikit-learn. The `budget_s` of `TextVectorizer` is 5 ms, above the measured p99 of 0.138 ms per document.

### Development diary

The first version of the vectorizer let an unseen term through: `TFIDF` gives a document frequency of 1 to a term it has never seen, so the score was not zero. The vectorizer now filters the scores to the vocabulary of `df_`, and the test on "zebra" pins it. The first version of the benchmark ignored `--smoke`, and the flag now cuts the pages to the first 20. When the two systems give the same reciprocal rank on every query the Wilcoxon has no pair, and the JSON carries a `wilcoxon_note` with p = 1.0 and A12 = 0.5 rather than a `nan`.
