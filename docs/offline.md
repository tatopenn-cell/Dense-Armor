# Offline: the same algorithm on the whole dataset

Every online algorithm of the library can also learn from a dataset you already have, all at once: you hand it the rows and it learns from them.

## 1. One call: `fit`

An online algorithm learns one row at a time with `learn_one`. `fit` takes all the rows and feeds them to `learn_one` in order, so the result is the same as the loop you would write by hand.

```python
import numpy as np
from dense_armor.utility.linear.glm import LinearRegression
from dense_armor.utility.optim.sgd import SGD

rng = np.random.default_rng(0)
X = [{"a": float(u), "b": float(v)} for u, v in rng.normal(size=(200, 2))]
y = [2 * x["a"] - x["b"] for x in X]
m = LinearRegression(optimizer=SGD()).fit(X, y)
print(round(m.predict_one({"a": 1.0, "b": 1.0}), 2))
```

```
1.0
```

The rule behind the data is `2a - b`, so the answer for `a = 1, b = 1` is 1.0. `fit` returns the estimator itself, so the call can be chained.

## 2. Several passes, in a shuffled order

An offline dataset can be read more than once. `epochs` sets the number of passes, `shuffle` changes the order at every pass, and `seed` makes the order repeatable.

```python
m = LinearRegression(optimizer=SGD()).fit(X, y, epochs=5, shuffle=True, seed=0)
print(round(m.predict_one({"a": 1.0, "b": 1.0}), 2))
```

```
1.0
```

## 3. Algorithms without a label, and timestamps

Without `y` the call feeds `learn_one(x)`, which is what the statistics, the detectors and the transformers need. A list `t` with one timestamp per row is passed row by row and needs a single pass, because the time of a row cannot go back.

## 4. Algorithms that work on the whole dataset by nature

Most of the library learns one row at a time. Three pieces learn from all the rows in one call, the classic way, and sit next to their online counterparts so that the two protocols can be run on the same data.

### 4.1 A decision tree on the whole training set

`BatchDecisionTreeClassifier` builds a CART tree from all the rows at once.

```python
from dense_armor.utility.tree.batch import BatchDecisionTreeClassifier

X = [{"x": 0.0}, {"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
y = [0, 0, 1, 1]
clf = BatchDecisionTreeClassifier(max_depth=2).fit(X, y)
print(clf.predict_one({"x": 0.5}), clf.predict_one({"x": 2.5}))
```

```
0 1
```

At each node the tree takes the feature and the threshold with the lowest weighted gini impurity (Breiman, Friedman, Olshen and Stone, 1984, chapter 2). The tree is frozen when `fit` returns: `learn_one` after the fit leaves it unchanged. Its online counterpart is `HoeffdingTreeClassifier`, and [Static vs online](drones/static_vs_online.md) runs the two on the same stream with accuracy, time and memory.

### 4.2 Bagging fitted on bootstrap samples

`OnlineBaggingClassifier.fit_batch` fits each member on a bootstrap of the dataset, the same size with replacement, as in the batch protocol of Oza and Russell (2001), section 2, page 2.

```python
from dense_armor.utility.ensemble.bagging import OnlineBaggingClassifier
from dense_armor.utility.tree.batch import BatchDecisionTreeClassifier

X = [{"x": float(i)} for i in range(8)]
y = [0, 0, 0, 0, 1, 1, 1, 1]
bag = OnlineBaggingClassifier(BatchDecisionTreeClassifier(max_depth=2), n_models=5, seed=0)
bag.fit_batch(X, y, seed=0)
print(bag.predict_one({"x": 1.0}), bag.predict_one({"x": 6.0}), len(bag.members_))
```

```
0 1 5
```

The row with `x = 1.0` gets the class 0, the row with `x = 6.0` the class 1, and the ensemble has five members. The online protocol, `learn_one`, gives each member a number of copies of each row drawn from a Poisson law with mean 1; the two protocols reach the same limit as the number of rows grows and differ on a finite sample. Each member is any estimator with a `fit`; with the batch tree as above, the whole ensemble works offline.

### 4.3 Text: a vocabulary fitted on a list of documents

`TextVectorizer.fit` builds the vocabulary and the document frequencies from a list of documents; `learn_one` builds the same state one document at a time.

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

The three documents, fitted in one call or learned one at a time, give the same number of documents and the same document frequency table. `transform_matrix(docs)` turns a list of documents into a sparse matrix of TF-IDF scores in one pass. The cosine ranking and the comparison with the batch library are on the page [Text vectorizer](preprocessing/text_vectorizer.md).

## Details

### What `fit` does and does not do
`fit` is defined once on the base class `Root`, so every estimator of the library has it. It adds an offline way in and changes nothing that exists: `learn_one`, `learn_many` and every algorithm stay as they were. It is a loop over `learn_one`; the classic batch version of each algorithm, written from its original paper, is the next step of the program (row 24 of the master program), one family at a time; the ones that exist today are in section 4.

### Arguments
`X` is a list of rows, `y` a list of labels of the same length, `t` a list of timestamps of the same length, `epochs` an integer of at least 1, `shuffle` a boolean, `seed` an integer. A mismatch of lengths, `epochs < 1`, or `t` with more than one epoch raises `ValueError`; an object without `learn_one` raises `TypeError`.
