# Text: words, counts, weights and hashing

## 6. Text: words, counts and weights

A sentence becomes a list of words, then counts; TF-IDF weighs each word by how often it
appears in this document and how rare it is across all documents seen.

```python
from dense_armor.utility.preprocessing.text import TFIDF, BagOfWords, Tokenizer

tk, bw, tf = Tokenizer(), BagOfWords(), TFIDF()
docs = ["joint three overheats", "joint three ok", "gripper ok"]
for d in docs:
    tf.learn_one(bw.transform_one(tk.transform_one(d)))
out = tf.transform_one(bw.transform_one(tk.transform_one("joint three overheats")))
print({k: round(v, 3) for k, v in out["scores"].items()})
```

```
{'joint': 0.135, 'three': 0.135, 'overheats': 0.366}
```

The formula is the one of Silajev (2026), section 1:

$$\text{TF-IDF}(t, i) = \frac{d_i(t)}{|d_i|}\,\ln\frac{N}{\text{DF}(t)}$$

$d_i(t)$ is how many times the word appears in the document, $|d_i|$ the document length, $N$
the number of documents seen (3), $\text{DF}(t)$ how many contain the word: "overheats"
appears in one document, so it weighs most; "joint" and "three" appear in two.

The word "overheats" is rare and informative; "ok" appears in two documents out of three, so
its weight is small. The ratio $d_i(t)/|d_i|$ is the frequency of the word inside the
document, and the logarithm of $N/\text{DF}(t)$ rewards words that few documents share. Silajev
(2026) shows that the same formula is exactly a Kullback-Leibler divergence between two
probability models.

`TFIDF()` follows the paper exactly. The option `smooth_idf=True` replaces the logarithm with
$\ln(1 + N/\text{DF})$, a common practical variant that keeps every word with at least a small
positive score; it is off by default because the paper's formula does not include it.

## 7. Text in a fixed size: hashing

```python
from dense_armor.utility.preprocessing.text import FeatureHasher

fh = FeatureHasher(n_features=8, seed=0)
print(fh.transform_one({"counts": {"joint": 1.0, "three": 1.0, "overheats": 1.0}})["hashed"])
```

```
{7: -1.0, 2: 1.0, 5: 1.0}
```

Every word is sent by a hash function to one of 8 slots, with a random sign; the vocabulary
can grow forever, the vector stays the same size. Weinberger et al. (2009), Theorem 3
(eq. 4): for a unit vector $x$ with
$\|x\|_\infty \le \epsilon/(18\sqrt{\log(1/\delta)\log(m/\delta)})$ and
$m \ge 72\log(1/\delta)/\epsilon^2$, the hashed squared length is within $1 \pm \epsilon$ with
probability at least $1 - 2\delta$.

The map used here is the *signed* hash of the paper: each word gets an index and a random
sign, and words that share a slot add their weights with signs that cancel on average. Without
the sign, two words in the same slot would always add up, biasing the sum upward. The bound
above says how many slots are enough: the error decreases with more slots, the confidence
increases with fewer collisions, and the largest single coordinate of the input must not
dominate the vector. In practice eight slots already separate the three words of the example,
each landing in its own bucket.

- [Text vectorizer](text_vectorizer.md): TF-IDF with a fixed vocabulary, batch or online, and the cosine ranking
