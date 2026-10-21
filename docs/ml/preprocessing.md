# Preprocessing: the algorithms

Before a model sees the data, the table is rescaled, some columns are dropped, the joint records are turned into velocities and the text is turned into numbers. Each class reads the whole table with `fit` and applies the result with `transform`.

Every class reads the whole dataset with `fit` and leaves its results in attributes with a trailing underscore; `transform` or a query method applies them. The data are `synthetic` where a block makes them up.

## 1. `StandardScaler`

Standardise with the mean and the standard deviation (`n - 1`) of the table.

~~~python
from dense_armor.ml.preprocessing import StandardScaler
print(StandardScaler().fit_transform([[1.0], [3.0]]).round(4).tolist())
~~~

~~~
[[-0.7071], [0.7071]]
~~~

## 2. `EWScaler`

Standardise with the exponentially weighted mean and standard deviation.

~~~python
from dense_armor.ml.preprocessing import EWScaler
print(EWScaler(alpha=0.5).fit([[1.0], [3.0]]).center_.tolist())
~~~

~~~
[2.0]
~~~

## 3. `RobustScaler`

Centre on the median and scale by the MAD of the last `window` rows.

~~~python
from dense_armor.ml.preprocessing import RobustScaler
print(RobustScaler(window=3).fit([[1.0], [2.0], [9.0]]).scale_.tolist())
~~~

~~~
[1.0]
~~~

## 4. `MinMaxScaler`

Rescale each column to `[0, 1]` with the minimum and the maximum of the table.

~~~python
from dense_armor.ml.preprocessing import MinMaxScaler
print(MinMaxScaler().fit_transform([[1.0], [3.0], [2.0]]).T.tolist())
~~~

~~~
[[0.0, 1.0, 0.5]]
~~~

## 5. `VarianceThreshold`

Drop the columns whose variance (`n - 1`) is at or below `threshold`.

~~~python
from dense_armor.ml.preprocessing import VarianceThreshold
print(VarianceThreshold().fit([[1.0, 5.0], [2.0, 5.0]]).keep_.tolist())
~~~

~~~
[True, False]
~~~

## 6. `SelectKBest`

Keep the `k` columns with the highest absolute Pearson correlation with `y`.

~~~python
from dense_armor.ml.preprocessing import SelectKBest
X = [[1.0, 3.0], [2.0, 1.0], [3.0, 2.0]]
print(SelectKBest(k=1).fit(X, [1.0, 2.0, 3.0]).selected_.tolist())
~~~

~~~
[0]
~~~

## 7. `JointDerivatives`

Velocity, acceleration and jerk by backward finite differences on the whole record.

~~~python
from dense_armor.ml.preprocessing import JointDerivatives
d = JointDerivatives(order=1).fit([[0.0], [1.0], [3.0]], [0.0, 1.0, 2.0])
print(d.qd_.T.tolist())
~~~

~~~
[[0.0, 1.0, 2.0]]
~~~

## 8. `JointPower`

Mechanical power `tau * qd` per joint and its total, for every row.

~~~python
from dense_armor.ml.preprocessing import JointPower
print(JointPower().fit([[2.0, 1.0]], [[3.0, 4.0]]).total_.tolist())
~~~

~~~
[10.0]
~~~

## 9. `Tokenizer`

Lower-case word tokens with optional n-grams, for a list of documents.

~~~python
from dense_armor.ml.preprocessing import Tokenizer
print(Tokenizer(ngrams=2).tokenize("The cat sat"))
~~~

~~~
['the', 'cat', 'sat', 'the cat', 'cat sat']
~~~

## 10. `BagOfWords`

Term counts of every document of a list of token lists.

~~~python
from dense_armor.ml.preprocessing import BagOfWords
print(BagOfWords().fit([["a", "b", "a"]]).counts_)
~~~

~~~
[{'a': 2, 'b': 1}]
~~~

## 11. `TFIDF`

TF-IDF of a corpus: `tf = count / length`, `idf = ln(N / DF)`.

~~~python
from dense_armor.ml.preprocessing import TFIDF
t = TFIDF().fit([["a", "b"], ["a", "c"]])
print(t.df_["a"], round(t.transform_one(["b", "b"])["b"], 4))
~~~

~~~
2 0.6931
~~~

## 12. `FeatureHasher`

Signed feature hashing of a list of count dictionaries into a dense matrix.

~~~python
from dense_armor.ml.preprocessing import FeatureHasher
print(FeatureHasher(n_features=8).fit_transform([{"a": 2.0}]).shape)
~~~

~~~
(1, 8)
~~~

## 13. `TextVectorizer`

TF-IDF matrix of a corpus with a fixed vocabulary.

~~~python
from dense_armor.ml.preprocessing import TextVectorizer
v = TextVectorizer(ngram_range=(1, 1)).fit(["a b", "a c"])
print(v.vocabulary_)
~~~

~~~
['a', 'b', 'c']
~~~

## Details

### References

Standardisation, min-max scaling, variance threshold, correlation
    ranking and finite differences are textbook definitions.
    Silajev (2026), TF-IDF and BM25 as exact KL divergences, section 1:
        `tf = count / length` and `idf = ln(N / DF)`, or
        `ln(1 + N / DF)` when smoothed (not in the folder,
        bibliographic only).
    Weinberger, K. et al. (2009). Feature hashing for large scale multitask
        learning, Theorem 3: the signed hashing trick (not in the folder,
        bibliographic only).
    Malialis, K. et al. (2022). Queue-based resampling for online class
        imbalance learning: a sequential method over a stream, so
        `QueueResampler` has no class (not in the folder,
        bibliographic only).

### Choices stated explicitly

The class of a sketch is the exact value the sketch approximates, and the class of a statistic is the textbook formula on the whole array: both are `our choice`, because the references name no other batch algorithm.

### Development diary

The classes are new and independent of the online classes of dense_armor.utility.

Classes without a form:
- `QueueResampler`: the reference describes a queue over a stream: it holds no algorithm for the whole dataset
