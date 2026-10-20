"""Regression: the default output of Tokenizer and TFIDF on a fixed corpus.

Written before items A and B add stop_words and norm, so a change of the
default behaviour fails here.
"""

from dense_armor.utility.preprocessing.text import TFIDF, Tokenizer

CORPUS = ['the quick brown fox jumps over the lazy dog', 'a quick brown dog', 'the fox and the dog']
TOKENS = [['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', 'the quick', 'quick brown', 'brown fox', 'fox jumps', 'jumps over', 'over the', 'the lazy', 'lazy dog'], ['a', 'quick', 'brown', 'dog', 'a quick', 'quick brown', 'brown dog'], ['the', 'fox', 'and', 'the', 'dog', 'the fox', 'fox and', 'and the', 'the dog']]
SCORES = [{'the': 0.04770177742448993, 'quick': 0.023850888712244965, 'brown': 0.023850888712244965, 'fox': 0.023850888712244965, 'jumps': 0.06462425227459469, 'over': 0.06462425227459469, 'lazy': 0.06462425227459469, 'dog': 0.0, 'the quick': 0.06462425227459469, 'quick brown': 0.023850888712244965, 'brown fox': 0.06462425227459469, 'fox jumps': 0.06462425227459469, 'jumps over': 0.06462425227459469, 'over the': 0.06462425227459469, 'the lazy': 0.06462425227459469, 'lazy dog': 0.06462425227459469}, {'a': 0.15694461266687282, 'quick': 0.05792358687259491, 'brown': 0.05792358687259491, 'dog': 0.0, 'a quick': 0.15694461266687282, 'quick brown': 0.05792358687259491, 'brown dog': 0.15694461266687282}, {'the': 0.09010335735736986, 'fox': 0.04505167867868493, 'and': 0.12206803207423442, 'dog': 0.0, 'the fox': 0.12206803207423442, 'fox and': 0.12206803207423442, 'and the': 0.12206803207423442, 'the dog': 0.12206803207423442}]
N_DOCS = 3
DF = {'a': 1, 'a quick': 1, 'and': 1, 'and the': 1, 'brown': 2, 'brown dog': 1, 'brown fox': 1, 'dog': 3, 'fox': 2, 'fox and': 1, 'fox jumps': 1, 'jumps': 1, 'jumps over': 1, 'lazy': 1, 'lazy dog': 1, 'over': 1, 'over the': 1, 'quick': 2, 'quick brown': 2, 'the': 2, 'the dog': 1, 'the fox': 1, 'the lazy': 1, 'the quick': 1}


def test_tokenizer_default_unchanged():
    tk = Tokenizer(ngrams=2)
    got = [tk.transform_one({"text": d})["tokens"] for d in CORPUS]
    assert got == TOKENS


def test_tfidf_default_unchanged():
    tf = TFIDF()
    for toks in TOKENS:
        tf.learn_one({"tokens": toks})
    got = [tf.transform_one({"tokens": toks})["scores"] for toks in TOKENS]
    assert got == SCORES
    assert tf.n_docs_ == N_DOCS
    assert dict(sorted(tf.df_.items())) == DF
