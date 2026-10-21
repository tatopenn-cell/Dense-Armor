"""Hand cases and edge cases of the preprocessing of the machine learning package."""

import doctest

import numpy as np

import dense_armor.ml.preprocessing as P
from dense_armor.ml.preprocessing import (
    TFIDF,
    BagOfWords,
    EWScaler,
    FeatureHasher,
    JointDerivatives,
    JointPower,
    MinMaxScaler,
    RobustScaler,
    SelectKBest,
    StandardScaler,
    TextVectorizer,
    Tokenizer,
    VarianceThreshold,
)


def test_doctests():
    r = doctest.testmod(P)
    assert r.failed == 0 and r.attempted > 10


def test_standard_scaler_hand_case():
    s = StandardScaler().fit([[1.0, 5.0], [3.0, 5.0]])
    assert np.allclose(s.center_, [2.0, 5.0])
    out = s.transform([[3.0, 7.0]])
    assert np.allclose(out, [[1.0 / np.sqrt(2.0), 7.0]])
    one = StandardScaler().fit([[1.0, 2.0]])
    assert np.allclose(one.transform([[4.0, 4.0]]), [[4.0, 4.0]])
    assert StandardScaler().fit_transform([1.0, 3.0]).shape == (2, 1)


def test_ew_scaler_hand_case():
    s = EWScaler(alpha=0.5).fit([[1.0], [3.0], [3.0]])
    assert abs(s.center_[0] - 2.5) < 1e-12 and abs(s.scale_[0] - np.sqrt(0.75)) < 1e-12
    assert np.allclose(s.transform([[2.5 + np.sqrt(0.75)]]), [[1.0]])


def test_robust_scaler_hand_case():
    s = RobustScaler(window=3).fit([[100.0], [1.0], [2.0], [9.0]])
    assert s.center_.tolist() == [2.0] and s.scale_.tolist() == [1.0]
    assert s.transform([[5.0]]).tolist() == [[3.0]]
    flat = RobustScaler(window=3).fit([[4.0], [4.0], [4.0]])
    assert flat.transform([[9.0]]).tolist() == [[9.0]]


def test_minmax_scaler_hand_case():
    s = MinMaxScaler().fit([[2.0, 7.0], [4.0, 7.0], [3.0, 7.0]])
    assert s.transform([[3.0, 7.0], [6.0, 7.0]]).tolist() == [[0.5, 7.0], [2.0, 7.0]]


def test_variance_threshold_hand_case():
    v = VarianceThreshold(threshold=0.4).fit(
        [[1.0, 2.0, 5.0], [2.0, 2.5, 5.0], [3.0, 2.0, 5.0]]
    )
    assert np.allclose(v.variance_, [1.0, 1.0 / 12.0, 0.0])
    assert v.keep_.tolist() == [True, False, False]
    assert v.transform([[7.0, 8.0, 9.0]]).tolist() == [[7.0]]
    single = VarianceThreshold().fit([[1.0, 2.0]])
    assert single.keep_.tolist() == [True, True]


def test_select_k_best_hand_case():
    X = [[1.0, 3.0, 0.0], [2.0, 1.0, 5.0], [3.0, 2.0, 0.0], [4.0, 5.0, 5.0]]
    s = SelectKBest(k=2).fit(X, [1.0, 2.0, 3.0, 4.0])
    assert s.selected_.tolist() == [0, 1] and s.scores_[0] == 1.0
    assert s.transform(X).shape == (4, 2)
    flat = SelectKBest(k=1).fit([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]], [1.0, 2.0, 3.0])
    assert flat.selected_.tolist() == [1] and flat.scores_[0] == 0.0


def test_joint_derivatives_hand_case():
    q = [[0.0, 0.0], [1.0, 2.0], [4.0, 2.0], [9.0, 2.0]]
    d = JointDerivatives().fit(q, [0.0, 1.0, 2.0, 3.0])
    assert d.qd_.tolist() == [[0.0, 0.0], [1.0, 2.0], [3.0, 0.0], [5.0, 0.0]]
    assert d.qdd_.tolist() == [[0.0, 0.0], [1.0, 2.0], [2.0, -2.0], [2.0, 0.0]]
    assert d.jerk_.tolist() == [[0.0, 0.0], [1.0, 2.0], [1.0, -4.0], [0.0, 2.0]]


def test_joint_power_hand_case():
    p = JointPower().fit([[2.0, 1.0], [0.0, 3.0]], [[3.0, 4.0], [5.0, 2.0]])
    assert p.power_.tolist() == [[6.0, 4.0], [0.0, 6.0]] and p.total_.tolist() == [
        10.0,
        6.0,
    ]


def test_tokenizer_and_bag_of_words():
    t = Tokenizer(ngrams=2, stop_words=["The"])
    assert t.tokenize("The Cat, the DOG") == ["cat", "dog", "cat dog"]
    assert Tokenizer(lowercase=False).tokenize("Ab ab") == ["Ab", "ab"]
    assert Tokenizer().tokenize(42) == ["42"]
    assert Tokenizer().fit(["a b", "c"]).tokens_ == [["a", "b"], ["c"]]
    assert BagOfWords().fit([["a", "b", "a"], []]).counts_ == [{"a": 2, "b": 1}, {}]


def test_tfidf_hand_case():
    t = TFIDF().fit([["a", "b"], ["a", "c"], ["a", "b"]])
    assert t.n_docs_ == 3 and t.df_ == {"a": 3, "b": 2, "c": 1}
    s = t.transform_one(["b", "c", "c", "c"])
    assert (
        abs(s["b"] - 0.25 * np.log(1.5)) < 1e-12
        and abs(s["c"] - 0.75 * np.log(3.0)) < 1e-12
    )
    sm = TFIDF(smooth_idf=True).fit([["a"], ["a"]]).transform_one(["a"])
    assert abs(sm["a"] - np.log(2.0)) < 1e-12
    n = TFIDF(norm="l2").fit([["a", "b"], ["a", "c"]]).transform_one(["b", "c"])
    assert abs(np.hypot(*n.values()) - 1.0) < 1e-12
    z = TFIDF(norm="l2").fit([["a"], ["a"]]).transform_one(["a"])
    assert z == {"a": 0.0}
    assert TFIDF().fit([["a"]]).transform_one([]) == {}


def test_feature_hasher_hand_case():
    h = FeatureHasher(n_features=16, seed=3)
    M = h.fit([{}]).fit_transform([{"a": 2.0}, {"a": 1.0, "b": 1.0}, {}])
    assert M.shape == (3, 16) and abs(M[2]).sum() == 0.0
    assert np.count_nonzero(M[0]) == 1 and abs(M[0]).max() == 2.0
    j = int(np.flatnonzero(M[0])[0])
    assert abs(M[1][j]) in (1.0, 0.0, 2.0)


def test_text_vectorizer():
    docs = ["a b", "a c", "a b c", None]
    v = TextVectorizer(ngram_range=(1, 1)).fit(docs)
    assert v.vocabulary_ == ["a", "b", "c"] and v.n_docs_ == 3
    M = v.transform(["b", "zzz"])
    assert abs(np.linalg.norm(M[0]) - 1.0) < 1e-12 and M[1].sum() == 0.0
    top = TextVectorizer(
        ngram_range=(1, 1), max_features=2, norm=None, smooth_idf=True
    ).fit(docs)
    assert top.vocabulary_ == ["a", "b"]
    assert abs(top.transform(["a"])[0, 0] - np.log(1.0 + 3 / 3)) < 1e-12
    bi = TextVectorizer(ngram_range=(2, 2)).fit(["a b c", "a b"])
    assert bi.vocabulary_ == ["a b", "b c"]


def test_stopped_class_is_declared():
    assert "QueueResampler" in P.STOPPED
