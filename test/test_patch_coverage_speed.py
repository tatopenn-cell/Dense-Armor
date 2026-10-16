from types import SimpleNamespace

from dense_armor.checks.check_estimator import _roc_auc_score
from dense_armor.utility.drift.adwin import ADWIN
from dense_armor.utility.forest.arf import AdaptiveRandomForestClassifier
from dense_armor.utility.tree import hoeffding
from dense_armor.utility.tree.hoeffding import HoeffdingTreeClassifier


def test_auc_single_class_is_one_half():
    assert _roc_auc_score([1, 1, 1], [0.2, 0.5, 0.9]) == 0.5
    assert _roc_auc_score([0, 0], [0.2, 0.5]) == 0.5


def test_auc_ties_share_the_average_rank():
    assert _roc_auc_score([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert _roc_auc_score([0, 0, 1, 1], [0.1, 0.4, 0.4, 0.9]) == 0.875


def test_auc_perfect_and_reversed():
    assert _roc_auc_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert _roc_auc_score([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0


def test_adwin_ignores_non_finite_and_counts_them():
    det = ADWIN()
    det.update(1.0)
    det.update(float("nan"))
    det.update(float("inf"))
    assert det.n_missing == 2
    assert not det.drift_detected


def test_adwin_window_cap_drops_oldest():
    det = ADWIN(max_window=64)
    for i in range(500):
        det.update(float(i % 2))
    assert det._n <= 65


def test_arf_empty_vote_when_all_probabilities_are_zero():
    f = AdaptiveRandomForestClassifier(n_models=2, seed=1)
    stub = SimpleNamespace(predict_proba_one=lambda d: {"a": 0.0, "b": 0.0})
    f.members_ = [SimpleNamespace(weight=1.0, tree=stub), SimpleNamespace(weight=1.0, tree=stub)]
    assert f.predict_proba_one({"x": 1.0}) == {}


def test_hoeffding_entropy_of_nothing_is_zero():
    assert hoeffding._entropy({}) == 0.0
    assert hoeffding._entropy({"a": 0.0}) == 0.0


def test_hoeffding_leaf_without_classes_returns_nothing():
    t = HoeffdingTreeClassifier()
    assert t._predict_proba_leaf(None, {"x": 1.0}) == {}
    assert t._predict_class_leaf(None, {"x": 1.0}) is None
    assert t.predict_proba_one({"x": 1.0}) == {}
