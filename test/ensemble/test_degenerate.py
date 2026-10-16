"""Cover the degenerate paths: meta is None, members list is empty."""

from dense_armor.utility.ensemble import StackingClassifier, VotingClassifier
from dense_armor.utility.tree.hoeffding import HoeffdingTreeClassifier


def _tree():
    return HoeffdingTreeClassifier(grace_period=20)


def test_stacking_predict_proba_with_meta_none():
    s = StackingClassifier([_tree], _tree)
    # forzo lo stato inconsistente: meta_ None ma members_ pieni.
    # _ensure() ripristinerebbe meta_, quindi chiamo direttamente il
    # metodo sulla istanza dopo aver forzato meta_ = None.
    s.members_ = [s._new_member(m) for m in s.models]
    s.meta_ = None
    # bypass _ensure sostituendo il metodo
    s._ensure = lambda: None
    assert s.predict_proba_one({"x": 0.0}) == {}
    assert s.predict_one({"x": 0.0}) is None


def test_voting_predict_proba_with_empty_members():
    v = VotingClassifier([_tree])
    # forzo members_ vuoto e bypasso _ensure_members
    v.members_ = []
    v._ensure_members = lambda: None
    assert v.predict_proba_one({"x": 0.0}) == {}
