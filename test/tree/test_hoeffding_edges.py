"""Edge branches of HoeffdingTreeClassifier and _Gauss that a stream does not reach."""

import numpy as np

from dense_armor.utility.tree.hoeffding import (
    HoeffdingTreeClassifier,
    _Gauss,
    _Node,
)


def test_gauss_float32_variance_of_one_sample_is_zero():
    g = _Gauss(dtype=np.float32)
    g.update(1000.0)
    assert g.n_ == 1
    assert g.var == 0.0


def test_explain_one_on_an_empty_tree_is_empty():
    assert HoeffdingTreeClassifier().explain_one({"a": 1.0}) == []


def test_try_split_with_one_class_does_not_split():
    h = HoeffdingTreeClassifier(grace_period=1)
    node = _Node(0)
    node.n_ = 1
    node.class_counts_ = {0: 1}
    assert h._try_split(node) is False


def test_split_gain_without_statistics_for_the_feature_is_none():
    h = HoeffdingTreeClassifier()
    node = _Node(0)
    node.n_ = 5
    node.class_counts_ = {0: 5}
    assert h._split_gain(node, "a", 0.5) is None


def test_split_gain_of_a_node_with_no_samples_is_none():
    h = HoeffdingTreeClassifier()
    node = _Node(0)
    g = _Gauss()
    g.update(1.0)
    node.feature_stats_ = {"a": {0: g}}
    assert node.n_ == 0
    assert h._split_gain(node, "a", 0.5) is None


def test_split_gain_skips_a_class_with_no_samples_for_the_feature():
    h = HoeffdingTreeClassifier()
    node = _Node(0)
    node.n_ = 5
    node.class_counts_ = {0: 5}
    node.feature_stats_ = {"a": {0: _Gauss()}}
    assert h._split_gain(node, "a", 0.5) is None


def test_path_stops_when_a_child_is_missing():
    h = HoeffdingTreeClassifier()
    root = _Node(0)
    root.split_feature_ = "a"
    root.split_threshold_ = 0.5
    h.root_ = root
    assert h._path({"a": 0.0}) == [root]
