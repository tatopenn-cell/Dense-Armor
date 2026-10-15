"""check_estimator reports a check that cannot run instead of passing it."""

import pytest

from dense_armor.checks import check_estimator
from dense_armor.utility.learn.online_classifiers import OnlineGaussianNB


class _NoPickle(OnlineGaussianNB):
    """Naive Bayes whose state cannot be copied."""

    def __getstate__(self):
        raise AttributeError("no state")


def test_unlisted_not_applicable_check_fails():
    with pytest.raises(AssertionError, match="allow-list") as exc:
        check_estimator(_NoPickle())
    assert "check_pickle_roundtrip" in str(exc.value)


def test_listed_not_applicable_check_is_reported():
    report = check_estimator(
        _NoPickle(),
        allow_not_applicable={"check_pickle_roundtrip": "state is not copyable"},
    )
    assert [n for n, _ in report["not_applicable"]] == ["check_pickle_roundtrip"]
    assert "check_pickle_roundtrip" not in report["run"]
    assert report["run"]
    assert report["failed"] == []


def test_applicable_estimator_has_no_not_applicable_check():
    report = check_estimator(OnlineGaussianNB())
    assert report["not_applicable"] == []
    assert "check_pickle_roundtrip" in report["run"]
