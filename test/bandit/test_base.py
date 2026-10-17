"""Tests for BanditPolicy."""

import pytest

from dense_armor.utility.bandit.base import BanditPolicy


def test_base_policy_pull_and_update_are_abstract():
    p = BanditPolicy(arms=[0, 1])
    with pytest.raises(NotImplementedError):
        p.pull()
    with pytest.raises(NotImplementedError):
        p.update(0, 1.0)


def test_valid_rejects_a_non_numeric_reward():
    p = BanditPolicy(arms=[0, 1])
    assert p._valid(0, "x") is False


def test_unit_test_skips_is_a_set():
    assert BanditPolicy(arms=[0, 1])._unit_test_skips() == set()
