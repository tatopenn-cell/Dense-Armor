"""Tests for ActiveStrategy."""

import pytest

from dense_armor.utility.active.base import ActiveStrategy


def test_base_strategy_is_abstract():
    with pytest.raises(NotImplementedError):
        ActiveStrategy(budget=0.1).should_label({"x": 1.0}, None)
