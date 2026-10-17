"""Tests for dense_armor/utility/_validation.py."""

import pytest

from dense_armor.utility._validation import as_dict, finite_row, is_number


class _ToDict:
    def __init__(self, d):
        self._d = d

    def to_dict(self):
        return self._d


class _NotCallable:
    to_dict = "hello"


class _Raises:
    def to_dict(self):
        raise ValueError("boom")


def test_as_dict_passes_dict_through():
    d = {"a": 1.0}
    assert as_dict(d) is d


def test_as_dict_from_to_dict():
    assert as_dict(_ToDict({"b": 2.0})) == {"b": 2.0}


def test_as_dict_empty_dict():
    assert as_dict({}) == {}


def test_as_dict_raises_on_int():
    with pytest.raises(TypeError, match="expected a dict or an object with to_dict"):
        as_dict(3)


def test_as_dict_raises_on_string():
    with pytest.raises(TypeError):
        as_dict("hello")


def test_as_dict_raises_on_non_callable_to_dict():
    with pytest.raises(TypeError):
        as_dict(_NotCallable())


def test_as_dict_propagates_exception_from_to_dict():
    with pytest.raises(ValueError, match="boom"):
        as_dict(_Raises())


def test_is_number_accepts_float_int_string_number():
    assert is_number(1.0)
    assert is_number(2)
    assert is_number("3.5")
    assert is_number(True)
    assert is_number(False)


def test_is_number_rejects_none():
    assert not is_number(None)


def test_is_number_rejects_nan_and_inf():
    assert not is_number(float("nan"))
    assert not is_number(float("inf"))
    assert not is_number(float("-inf"))


def test_is_number_rejects_string():
    assert not is_number("hello")
    assert not is_number("")


def test_is_number_rejects_object():
    assert not is_number(object())


def test_finite_row_all_valid():
    assert finite_row({"a": 1.0, "b": 2, "c": "3.5"})


def test_finite_row_empty_is_true():
    assert finite_row({})


def test_finite_row_with_none():
    assert not finite_row({"a": 1.0, "b": None})


def test_finite_row_with_nan():
    assert not finite_row({"a": float("nan")})


def test_finite_row_with_inf():
    assert not finite_row({"a": float("inf")})


def test_finite_row_with_string():
    assert not finite_row({"a": "hello"})


def test_finite_row_short_circuits():
    assert not finite_row({"a": None, "b": 1.0})
