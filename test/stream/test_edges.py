"""Edge paths of the stream readers."""

import numpy as np
import pytest

from dense_armor.roles import Signal
from dense_armor.utility.stream import iter_array, iter_csv, merge_by_time
from dense_armor.utility.stream.csv import _parse_array_cell, _parse_scalar_cell


def test_cell_parsers():
    for cell in (None, "", "[1, ", "5", "[1, 'x']"):
        assert _parse_array_cell(cell) is None
    for cell in (None, "", "x"):
        assert _parse_scalar_cell(cell) is None
    assert _parse_array_cell("(1, 2)") == [1.0, 2.0]


def test_csv_errors_empty_and_bad_cells(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text('t,v,y\n0,"[1, 2]",a\n1,"[1]",\n2,"[x]",3\n')
    for kw in ({"columns": ["zz"]}, {"target": "zz"}, {"time_col": "zz"}):
        with pytest.raises(ValueError):
            list(iter_csv(p, **kw))
    rows = list(iter_csv(p, time_col="t", target="y", array_cols=["v"]))
    assert [y for _, y in rows] == ["a", None, 3.0]
    assert np.isnan(np.asarray(rows[1][0].array)).all()
    empty = tmp_path / "e.csv"
    empty.write_text("t,v\n")
    assert list(iter_csv(empty, time_col="t")) == []
    only_t = tmp_path / "t.csv"
    only_t.write_text("t\n0\n1\n")
    assert [s.names for s, _ in iter_csv(only_t, time_col="t")] == [[], []]


def test_merge_empty_and_missing_time():
    a = iter_array(np.array([[1.0]]), t=np.array([0.0]), names=["a"])
    assert len(list(merge_by_time(iter([]), a))) == 1
    bad = Signal(values=[1.0], names=["a"], units=[""])
    with pytest.raises(ValueError):
        list(merge_by_time(iter([(bad, None)])))
    good = Signal(values=[1.0], names=["a"], units=[""], t=0.0)
    with pytest.raises(ValueError):
        list(merge_by_time(iter([(good, None), (bad, None)])))


def test_array_units_mismatch():
    with pytest.raises(ValueError):
        list(iter_array(np.ones((2, 2)), units=["m"]))
