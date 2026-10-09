"""Tests for iter_csv."""

import tracemalloc

import pytest

from dense_armor.utility.stream import iter_csv


def _write(tmp_path, text):
    p = tmp_path / "data.csv"
    p.write_text(text, encoding="utf-8")
    return p


def test_iter_csv_basic(tmp_path):
    p = _write(tmp_path, "t,a,b\n0,1,2\n1,3,4\n")
    out = list(iter_csv(p, time_col="t"))
    assert len(out) == 2
    assert out[0][0]["a"] == 1.0
    assert out[1][0]["b"] == 4.0


def test_iter_csv_target(tmp_path):
    p = _write(tmp_path, "t,a,y\n0,1,0\n1,3,1\n")
    out = list(iter_csv(p, target="y", time_col="t"))
    assert out[0][1] == 0.0
    assert out[1][1] == 1.0


def test_iter_csv_columns_selected(tmp_path):
    p = _write(tmp_path, "t,a,b,c\n0,1,2,3\n")
    out = list(iter_csv(p, time_col="t", columns=["a", "c"]))
    sig, _ = out[0]
    assert sig.names == ["a", "c"]
    assert sig["c"] == 3.0


def test_iter_csv_array_cols(tmp_path):
    p = _write(
        tmp_path,
        't,q\n0,"[1.0, 2.0, 3.0]"\n1,"[4.0, 5.0, 6.0]"\n',
    )
    out = list(iter_csv(p, time_col="t", array_cols=["q"]))
    sig0, _ = out[0]
    assert sig0.names == ["q_0", "q_1", "q_2"]
    assert sig0["q_2"] == 3.0


def test_iter_csv_missing_cell(tmp_path):
    p = _write(tmp_path, "t,a\n0,\n1,5\n")
    out = list(iter_csv(p, time_col="t"))
    assert out[0][0].n_missing == 1
    assert out[1][0]["a"] == 5.0


def test_iter_csv_start_stop(tmp_path):
    p = _write(tmp_path, "t,a\n0,1\n1,2\n2,3\n3,4\n4,5\n")
    out = list(iter_csv(p, time_col="t", start=1.0, stop=3.0))
    assert [s.t for s, _ in out] == [1.0, 2.0, 3.0]


def test_iter_csv_unknown_column(tmp_path):
    p = _write(tmp_path, "t,a\n0,1\n")
    with pytest.raises(ValueError):
        list(iter_csv(p, time_col="t", columns=["b"]))


def test_iter_csv_memory_bounded(tmp_path):
    def _write(path, n, chunk=2000):
        with path.open("w", encoding="utf-8") as f:
            f.write("t,a\n")
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                lines = [f"{i},{i * 0.1}" for i in range(start, end)]
                f.write("\n".join(lines) + "\n")

    def _peak(n):
        path = tmp_path / f"big_{n}.csv"
        _write(path, n)
        tracemalloc.start()
        count = 0
        for _ in iter_csv(path, time_col="t"):
            count += 1
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert count == n
        return peak

    small = _peak(500)
    big = _peak(5_000)
    assert big < small * 4 + 2_000_000
