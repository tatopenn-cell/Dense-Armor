"""Tests for the CASPER reader.

The real Kaggle file is not in the repository. The test writes a tiny
CSV with the same columns in ``tmp_path`` and checks the parser. All
numbers in the CSV are made up.
"""

from pathlib import Path

import pytest

from dense_armor.utility.datasets import Casper


def _write_mini(path: Path) -> None:
    rows = ["Timestamp,Actual Joint Velocities,Actual Joint Positions"]
    for i in range(6):
        t = i * 0.05
        vel = f'"[{0.1 + i * 0.1}, {-0.2 - i * 0.1}, 0.3, 0.0, 0.4, -0.5]"'
        pos = f'"[{1.0 + i * 0.1}, {-1.0 - i * 0.1}, 2.0, 0.5, 0.0, 0.2]"'
        rows.append(f"{t},{vel},{pos}")
    (path / "right_arm.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_casper_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        Casper(tmp_path)


def test_casper_reads_joint_velocities(tmp_path):
    _write_mini(tmp_path)
    ds = Casper(tmp_path)
    out = list(ds.stream(columns=["Actual Joint Velocities"]))
    assert len(out) == 6
    sig0, y0 = out[0]
    assert y0 is None
    assert sig0.t == 0.0
    assert "Actual_Joint_Velocities_0" in sig0.names
    assert sig0["Actual_Joint_Velocities_0"] == pytest.approx(0.1)
    assert sig0["Actual_Joint_Velocities_1"] == pytest.approx(-0.2)
    sig5, _ = out[5]
    assert sig5["Actual_Joint_Velocities_0"] == pytest.approx(0.6)


def test_casper_reads_two_columns(tmp_path):
    _write_mini(tmp_path)
    ds = Casper(tmp_path)
    out = list(
        ds.stream(
            columns=[
                "Actual Joint Velocities",
                "Actual Joint Positions",
            ]
        )
    )
    sig0, _ = out[0]
    assert "Actual_Joint_Velocities_0" in sig0.names
    assert "Actual_Joint_Positions_0" in sig0.names
    assert sig0["Actual_Joint_Positions_0"] == pytest.approx(1.0)


def test_casper_hours_window(tmp_path):
    rows = ["Timestamp,Actual Joint Velocities"]
    for i in range(20):
        t = i * 3600.0
        vel = '"[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"'
        rows.append(f"{t},{vel}")
    (tmp_path / "right_arm.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    ds = Casper(tmp_path)
    out = list(
        ds.stream(
            columns=["Actual Joint Velocities"],
            start_h=5.0,
            stop_h=10.0,
        )
    )
    times_h = [sig.t / 3600.0 for sig, _ in out]
    assert times_h[0] == pytest.approx(5.0)
    assert times_h[-1] == pytest.approx(10.0)
    assert len(out) == 6


def test_casper_n_rows(tmp_path):
    _write_mini(tmp_path)
    assert Casper(tmp_path).n_rows() == 6
