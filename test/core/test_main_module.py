import runpy
import sys

import pytest


def test_python_m_dense_armor_without_arguments_prints_the_usage_and_exits_with_zero(
    monkeypatch, capsys
):
    monkeypatch.setattr(sys, "argv", ["dense_armor"])
    with pytest.raises(SystemExit) as stop:
        runpy.run_module("dense_armor", run_name="__main__")
    assert stop.value.code == 0
    assert "USO: python -m dense_armor" in capsys.readouterr().out
