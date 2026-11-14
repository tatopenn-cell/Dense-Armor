"""Tests for the interactive prompt (item B)."""
import io
import os
import sys
import types
from unittest.mock import patch

import pytest

from dense_armor.usable import shell as sh


# --------------------------------------------------------------
# Helper: simulates a Windows terminal with msvcrt.getwch
# --------------------------------------------------------------
def install_fake_msvcrt(chars):
    """Injects a fake msvcrt.getwch that serves the characters in order.

    Usage:
        with patch.object(os, "name", "nt"), \
             patch.object(sys, "stdin", _FakeStdin()), \
             patch.dict(sys.modules, {"msvcrt": fake}), \
             patch("sys.stdout", new_callable=io.StringIO):
            ...
    """
    it = iter(chars)
    fake = types.SimpleNamespace(getwch=lambda: next(it))
    return fake, it


class _FakeStdin(io.StringIO):
    """Fake stdin: isatty() always True, nothing else needed."""
    def isatty(self):
        return True


def _run_ask(chars):
    """Runs sh._ask('> ') with the given characters, returns (result, stdout)."""
    fake, _ = install_fake_msvcrt(chars)
    with patch.object(os, "name", "nt"), \
         patch.object(sys, "stdin", _FakeStdin()), \
         patch.dict(sys.modules, {"msvcrt": fake}), \
         patch("sys.stdout", new_callable=io.StringIO):
        return sh._ask("> ")


# --------------------------------------------------------------
# _ask with msvcrt.getwch simulated
# --------------------------------------------------------------
def test_ask_returns_line_with_enter():
    assert _run_ask(["a", "b", "c", "\r"]) == "abc"


def test_ask_strips_spaces():
    assert _run_ask([" ", "a", " ", "\r"]) == "a"


def test_ask_backspace_deletes_last_char():
    assert _run_ask(["a", "b", "\x08", "c", "\r"]) == "ac"


def test_ask_backspace_on_empty_string_is_noop():
    assert _run_ask(["\x08", "\x08", "a", "\r"]) == "a"


def test_ask_escape_raises_exit():
    with pytest.raises(sh._Exit):
        _run_ask(["a", "\x1b"])


def test_ask_ctrl_c_raises_keyboard_interrupt():
    with pytest.raises(KeyboardInterrupt):
        _run_ask(["a", "\x03"])


def test_ask_arrow_keys_are_ignored():
    """\x00\xe0 (arrow) consumes two characters, the next one enters."""
    assert _run_ask(["a", "\x00", "\xe0", "H", "\r"]) == "aH"


# --------------------------------------------------------------
# _ask su stdin da pipe (EOF = Ctrl+D)
# --------------------------------------------------------------
def test_ask_eof_returns_q():
    with patch("sys.stdin", io.StringIO("")):
        assert sh._ask("> ") == "q"


# --------------------------------------------------------------
# HELP: it does not ask for the file first
# --------------------------------------------------------------
def test_help_does_not_ask_file_first():
    assert "asks for the file first" not in sh.HELP
    assert "shows the commands" in sh.HELP


# --------------------------------------------------------------
# MENU removed (it was an orphan constant)
# --------------------------------------------------------------
def test_menu_is_removed():
    assert not hasattr(sh, "MENU")


# --------------------------------------------------------------
# _ask_file removed (dead function)
# --------------------------------------------------------------
def test_ask_file_is_removed():
    assert not hasattr(sh, "_ask_file")


# --------------------------------------------------------------
# shell() with stdin from a pipe (light integration, no network)
# --------------------------------------------------------------
def test_shell_exits_with_zero_on_pipe_exit():
    from dense_armor.usable.catalog import REPO
    with patch("sys.stdin", io.StringIO("exit\n")), \
         patch("sys.stdout", new_callable=io.StringIO):
        assert sh.shell(REPO) == 0


def test_shell_list_drift_outputs_five(capsys):
    from dense_armor.usable.catalog import REPO
    with patch("sys.stdin", io.StringIO("list drift\nexit\n")):
        assert sh.shell(REPO) == 0
    out = capsys.readouterr().out
    assert "5 primitives" in out
    assert "drift" in out


def test_shell_unknown_command(capsys):
    from dense_armor.usable.catalog import REPO
    with patch("sys.stdin", io.StringIO("boh\nexit\n")):
        assert sh.shell(REPO) == 0
    out = capsys.readouterr().out
    assert "unknown command: boh" in out
