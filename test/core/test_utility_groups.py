"""The folders of dense_armor/utility sit in six groups, and the old paths still work."""
import importlib
from pathlib import Path

import pytest

import dense_armor
from dense_armor._compat import UTILITY_GROUPS

CASES = [(g, f) for g, fs in UTILITY_GROUPS.items() for f in fs]
UTILITY = Path(dense_armor.__file__).parent / "utility"


def test_utility_holds_only_the_six_groups():
    folders = {p.name for p in UTILITY.iterdir() if p.is_dir() and p.name != "__pycache__"}
    assert folders == set(UTILITY_GROUPS)


@pytest.mark.parametrize("group,folder", CASES)
def test_every_folder_is_in_its_group(group, folder):
    assert (UTILITY / group / folder / "__init__.py").is_file()


@pytest.mark.parametrize("group,folder", CASES)
def test_old_path_is_the_new_module(group, folder):
    new = importlib.import_module(f"dense_armor.utility.{group}.{folder}")
    old = importlib.import_module(f"dense_armor.utility.{folder}")
    assert old is new


def test_old_submodule_is_the_same_object():
    new = importlib.import_module("dense_armor.utility.online.bandit.ucb")
    old = importlib.import_module("dense_armor.utility.bandit.ucb")
    assert old is new
