"""The folders of dense_armor/core sit in the declared groups, and the old paths still work.

The library is ``dense_armor.core`` and the shielding engine is ``dense_armor.utility``. Before
this swap the library was ``dense_armor.utility`` and the engine ``dense_armor.core``: every
old path (the flat ones, the ``utility.<group>.<folder>`` of the last release, and the engine
modules under ``dense_armor.core``) still resolves to the same module object.

The set of groups is the one in ``UTILITY_GROUPS``; the test compares the
folders of ``dense_armor/core/`` with that set, not with a fixed number.
A group with no folder (``finance``, ``audio``) is a valid entry: it has no subfolder
to alias, and ``test_every_folder_is_in_its_group`` does not produce a case
for it.
"""

import importlib
from pathlib import Path

import pytest

import dense_armor
from dense_armor._compat import UTILITY_GROUPS

CASES = [(g, f) for g, fs in UTILITY_GROUPS.items() for f in fs]
CORE = Path(dense_armor.__file__).parent / "core"
ENGINE = ("damping_operator", "engine", "hybrid_engine", "memory", "noise", "preset")


def test_core_holds_only_the_declared_groups():
    folders = {
        p.name
        for p in CORE.iterdir()
        if p.is_dir() and p.name not in ("__pycache__",)
    }
    assert folders == set(UTILITY_GROUPS)


@pytest.mark.parametrize("group,folder", CASES)
def test_every_folder_is_in_its_group(group, folder):
    assert (CORE / group / folder / "__init__.py").is_file()


@pytest.mark.parametrize("group,folder", CASES)
def test_old_path_is_the_new_module(group, folder):
    new = importlib.import_module(f"dense_armor.core.{group}.{folder}")
    old = importlib.import_module(f"dense_armor.utility.{folder}")
    assert old is new


@pytest.mark.parametrize("group,folder", CASES)
def test_path_of_the_last_release_is_the_new_module(group, folder):
    new = importlib.import_module(f"dense_armor.core.{group}.{folder}")
    last = importlib.import_module(f"dense_armor.utility.{group}.{folder}")
    assert last is new


def test_old_submodule_is_the_same_object():
    new = importlib.import_module("dense_armor.core.online.bandit.ucb")
    old = importlib.import_module("dense_armor.utility.bandit.ucb")
    assert old is new


@pytest.mark.parametrize("module", ENGINE)
def test_engine_module_old_path_is_the_new_module(module):
    new = importlib.import_module(f"dense_armor.utility.{module}")
    old = importlib.import_module(f"dense_armor.core.{module}")
    assert old is new
    assert (Path(dense_armor.__file__).parent / "utility" / f"{module}.py").is_file()


MISC_MODULES = {
    "sdft": "signals.stats.sdft",
    "skyline": "signals.stats.skyline",
    "metro": "signals.preprocessing.metro",
    "diagnostic": "evaluation.evaluate.diagnostic",
    "iodat": "signals.stream.iodat",
}
MISC_PREFIXES = ("dense_armor.utility.misc", "dense_armor.utility.infrastructure.misc")


@pytest.mark.parametrize("prefix", MISC_PREFIXES)
@pytest.mark.parametrize("name,new", MISC_MODULES.items())
def test_old_misc_module_is_the_module_in_its_new_home(prefix, name, new):
    old = importlib.import_module(f"{prefix}.{name}")
    assert old is importlib.import_module(f"dense_armor.core.{new}")


@pytest.mark.parametrize("prefix", MISC_PREFIXES)
def test_old_misc_package_still_gives_sdft_and_skyline(prefix):
    from dense_armor.core.signals.stats.sdft import SDFT
    from dense_armor.core.signals.stats.skyline import Skyline

    old = importlib.import_module(prefix)
    assert old.SDFT is SDFT
    assert old.Skyline is Skyline
