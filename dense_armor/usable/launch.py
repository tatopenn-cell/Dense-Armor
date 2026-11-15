"""Start usable without running ``dense_armor/__init__.py``.

Importing ``dense_armor`` loads JAX (about 1.8 s). The commands ``list``,
``select``, ``help``, the prompt and the readers do not need the library at
run time: they read its sources with ``ast``. This launcher puts an empty
``dense_armor`` package (same path) in ``sys.modules``, so ``dense_armor.usable``
imports without the rest. ``ensure_real`` swaps it for the real package right
before the three places that call the library (audio, text, image features).
The library itself is not changed.
"""

from __future__ import annotations

import sys
from pathlib import Path

# started as a script, sys.path[0] is this folder, where select.py hides the standard one
sys.path[:] = [
    p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent
]

import importlib.machinery  # noqa: E402
import types  # noqa: E402


def _install_stub() -> None:
    if "dense_armor" in sys.modules:
        return
    root = str(Path(__file__).resolve().parents[1])
    parent = str(Path(root).parent)
    if parent in sys.path:
        sys.path.remove(parent)
    sys.path.insert(0, parent)
    pkg = types.ModuleType("dense_armor")
    pkg.__path__ = [root]
    spec = importlib.machinery.ModuleSpec(
        "dense_armor", None, origin=str(Path(root) / "__init__.py"), is_package=True
    )
    spec.submodule_search_locations = [root]
    pkg.__spec__ = spec
    pkg._usable_stub = True  # type: ignore[attr-defined]
    sys.modules["dense_armor"] = pkg


def ensure_real() -> None:
    """Replace the stub with the real ``dense_armor`` (a no-op if it is real)."""
    pkg = sys.modules.get("dense_armor")
    if pkg is None or not getattr(pkg, "_usable_stub", False):
        return
    del sys.modules["dense_armor"]
    for name in [n for n in sys.modules if n.startswith("dense_armor.")]:
        if not name.startswith("dense_armor.usable"):
            del sys.modules[name]  # stand-ins made to read the ontology by path
    import dense_armor  # noqa: F401


def main() -> int:
    _install_stub()
    from dense_armor.usable.__main__ import main as entry
    from dense_armor.usable.cli import run

    run(entry)
    return 0


if __name__ == "__main__":
    sys.exit(main())
