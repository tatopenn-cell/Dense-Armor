"""Write ``docs/ml/<family>.md`` from the docstrings of the classes.

    python benchmarks/ml/make_page.py FAMILY

Each class shows the doctest example of its docstring with the output of a real run.
"""

from __future__ import annotations

import ast
import contextlib
import doctest
import importlib
import inspect
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

INTRO = json.loads(Path(__file__).with_name("intro.json").read_text(encoding="utf-8"))


def _block(cls) -> tuple[str, str]:
    """Self-contained code and the real output of the doctest example of ``cls``."""
    doc = inspect.getdoc(cls) or ""
    lines = [f"from {cls.__module__} import {cls.__name__}"]
    for e in doctest.DocTestParser().get_examples(doc):
        src = e.source.rstrip("\n")
        try:
            tree = ast.parse(src)
            if e.want and len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
                src = f"print({src})"
        except SyntaxError:
            pass
        lines.append(src)
    code = "\n".join(lines)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(code, "<page>", "exec"), {"__name__": "__page__"})
    return code, buf.getvalue().rstrip("\n")


def main(family: str) -> None:
    mod = importlib.import_module(f"dense_armor.ml.{family}")
    out = [f"# {family.capitalize()}: the algorithms\n", INTRO[family] + "\n"]
    out.append(
        "Every class reads the whole dataset with `fit` and leaves its results in attributes with a trailing underscore; `transform` or a query method applies them. The data are `synthetic` where a block makes them up.\n"
    )
    for i, n in enumerate(mod.__all__, 1):
        cls = getattr(mod, n)
        doc = inspect.getdoc(cls) or ""
        first = doc.split("\n")[0].replace("``", "`")
        code, res = _block(cls)
        out.append(
            f"## {i}. `{n}`\n\n{first}\n\n~~~python\n{code}\n~~~\n\n~~~\n{res}\n~~~\n"
        )
    out.append("## Details\n\n### References\n")
    mdoc = inspect.getdoc(mod) or ""
    out.append(
        mdoc.split("References:")[-1].strip().replace("``", "`") + "\n"
        if "References:" in mdoc
        else "The formulas are textbook definitions.\n"
    )
    stopped = getattr(mod, "STOPPED", {})
    out.append(
        "### Choices stated explicitly\n\nThe class of a sketch is the exact value the sketch approximates, and the class of a statistic is the textbook formula on the whole array: both are `our choice`, because the references name no other batch algorithm.\n"
    )
    out.append(
        "### Development diary\n\nThe classes are new and independent of the online classes of dense_armor.utility.\n"
    )
    if stopped:
        out.append(
            "Classes without a form:\n"
            + "\n".join(f"- `{k}`: {v}" for k, v in stopped.items())
            + "\n"
        )
    dest = ROOT / "docs" / "ml"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{family}.md").write_text("\n".join(out), encoding="utf-8")
    print("wrote", dest / f"{family}.md")


if __name__ == "__main__":
    main(sys.argv[1])
