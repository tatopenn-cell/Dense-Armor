"""Write ``docs/offline_algorithms/<family>.md`` from the docstrings and the JSON.

    python benchmarks/offline/make_page.py FAMILY

The two sections (classic, online) show the doctest example of each class with
the output of a real run; the table of the difference is the JSON of the family.
"""

from __future__ import annotations

import ast
import contextlib
import doctest
import importlib
import inspect
import io
import json
import pkgutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

INTRO = json.loads((Path(__file__).with_name("intro.json")).read_text(encoding="utf-8"))


def _online_class(family: str, name: str):
    pkg = importlib.import_module(f"dense_armor.utility.{family}")
    for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        mod = importlib.import_module(m.name)
        cls = getattr(mod, name, None)
        if inspect.isclass(cls) and cls.__module__ == mod.__name__:
            return cls
    raise LookupError(name)


def _block(cls) -> tuple[str, str]:
    """Self-contained code and the real output of the doctest example of ``cls``."""
    doc = inspect.getdoc(cls) or ""
    exs = [e for e in doctest.DocTestParser().get_examples(doc)]
    lines = [f"from {cls.__module__} import {cls.__name__}"]
    for e in exs:
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
    mod = importlib.import_module(f"dense_armor.offline.{family}")
    data = json.loads(
        (ROOT / "benchmarks" / "offline" / f"{family}.json").read_text(encoding="utf-8")
    )
    names = [n for n in mod.__all__]
    out = [
        f"# Classic and online {family}: what is the difference?\n",
        INTRO[family] + "\n",
    ]
    out.append("## 1. Classic: the algorithm of the paper on the whole dataset\n")
    out.append(
        "Each class reads the whole dataset with `fit` and leaves its results in attributes with a trailing underscore.\n"
    )
    for n in names:
        cls = getattr(mod, n)
        first = (inspect.getdoc(cls) or "").split("\n")[0]
        code, res = _block(cls)
        out.append(
            f"### `{n}`\n\n{first}\n\n~~~python\n{code}\n~~~\n\n~~~\n{res}\n~~~\n"
        )
    out.append("## 2. Online: one sample at a time\n")
    out.append(
        "The same quantity from `learn_one`; the class is the one of `dense_armor.utility`.\n"
    )
    for n in names:
        on = n.removeprefix("Classic")
        try:
            cls = _online_class(family, on)
        except LookupError:
            continue
        first = (inspect.getdoc(cls) or "").split("\n")[0]
        try:
            code, res = _block(cls)
        except Exception:
            continue
        out.append(
            f"### `{on}`\n\n{first}\n\n~~~python\n{code}\n~~~\n\n~~~\n{res}\n~~~\n"
        )
    out.append("## 3. The difference, on synthetic data\n")
    out.append(
        f"{data['seeds']} seeds. `abs` is the largest absolute difference between the two outputs, `rel` the largest relative difference. Median, 90th percentile and maximum over the seeds; the data are `synthetic`.\n"
    )
    out.append("| class | kind | median | p90 | max | why |\n|---|---|---|---|---|---|")
    for n, r in data["classes"].items():
        out.append(
            f"| `{n}` | {r['kind']} | {r['median']:.3g} | {r['p90']:.3g} | {r['max']:.3g} | {r['why']} |"
        )
    out.append(f"\nWilcoxon and A12: {data['wilcoxon_a12']}.\n")
    out.append("## Details\n\n### References\n")
    for n in names:
        doc = inspect.getdoc(getattr(mod, n)) or ""
        if "Eq" in doc or "page" in doc:
            out.append(f"- `{n}`: " + " ".join(doc.split())[:300])
    mdoc = inspect.getdoc(mod) or ""
    out.append(
        "\n" + mdoc.split("References:")[-1].strip() + "\n"
        if "References:" in mdoc
        else ""
    )
    out.append(
        "### Choices stated explicitly\n\nThe classic class of an online sketch is the exact value the sketch approximates; the classic class of a statistic is the textbook formula on the whole array. Both are `our choice`, because the online classes name no batch algorithm.\n"
    )
    out.append(
        "### Development diary\n\nThe classic classes are new; the online classes are unchanged. The numbers of section 3 come from `benchmarks/offline/"
        + family
        + ".py`.\n"
    )
    dest = ROOT / "docs" / "offline_algorithms"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{family}.md").write_text("\n".join(out), encoding="utf-8")
    print("wrote", dest / f"{family}.md")


if __name__ == "__main__":
    main(sys.argv[1])
