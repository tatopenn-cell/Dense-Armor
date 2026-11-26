"""catalog: read the classes of Dense-Armor from its source."""

from __future__ import annotations

import ast
import builtins
import importlib.util
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def default_repo() -> Path:
    """Where the primitives are read from: USABLE_REPO, else the installed
    Dense-Armor package. No path that exists on one machine is written here
    (check 5 of ``04_SELF_CHECK.md``).
    """

    env = os.environ.get("USABLE_REPO")
    if env:
        p = Path(env)
        if not p.is_dir():
            raise RuntimeError(f"USABLE_REPO is not a directory: {p}")
        return p
    spec = importlib.util.find_spec("dense_armor")
    if spec and spec.origin:
        return Path(spec.origin).resolve().parent.parent
    raise RuntimeError(
        "Cannot find Dense-Armor: set USABLE_REPO or install the package."
    )


REPO = default_repo()

ABSTRACT = {
    "Estimator",
    "Classifier",
    "Regressor",
    "Transformer",
    "AnomalyDetector",
    "DriftDetector",
    "Root",
    "Signal",
}

ROLES = ABSTRACT - {"Estimator", "Root", "Signal"}

SKIP_SUFFIX = ("Config",)

NEURAL_SECTIONS = (
    "layers",
    "attention",
    "transformer",
    "losses",
    "training",
    "online",
    "models",
)


def audio_functions(repo: Path) -> list[dict]:
    """The public functions of the audio modules, which are most of what the audio domain offers.

    The rest of the catalog lists classes; in audio the primitives are mostly functions (``stft``,
    ``lowpass``, ``mfcc``), so they are listed too, with ``kind`` equal to ``function``."""
    rows = []
    for p in sorted((repo / "dense_armor" / "core" / "audio").glob("*.py")):
        if p.name.startswith("_"):
            continue
        try:
            t = ast.parse(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        module = ".".join(p.relative_to(repo).with_suffix("").parts)
        for n in t.body:
            if not isinstance(n, ast.FunctionDef) or n.name.startswith("_"):
                continue
            pos = [a.arg for a in n.args.args]
            required = pos[: len(pos) - len(n.args.defaults)]
            doc = (ast.get_docstring(n) or "").strip()
            rows.append(
                {
                    "name": n.name,
                    "class": n.name,
                    "kind": "function",
                    "domain": "audio",
                    "role": None,
                    "module": module,
                    "supervised": False,
                    "consumes": "sound",
                    "budget_s": None,
                    "memory": None,
                    "required": required,
                    "methods": [],
                    "doc": doc.split("\n")[0][:80],
                    "doc_full": " ".join(doc.split("\n\n")[0].split())[:600],
                }
            )
    return rows


def build_catalog(repo: Path) -> list[dict]:
    """Every public class with its role, domain, budget and what it reads. Classes that share a name in two
    places (core and ml) are both kept; the name then carries the folder: ``TFIDF [ml]``."""
    classes = []
    idx: dict[str, tuple[Path, ast.ClassDef]] = {}
    for p in sorted((repo / "dense_armor").rglob("*.py")):
        if "__pycache__" in p.parts or (
            p.name.startswith("_") and p.name != "__init__.py"
        ):
            continue
        rel_parts = p.relative_to(repo).parts
        if (
            len(rel_parts) >= 2
            and rel_parts[0] == "dense_armor"
            and rel_parts[1] == "usable"
        ):
            continue
        try:
            t = ast.parse(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for n in t.body:
            if isinstance(n, ast.ClassDef):
                classes.append((p, n))
                idx.setdefault(n.name, (p, n))

    def lit(node, name):
        for s in node.body:
            if isinstance(s, (ast.Assign, ast.AnnAssign)):
                tg = s.targets[0] if isinstance(s, ast.Assign) else s.target
                if isinstance(tg, ast.Name) and tg.id == name and s.value is not None:
                    try:
                        return ast.literal_eval(s.value)
                    except Exception:
                        return None
        return None

    def bases(node):
        for b in node.bases:
            bn = b.id if isinstance(b, ast.Name) else getattr(b, "attr", None)
            if bn:
                yield bn

    def up(node, read, seen=()):
        value = read(node)
        if value is not None:
            return value
        for bn in bases(node):
            if bn in idx and bn not in seen:
                value = up(idx[bn][1], read, seen + (bn,))
                if value is not None:
                    return value
        return None

    def is_exception(node, seen=()):
        """True for an exception or a warning: a base that is a built-in exception, directly or through a class of the package.
        The name of the class is not read: MeanAbsoluteError is a metric, and so is the class that derives from it."""
        for bn in bases(node):
            builtin = getattr(builtins, bn, None)
            if isinstance(builtin, type) and issubclass(builtin, BaseException):
                return True
            if bn in idx and bn not in seen and is_exception(idx[bn][1], seen + (bn,)):
                return True
        return False

    def role_of(node, seen=()):
        for bn in bases(node):
            if bn in ROLES:
                return bn
            if bn in idx and bn not in seen:
                r = role_of(idx[bn][1], seen + (bn,))
                if r:
                    return r
        return None

    def methods_of(node, seen=()):
        found = {
            n.name
            for n in node.body
            if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")
        }
        for bn in bases(node):
            if bn in idx and bn not in seen:
                found |= methods_of(idx[bn][1], seen + (bn,))
        return found

    def required_args(node):
        for s in node.body:
            if isinstance(s, ast.FunctionDef) and s.name == "__init__":
                a = s.args
                pos = [x.arg for x in a.args][1:]
                return pos[: len(pos) - len(a.defaults)]
        return []

    rows = []
    for p, n in classes:
        name = n.name
        if name.startswith("_") or name in ABSTRACT or name.endswith(SKIP_SUFFIX) or is_exception(n):
            continue
        rel = p.relative_to(repo).parts
        sub = rel[rel.index("core") + 1 : -1] if "core" in rel else rel[1:-1]
        domain = sub[-1] if sub else "core"
        if len(sub) >= 2 and sub[-2] == "neural" and domain in NEURAL_SECTIONS:
            domain = "neural"
        role = role_of(n)
        if domain in ("roles", "checks", "mcp_server") and not role:
            continue
        module = ".".join(p.relative_to(repo).with_suffix("").parts)
        sup = up(n, lambda nd: lit(nd, "_supervised"))
        supervised = (
            bool(sup) if sup is not None else role in ("Classifier", "Regressor")
        )
        if role == "DriftDetector":
            consumes = "scalar"
        elif domain == "audio":
            consumes = "sound"
        elif domain == "vision":
            consumes = "frames"
        elif "preprocessing.text" in module or "preprocessing.vectorizer" in module:
            consumes = "text"
        else:
            consumes = "rows"
        rows.append(
            {
                "name": name,
                "class": name,
                "kind": "class",
                "domain": domain,
                "role": role,
                "module": module,
                "supervised": supervised,
                "consumes": consumes,
                "budget_s": up(n, lambda nd: lit(nd, "budget_s")),
                "memory": up(n, lambda nd: lit(nd, "memory_class")),
                "required": required_args(n),
                "methods": sorted(methods_of(n)),
                "doc": (ast.get_docstring(n) or "").strip().split("\n")[0][:80],
                "doc_full": " ".join(
                    (ast.get_docstring(n) or "").strip().split("\n\n")[0].split()
                )[:600],
            }
        )
    rows.extend(audio_functions(repo))
    count: dict[str, int] = {}
    for r in rows:
        count[r["name"]] = count.get(r["name"], 0) + 1
    for r in rows:
        if count[r["name"]] > 1:
            r["name"] = f"{r['class']} [{r['module'].split('.')[1]}]"
    seen: dict[str, int] = {}
    for r in rows:
        seen[r["name"]] = seen.get(r["name"], 0) + 1
    for r in rows:
        if seen[r["name"]] > 1:
            r["name"] = f"{r['class']} [{r['module'].removeprefix('dense_armor.')}]"
    return rows


def get_catalog(repo: Path, refresh: bool = False) -> list[dict]:
    """The catalog, read from catalog.json when the repository is at the same commit."""
    cache = HERE / "catalog.json"
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except Exception:
        sha = ""
    if cache.exists() and not refresh and sha:
        d = json.loads(cache.read_text(encoding="utf-8"))
        if d.get("sha") == sha and d.get("v") == 9:
            return d["rows"]
    rows = build_catalog(repo)
    if sha:
        cache.write_text(
            json.dumps({"v": 9, "sha": sha, "rows": rows}), encoding="utf-8"
        )
    return rows
