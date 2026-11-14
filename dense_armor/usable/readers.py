"""Read a file or a web address and say what it is.

Kinds: a table (CSV, TSV, JSON, JSON lines), a sound (WAV), a text (TXT, MD, HTML, PDF, ...), a picture or a
video. Anything else is reported as unsupported, with the reason: the answer is never empty.
The standard library is enough; pypdf (PDF) and Pillow (picture size) are used when they are installed.
"""

import ast
import csv
import html.parser
import io
import json
import keyword
import math
import re
import tempfile
import tokenize
import urllib.error
import urllib.parse
import urllib.request
import wave
from datetime import datetime
from pathlib import Path
from typing import Any

TABLE_EXT = {".csv", ".tsv", ".json", ".jsonl", ".ndjson"}
TEXT_EXT = {".txt", ".md", ".rst", ".log", ".html", ".htm", ".xml", ".pdf"}
CODE_EXT = {".py", ".ipynb"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
CONVERT_SOUND = {".mp3", ".flac", ".ogg", ".m4a", ".aac"}
MAX_URL_BYTES = 8_000_000
MAX_TEXT_BYTES = 5_000_000

SUPPORTED = (
    "CSV, TSV, JSON, JSON lines (tables); WAV (sound); TXT, MD, HTML, PDF (text); PY, IPYNB (code); "
    "PNG, JPG, MP4, AVI (pictures and video); http and https addresses of these"
)

LABEL_NAMES = {
    "y",
    "label",
    "target",
    "class",
    "is_fraud",
    "fraud",
    "anomaly",
    "outlier",
    "attack",
    "default",
}
TIME_NAMES = re.compile(r"^(t|ts|time|timestamp|date|datetime|seconds|sec)$", re.I)
ROBOT_NAMES = re.compile(
    r"(joint|^q\d|^qd\d|tau|torque|velocity|position|effort)", re.I
)
FINANCE_NAMES = re.compile(
    r"(price|order|qty|quantity|notional|side|symbol|ticker|volume|amount)", re.I
)


# ---------- addresses ----------


def is_url(text: str) -> bool:
    return text.lower().startswith(("http://", "https://"))


def _open(req, insecure: bool):
    """The certificate is checked: with the system store, then with certifi. Skipping the check is opt-in."""
    import ssl

    if insecure:
        ctx = ssl._create_unverified_context()  # noqa: S323 - asked for with --insecure
        return urllib.request.urlopen(req, timeout=15, context=ctx)  # noqa: S310
    try:
        return urllib.request.urlopen(req, timeout=15)  # noqa: S310
    except urllib.error.URLError as exc:
        if not isinstance(getattr(exc, "reason", None), ssl.SSLCertVerificationError):
            raise
        try:
            import certifi
        except ImportError:
            raise ValueError(
                "the certificate of the site could not be verified; if you trust the site, "
                "ask for the unverified download (--insecure, or the box in the window)"
            ) from exc
        try:
            return urllib.request.urlopen(
                req,
                timeout=15,
                context=ssl.create_default_context(cafile=certifi.where()),
            )  # noqa: S310
        except urllib.error.URLError as exc2:
            raise ValueError(
                "the certificate of the site could not be verified; if you trust the site, "
                "ask for the unverified download (--insecure, or the box in the window)"
            ) from exc2


def fetch(url: str, insecure: bool = False) -> tuple[Path, str]:
    """Download an address to a temporary file (at most 8 MB, 15 s). Returns the path and a note."""
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)/blob/(.+)", url)
    if m:
        url = (
            f"https://raw.githubusercontent.com/{m.group(1)}/{m.group(2)}/{m.group(3)}"
        )
    req = urllib.request.Request(url, headers={"User-Agent": "usable/0.1"})
    with _open(req, insecure) as resp:
        if urllib.parse.urlparse(resp.geturl()).scheme not in ("http", "https"):
            raise ValueError(
                "the address redirects to something that is not http or https"
            )
        data = resp.read(MAX_URL_BYTES + 1)
        ctype = (resp.headers.get_content_type() or "").lower()
    if len(data) > MAX_URL_BYTES:
        raise ValueError(f"the address is larger than {MAX_URL_BYTES // 1_000_000} MB")
    name = Path(urllib.parse.urlparse(url).path).name or "page"
    suffix = Path(name).suffix.lower()
    by_type = {
        "text/html": ".html",
        "application/pdf": ".pdf",
        "text/csv": ".csv",
        "application/json": ".json",
        "text/plain": ".txt",
        "text/x-python": ".py",
        "application/x-ipynb+json": ".ipynb",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "video/mp4": ".mp4",
    }
    if suffix not in TABLE_EXT | TEXT_EXT | CODE_EXT | IMAGE_EXT | VIDEO_EXT | {".wav"}:
        name += by_type.get(ctype, "")
    base = Path(__file__).resolve().parent / "tmp"
    base.mkdir(exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="usable_", dir=base))
    path = folder / (re.sub(r"[^\w.\-]", "_", name) or "page")
    path.write_bytes(data)
    note = f"downloaded {len(data) / 1000:.0f} KB from {urllib.parse.urlparse(url).netloc} ({ctype or 'unknown type'})"
    return path, note + (", certificate not checked" if insecure else "")


# ---------- tables ----------


def _num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _infer_type(values: list[str]) -> str:
    """Best type for a column, from the non-missing string values.

    Order: int, float, datetime, string. A column of "0"/"1" is int;
    "0.0"/"1.5" is float; an ISO date is datetime.
    """
    if not values:
        return "string"
    for typ, parse in (
        ("int", int),
        ("float", float),
        ("datetime", datetime.fromisoformat),
    ):
        try:
            [parse(x) for x in values]
            return typ
        except (ValueError, TypeError):
            continue
    return "string"


def _semantic(
    col_type: str,
    n_non_missing: int,
    n_total: int,
    missing_frac: float,
    unique: int,
    unique_ratio: float,
    non_missing: list[str],
) -> str:
    """Best semantic hint: id, category, measure, datetime, constant,
    text, missing-heavy, unknown."""
    if missing_frac > 0.5:
        return "missing-heavy"
    if unique == 1:
        return "constant"
    if col_type == "datetime":
        return "datetime"
    if col_type == "string":
        mean_len = sum(len(x) for x in non_missing) / max(len(non_missing), 1)
        if mean_len > 100:
            return "text"
    if unique_ratio < 0.05:
        return "category"
    if len(set(non_missing)) == n_non_missing:
        return "id"
    if col_type in ("int", "float"):
        return "measure"
    return "text"


def _range(values: list, col_type: str) -> dict | None:
    """min, max, mean, median, std for a numeric column; None otherwise."""
    if not values:
        return None
    try:
        import statistics as _st

        out = {
            "min": min(values),
            "max": max(values),
            "mean": _st.mean(values),
            "median": _st.median(values),
            "std": _st.pstdev(values) if len(values) > 1 else 0.0,
        }
        if col_type == "int":
            out = {
                k: (int(v) if isinstance(v, float) and v == int(v) else v)
                for k, v in out.items()
            }
        return out
    except (TypeError, ValueError):
        return None


def _profile_table(names: list[str], body: list[list[str]]) -> dict:
    """Profile a table: one dict per column, with type, semantic, range.

    Type is inferred from the non-missing values (int, float, datetime,
    string). Semantic is id, category, measure, datetime, constant, text,
    or missing-heavy. Numeric columns carry range (min, max, mean,
    median, std); datetime columns carry range_t (min, max, range_s).
    String columns carry lengths. Every column carries top3.
    """
    n = len(body)
    missing_tokens = {"", "nan", "na", "null", "none", "n/a"}
    cols = []
    for j, name in enumerate(names):
        raw = [r[j].strip() if j < len(r) else "" for r in body]
        missing = sum(1 for x in raw if x.lower() in missing_tokens)
        non_missing = [x for x in raw if x.lower() not in missing_tokens]
        col_type = _infer_type(non_missing)
        if col_type == "int":
            parsed: list = [int(x) for x in non_missing]
        elif col_type == "float":
            parsed = [float(x) for x in non_missing]
        elif col_type == "datetime":
            parsed = [datetime.fromisoformat(x) for x in non_missing]
        else:
            parsed = list(non_missing)
        unique = len(set(raw))
        unique_ratio = unique / max(n, 1)
        mono = len(parsed) > 2 and all(b >= a for a, b in zip(parsed, parsed[1:]))
        dt = None
        if mono and len(parsed) > 2:
            if col_type == "datetime":
                steps = sorted(
                    (b - a).total_seconds() for a, b in zip(parsed, parsed[1:]) if b > a
                )
            else:
                steps = sorted(b - a for a, b in zip(parsed, parsed[1:]) if b > a)
            if steps:
                dt = float(f"{steps[len(steps) // 2]:.6g}")
        semantic = _semantic(
            col_type,
            len(non_missing),
            n,
            missing / max(n, 1),
            unique,
            unique_ratio,
            non_missing,
        )
        range_num = _range(parsed, col_type) if col_type in ("int", "float") else None
        range_t = None
        if col_type == "datetime" and parsed:
            mn, mx = min(parsed), max(parsed)
            range_t = {
                "min": mn.isoformat(),
                "max": mx.isoformat(),
                "range_s": (mx - mn).total_seconds(),
            }
        lengths = None
        if col_type == "string" and non_missing:
            ls = [len(x) for x in non_missing]
            lengths = {
                "min": int(min(ls)),
                "max": int(max(ls)),
                "mean": round(sum(ls) / len(ls), 2),
            }
        from collections import Counter as _Counter

        top3 = [[v, int(c)] for v, c in _Counter(raw).most_common(3) if v]
        cols.append(
            {
                "name": name,
                "type": col_type,
                "semantic": semantic,
                "missing": missing / max(n, 1),
                "unique": unique,
                "unique_ratio": round(unique_ratio, 4),
                "monotonic": mono,
                "dt": dt,
                "range": range_num,
                "range_t": range_t,
                "lengths": lengths,
                "top3": top3,
                "numeric": col_type in ("int", "float"),
                "date": col_type == "datetime",
            }
        )
    from . import keywords as K

    words = []
    for name in names:
        for w in re.findall(
            r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+",
            name.replace("_", " "),
        ):
            w = w.lower()
            if len(w) >= 3 and w not in words:
                words.append(w)
    glossary = K._glossary()
    lang = "it" if any(K._to_english(w, glossary) for w in words) else "en"
    return {
        "kind": "table",
        "rows": n,
        "columns": cols,
        "keywords": [[w, 1] for w in words],
        "lang": lang,
    }


def profile_csv(path: Path) -> dict:
    with open(path, newline="", encoding="utf-8", errors="ignore") as fh:
        sample = fh.read(8192)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        rows = list(csv.reader(fh, dialect))
    if not rows:
        raise ValueError("the file is empty")
    header = rows[0]
    has_header = any(_num(c) is None and c.strip() for c in header)
    names = (
        [c.strip() for c in header]
        if has_header
        else [f"c{i}" for i in range(len(header))]
    )
    return _profile_table(names, rows[1:] if has_header else rows)


def _records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() in (".jsonl", ".ndjson"):
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return value
        return [data]
    return []


def profile_json(path: Path) -> dict:
    records = _records(path)
    if not records:
        return {
            "kind": "unsupported",
            "reason": "the JSON holds no list of records (objects with the same keys)",
        }
    keys = []
    for r in records[:2000]:
        for k, v in r.items():
            if not isinstance(v, (dict, list)) and k not in keys:
                keys.append(k)
    body = [["" if r.get(k) is None else str(r.get(k)) for k in keys] for r in records]
    return _profile_table(keys, body)


# ---------- sound, text, pictures ----------


def profile_wav(path: Path) -> dict:
    with wave.open(str(path), "rb") as w:
        return {
            "kind": "sound",
            "rows": w.getnframes(),
            "rate": w.getframerate(),
            "channels": w.getnchannels(),
            "seconds": w.getnframes() / w.getframerate(),
        }


class _Strip(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self._skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


STOP = set(
    """the and for that with this from are was were have has had not but you your our their which these those
there been being than then them they will would can could also such into more most other some any all one two
della delle degli dello dei del che per con non una uno gli nel nella sono come anche questo questa suo sua tra fra vanno quando dove ogni solo molto fare fatto essere stato sempre prima dopo poi quindi invece mentre cosa cose sulla sulle sugli delle alla alle allo agli dalla dalle dallo dagli ancora molto tutto tutti quella quello quelli quelle puoi vuoi devi deve hanno viene vengono sia siano stati stata state""".split()
)
NUMBER = re.compile(r"^[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?%?$")


def _extract_text(path: Path, as_pdf: bool) -> tuple[str, int | None, str | None]:
    ext = path.suffix.lower()
    if ext == ".pdf" or as_pdf:
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            pages = len(reader.pages)
            return (
                "\n".join((pg.extract_text() or "") for pg in reader.pages[:200]),
                pages,
                None,
            )
        except ImportError:
            return "", None, "pypdf is not installed, so the PDF was not read"
        except Exception as exc:
            return "", None, f"the PDF could not be read ({type(exc).__name__})"
    raw = path.read_bytes()[:MAX_TEXT_BYTES].decode("utf-8", errors="ignore")
    if ext in (".html", ".htm", ".xml"):
        raw = re.sub(
            r"\{%.*?%\}|\{#.*?#\}", " ", raw, flags=re.S
        )  # template statements, not words
        strip = _Strip()
        strip.feed(raw)
        raw = " ".join(strip.parts)
    return raw, None, None


def _tables(text: str) -> list[list[list[float]]]:
    """Blocks of at least 4 consecutive lines that hold the same number (3 or more) of numeric tokens."""
    blocks: list[list[list[float]]] = []
    current: list[list[float]] = []
    for line in text.splitlines():
        nums = [
            float(t.replace(",", ".").rstrip("%"))
            for t in line.split()
            if NUMBER.match(t)
        ]
        if len(nums) >= 3 and (not current or len(nums) == len(current[-1])):
            current.append(nums)
        else:
            if len(current) >= 4:
                blocks.append(current)
            current = [nums] if len(nums) >= 3 else []
    if len(current) >= 4:
        blocks.append(current)
    return blocks


def profile_text(path: Path, as_pdf: bool = False) -> dict:
    text, pages, note = _extract_text(path, as_pdf)
    words = re.findall(r"\w+", text)
    sentences = [
        x
        for x in re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
        if len(x.split()) >= 4
    ]
    terms: dict[str, int] = {}
    for w in re.findall(r"[A-Za-zÀ-ÿ]{4,}", text):
        w = w.lower()
        if w not in STOP:
            terms[w] = terms.get(w, 0) + 1
    ranked = sorted(terms.items(), key=lambda kv: -kv[1])
    top = [w for w, _ in ranked[:8]]
    keywords = [[w, c] for w, c in ranked[:40]]
    from . import keywords as K

    lang = K.detect_language(text)
    numeric = sum(1 for t in text.split() if NUMBER.match(t))
    blocks = _tables(text)
    table = None
    if blocks:
        best = max(blocks, key=lambda b: len(b) * len(b[0]))
        names = [f"col{k + 1}" for k in range(len(best[0]))]
        table = _profile_table(names, [[str(v) for v in row] for row in best])
    return {
        "kind": "text",
        "rows": len(sentences),
        "sentences": len(sentences),
        "words": len(words),
        "pages": pages,
        "type": path.suffix.lower().lstrip(".") or "text",
        "note": note,
        "terms": top,
        "numeric_share": numeric / max(len(text.split()), 1),
        "tables": len(blocks),
        "table": table,
        "keywords": keywords,
        "lang": lang,
    }


def profile_vision(path: Path) -> dict:
    ext = path.suffix.lower()
    out = {
        "kind": "vision",
        "rows": 1,
        "video": ext in VIDEO_EXT,
        "type": ext.lstrip("."),
        "width": None,
        "height": None,
    }
    if ext in IMAGE_EXT:
        try:
            from PIL import Image

            with Image.open(path) as im:
                out["width"], out["height"] = im.size
        except Exception:
            pass
    return out


# ---------- code ----------

# words found in the imports, the calls and the names of a source, and the job they point at
SIGNALS = {
    "classify": [
        "classifier",
        "logisticregression",
        "svc",
        "naivebayes",
        "decisiontree",
        "randomforest",
    ],
    "regress": ["regressor", "linearregression", "ridge", "lasso"],
    "detect_anomaly": [
        "isolationforest",
        "outlier",
        "anomaly",
        "localoutlier",
        "oneclasssvm",
        "mahalanobis",
    ],
    "detect_drift": [
        "adwin",
        "drift",
        "changepoint",
        "ruptures",
        "cusum",
        "pagehinkley",
    ],
    "forecast": [
        "arima",
        "statsmodels",
        "prophet",
        "kalman",
        "forecast",
        "exponentialsmoothing",
    ],
    "audio_features": [
        "librosa",
        "sounddevice",
        "soundfile",
        "wave",
        "pyaudio",
        "scipy.signal",
        "fft",
        "melspectrogram",
    ],
    "vision_features": ["cv2", "opencv", "pil", "torchvision", "imageio", "skimage"],
    "text_features": [
        "nltk",
        "spacy",
        "tfidf",
        "countvectorizer",
        "tokenizer",
        "gensim",
    ],
    "robot": ["urdf", "pybullet", "rospy", "rclpy", "moveit", "kinematics"],
    "finance_controls": ["ccxt", "yfinance", "backtrader", "portfolio"],
    "summarize_stream": ["rolling", "quantile", "percentile"],
    "resources": ["psutil"],
}
SHORT = {"cv2", "pil", "wave", "fft", "svc"}
CODE_STOP = set(keyword.kwlist) | {
    "self",
    "data",
    "return",
    "none",
    "true",
    "false",
    "print",
    "len",
    "range",
    "args",
    "kwargs",
    "main",
    "test",
    "value",
    "values",
    "result",
    "results",
    "item",
    "items",
}


def _split_name(name: str) -> list[str]:
    parts = re.findall(
        r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", name.replace("_", " ")
    )
    return [x.lower() for x in parts if len(x) >= 3]


def _decorated_with(n: ast.AST, word: str) -> bool:
    """True when a function or class carries a decorator whose name contains ``word``."""
    for d in getattr(n, "decorator_list", []):
        f = d.func if isinstance(d, ast.Call) else d
        name = (
            f.id
            if isinstance(f, ast.Name)
            else f.attr
            if isinstance(f, ast.Attribute)
            else ""
        )
        if word in name:
            return True
    return False


def profile_code(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".ipynb":
        cells = json.loads(raw).get("cells", [])
        raw = "\n\n".join(
            "".join(c.get("source", [])) for c in cells if c.get("cell_type") == "code"
        )
    out: dict[str, Any] = {
        "kind": "code",
        "rows": 0,
        "lines": len(raw.splitlines()),
        "imports": [],
        "calls": [],
        "functions": 0,
        "classes": 0,
        "loops": 0,
        "dense_armor": [],
        "syntax_error": None,
    }
    names: list[str] = []
    try:
        tree = ast.parse(raw)
    except SyntaxError as exc:
        tree = None
        out["syntax_error"] = f"line {exc.lineno}: {exc.msg}"
    imports, calls, uses = [], [], []
    if tree is not None:
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                imports += [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module:
                imports += [f"{n.module}.{a.name}" for a in n.names]
                if n.module.startswith("dense_armor"):
                    uses += [a.name for a in n.names]
            elif isinstance(n, ast.Call):
                f = n.func
                calls.append(
                    f.id
                    if isinstance(f, ast.Name)
                    else f.attr
                    if isinstance(f, ast.Attribute)
                    else ""
                )
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out["functions"] = int(out["functions"]) + 1
                names.append(n.name)
            elif isinstance(n, ast.ClassDef):
                out["classes"] = int(out["classes"]) + 1
                names.append(n.name)
            elif isinstance(n, (ast.For, ast.While)):
                out["loops"] = int(out["loops"]) + 1
            elif isinstance(n, ast.Name):
                names.append(n.id)
            elif isinstance(n, ast.Attribute):
                names.append(n.attr)
    defs = []
    if tree is not None:
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                calls = []
                for c in ast.walk(n):
                    if isinstance(c, ast.Call):
                        f = c.func
                        if isinstance(f, ast.Name):
                            calls.append(f.id)
                        elif isinstance(f, ast.Attribute):
                            calls.append(
                                f"{f.value.id}.{f.attr}"
                                if isinstance(f.value, ast.Name)
                                else f.attr
                            )
                is_class = isinstance(n, ast.ClassDef)
                if is_class:
                    methods = [
                        m
                        for m in n.body  # type: ignore[union-attr]
                        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ]
                    data_only = all(_decorated_with(m, "validator") for m in methods)
                else:
                    data_only = _decorated_with(n, "validator")
                defs.append(
                    {
                        "name": n.name,
                        "kind": "class" if is_class else "function",
                        "line": n.lineno,
                        "args": [] if is_class else [a.arg for a in n.args.args],  # type: ignore[union-attr]
                        "doc": (ast.get_docstring(n) or "")
                        .strip()
                        .split("\n")[0][:100],
                        "calls": sorted(set(calls))[:6],
                        "data_only": data_only,
                    }
                )
    out["defs"] = sorted(defs, key=lambda d: d["line"])
    groups: dict[str, list[str]] = {
        "stdlib": [],
        "installed": [],
        "local or not installed": [],
    }
    import importlib.util as _iu
    import sys as _sys

    for top in sorted({m.split(".")[0] for m in imports}):
        if top in _sys.stdlib_module_names:
            groups["stdlib"].append(top)
        else:
            try:
                found = _iu.find_spec(top) is not None
            except (ImportError, ValueError):
                found = False
            groups["installed" if found else "local or not installed"].append(top)
    out["import_groups"] = groups
    out["imports"] = sorted(set(imports))
    out["calls"] = sorted({c for c in calls if c})
    out["dense_armor"] = sorted(set(uses))
    out["rows"] = out["functions"]
    comments: list[str] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(raw).readline):
            if tok.type == tokenize.COMMENT:
                comments.append(tok.string.lstrip("# "))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    if tree is not None:
        comments += [
            d
            for d in (
                ast.get_docstring(n)
                for n in ast.walk(tree)
                if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef))
            )
            if d
        ]
    words: dict[str, int] = {}
    for d in list(out["defs"]):
        for w in _split_name(d["name"]) + [
            x.lower() for x in re.findall(r"[A-Za-z]{4,}", d["doc"])
        ]:
            if w not in CODE_STOP and w not in STOP:
                words[w] = words.get(w, 0) + 1
    for text in comments:
        for w in re.findall(r"[A-Za-zÀ-ÿ]{4,}", text):
            w = w.lower()
            if w not in CODE_STOP and w not in STOP:
                words[w] = words.get(w, 0) + 1
    from . import keywords as K

    out["keywords"] = [
        [w, c] for w, c in sorted(words.items(), key=lambda kv: -kv[1])[:40]
    ]
    out["lang"] = K.detect_language(" ".join(comments)) if comments else "en"
    haystack = [
        x.lower() for x in list(out["imports"]) + list(out["calls"]) + list(names)
    ]
    signals: dict[str, list[str]] = {}
    for need, pats in SIGNALS.items():
        for h in haystack:
            for pat in pats:
                if (h == pat) if pat in SHORT else (pat in h):
                    signals.setdefault(need, [])
                    if h not in signals[need]:
                        signals[need].append(h)
    out["signals"] = signals
    return out


# ---------- the entry point of the readers ----------


def _looks_text(sample: bytes) -> bool:
    try:
        text = sample.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return (
        bool(text)
        and sum(c.isprintable() or c in "\r\n\t" for c in text) / len(text) > 0.95
    )


def _looks_tabular(path: Path) -> bool:
    try:
        sample = path.read_bytes()[:8192].decode("utf-8", errors="ignore")
        lines = [ln for ln in sample.splitlines() if ln.strip()][:20]
        if len(lines) < 3:
            return False
        dialect = csv.Sniffer().sniff("\n".join(lines), delimiters=",;\t|")
        counts = {len(next(csv.reader([ln], dialect))) for ln in lines[1:]}
        return len(counts) == 1 and counts.pop() >= 2
    except csv.Error:
        return False


def profile(path: Path) -> dict:
    """What the file is. Never empty: an unreadable file comes back as kind 'unsupported' with a reason."""
    ext = path.suffix.lower()
    head = path.read_bytes()[:4096] if path.exists() else b""
    try:
        if ext == ".wav" or (head[:4] == b"RIFF" and head[8:12] == b"WAVE"):
            return profile_wav(path)
        if ext in CONVERT_SOUND:
            return {
                "kind": "unsupported",
                "reason": f"{ext} sound is not read: convert it to WAV. Read: {SUPPORTED}",
            }
        if ext in IMAGE_EXT | VIDEO_EXT:
            return profile_vision(path)
        if ext == ".pdf" or head[:4] == b"%PDF":
            return profile_text(path, as_pdf=True)
        if ext in CODE_EXT:
            return profile_code(path)
        if ext in (".json", ".jsonl", ".ndjson"):
            return profile_json(path)
        if ext in (".csv", ".tsv"):
            return profile_csv(path)
        if ext in {".html", ".htm", ".xml", ".md", ".rst", ".log"}:
            return profile_text(path)
        if head[:8] == b"\x89PNG\r\n\x1a\n" or head[:3] == b"\xff\xd8\xff":
            return profile_vision(path)
        if _looks_text(head):
            return profile_csv(path) if _looks_tabular(path) else profile_text(path)
        return {
            "kind": "unsupported",
            "reason": f"I do not know this kind of file ({ext or 'no extension'}). Read: {SUPPORTED}",
        }
    except Exception as exc:
        return {
            "kind": "unsupported",
            "reason": f"the file could not be read: {type(exc).__name__}: {exc}. Read: {SUPPORTED}",
        }


def needs_from_profile(pr: dict) -> tuple[list[str], float | None, list[str]]:
    """The jobs a file asks for, the time available for one row (seconds) and notes that say why."""
    needs: list[str] = []
    notes: list[str] = []
    budget = None
    kind = pr["kind"]
    if kind == "unsupported":
        return [], None, [pr["reason"], f"supported: {SUPPORTED}"]
    if kind == "sound":
        budget = 1024.0 / pr["rate"]
        notes.append(
            f"sound, {pr['rate']} Hz, {pr['seconds']:.1f} s; blocks of 1024 samples leave {budget * 1e3:.0f} ms per block"
        )
        return ["audio_features", "detect_anomaly", "detect_drift"], budget, notes
    if kind == "code":
        notes.append(
            f"Python code: {pr['lines']} lines, {pr['functions']} functions, {pr['classes']} classes, {pr['loops']} loops"
        )
        if pr.get("syntax_error"):
            notes.append(
                "the source has a syntax error ("
                + pr["syntax_error"]
                + "): only its words were read"
            )
        if pr["imports"]:
            notes.append(
                "imports: "
                + ", ".join(pr["imports"][:10])
                + (" ..." if len(pr["imports"]) > 10 else "")
            )
        if pr["dense_armor"]:
            notes.append("already uses Dense-Armor: " + ", ".join(pr["dense_armor"]))
        needs = list(pr["signals"])
        for need, found in pr["signals"].items():
            notes.append(f"{need}: the code uses {', '.join(found[:3])}")
        if not needs:
            notes.append(
                "nothing in the imports or the calls points at one of the jobs of the library"
            )
        return needs, None, notes
    if kind == "text":
        what = f"{pr['type'].upper()} text" + (
            f", {pr['pages']} pages" if pr.get("pages") else ""
        )
        notes.append(f"{what}: {pr['words']} words in {pr['sentences']} sentences")
        if pr.get("note"):
            notes.append(pr["note"])
        if pr["terms"]:
            notes.append("most frequent terms: " + ", ".join(pr["terms"]))
        notes.append(
            f"{pr['numeric_share'] * 100:.0f}% of the tokens are numbers; "
            + (
                f"{pr['tables']} block(s) of numbers look like tables"
                if pr["tables"]
                else "no table of numbers found in the text"
            )
        )
        needs = ["text_features"] if pr["words"] else []
        if pr["sentences"] >= 30:
            notes.append(
                f"the {pr['sentences']} sentences could be read as a stream, but nothing in the file says it is one: "
                "ask for it with  select detect_anomaly detect_drift"
            )
        if pr["table"] is not None:
            t_needs, t_budget, t_notes = needs_from_profile(pr["table"])
            notes += ["the largest table of numbers: " + n for n in t_notes[:3]]
            for n in t_needs:
                if n not in needs:
                    needs.append(n)
        return needs, None, notes
    if kind == "vision":
        size = f", {pr['width']}x{pr['height']}" if pr.get("width") else ""
        notes.append(f"{'video' if pr['video'] else 'picture'} ({pr['type']}{size})")
        return ["vision_features", "detect_anomaly", "detect_drift"], None, notes
    cols = pr["columns"]
    time_col = next(
        (
            c
            for c in cols
            if (c["date"] or TIME_NAMES.match(c["name"])) and c["monotonic"]
        ),
        None,
    )
    label = None
    for c in cols:
        if c["name"].lower() in LABEL_NAMES and c["unique"] <= 20:
            label = c
    if (
        label is None
        and cols
        and cols[-1]["unique"] <= 10
        and not cols[-1]["monotonic"]
        and pr["rows"] > 50
    ):
        label = cols[-1]
    feats = [c for c in cols if c["numeric"] and c is not label and c is not time_col]
    notes.append(f"{pr['rows']} rows, {len(feats)} numeric columns")
    if not feats:
        notes.append("no numeric column: the primitives of the library read numbers")
    if label is not None:
        needs.append("classify")
        notes.append(f"label column '{label['name']}' with {label['unique']} values")
    first = cols[0] if cols else None
    if (
        label is None
        and time_col is None
        and first is not None
        and first["numeric"]
        and first["monotonic"]
        and len(feats) >= 2
    ):
        needs.append("regress")
        notes.append(
            f"looks like a curve: '{feats[-1]['name']}' as a function of '{first['name']}'"
        )
    if time_col is not None and time_col["dt"]:
        budget = time_col["dt"]
        notes.append(
            f"time column '{time_col['name']}': one row every {budget * 1e3:g} ms"
        )
    if pr["rows"] < 200:
        notes.append(
            f"only {pr['rows']} rows: an online learner has little to learn from"
        )
    else:
        needs.append("detect_drift")
    if pr["rows"] >= 50 and feats:
        needs.append("detect_anomaly")
    if len(feats) == 1 or time_col is not None:
        needs.append("forecast")
    if sum(1 for c in feats if ROBOT_NAMES.search(c["name"])) >= 2:
        needs.append("robot")
        notes.append("column names look like joints or torques")
    if sum(1 for c in cols if FINANCE_NAMES.search(c["name"])) >= 2:
        needs.append("finance_controls")
        notes.append("column names look like orders or prices")
    if any(c["missing"] > 0.01 for c in cols):
        needs.append("summarize_stream")
        notes.append("more than 1% of the values are missing")
    return needs, budget, notes
