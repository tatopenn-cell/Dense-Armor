"""Tests for the catalog (item D)."""
import ast
import time
from pathlib import Path

from dense_armor.usable import catalog as ct

REPO = ct.REPO


# --------------------------------------------------------------
# Independent count: an ast script that applies the same rules
# --------------------------------------------------------------
def _independent_count(repo: Path) -> int:
    """Count the public classes with the rules of build_catalog, without
    importing the catalog: its own ast reading, its own copy of the rules.
    """
    ABSTRACT = {
        "Estimator", "Classifier", "Regressor", "Transformer",
        "AnomalyDetector", "DriftDetector", "Root", "Signal",
    }
    ROLES = ABSTRACT - {"Estimator", "Root", "Signal"}
    SKIP_SUFFIX = ("Error", "Config")
    classes = []
    idx: dict[str, ast.ClassDef] = {}
    for p in sorted((repo / "dense_armor").rglob("*.py")):
        rel = p.relative_to(repo).parts
        if "__pycache__" in p.parts:
            continue
        if len(rel) >= 2 and rel[0] == "dense_armor" and rel[1] == "usable":
            continue
        if p.name.startswith("_") and p.name != "__init__.py":
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes.append((p, node))
                idx.setdefault(node.name, node)

    def role_of(node: ast.ClassDef, seen: tuple = ()) -> str | None:
        for b in node.bases:
            name = b.id if isinstance(b, ast.Name) else getattr(b, "attr", None)
            if name in ROLES:
                return name
            if name in idx and name not in seen:
                found = role_of(idx[name], seen + (name,))
                if found:
                    return found
        return None

    n = 0
    for p, node in classes:
        name = node.name
        if name.startswith("_") or name in ABSTRACT or name.endswith(SKIP_SUFFIX):
            continue
        rel = p.relative_to(repo).parts
        sub = rel[rel.index("core") + 1 : -1] if "core" in rel else rel[1:-1]
        domain = sub[-1] if sub else "core"
        # the folders of the contracts, the checks, the servers and the bridges
        # hold classes that are not primitives, unless they carry a role
        if domain in ("roles", "checks", "mcp_server", "bridges") and not role_of(node):
            continue
        n += 1
    return n


# --------------------------------------------------------------
# The number matches that of the independent script
# --------------------------------------------------------------
def test_catalog_count_matches_independent():
    cat = ct.get_catalog(REPO, refresh=True)
    assert len(cat) == _independent_count(REPO), (
        f"catalog: {len(cat)} rows; the independent count says "
        f"{_independent_count(REPO)}."
    )


def test_catalog_no_usable_classes():
    """The CLI does not count itself in the catalog."""
    cat = ct.get_catalog(REPO)
    usa = [r for r in cat if r["module"].startswith("dense_armor.usable")]
    assert usa == [], f"trovate nel catalogo: {[r['name'] for r in usa]}"


def test_catalog_no_mentions_field():
    """The citation count has been removed."""
    cat = ct.get_catalog(REPO)
    assert cat, "catalogo vuoto"
    for r in cat:
        assert "mentions" not in r, f"mentions presente in {r['name']}"


def test_catalog_no_load_priors():
    """load_priors is dead code."""
    assert not hasattr(ct, "load_priors"), "load_priors deve essere stata rimossa"


# --------------------------------------------------------------
# Homonyms: classes with the same name in different folders
# --------------------------------------------------------------
def test_homonyms_have_dir_suffix():
    """KNNClassifier exists in core and in ml; at least one of the two rows
    has the suffix [dir]."""
    cat = ct.get_catalog(REPO)
    knn = [r for r in cat if r["class"] == "KNNClassifier"]
    assert len(knn) >= 2, f"KNNClassifier: {len(knn)} righe (atteso >= 2)"
    con_suffisso = [r for r in knn if "[" in r["name"]]
    assert con_suffisso, "nessuna riga KNNClassifier ha [dir]"


def test_every_name_is_unique():
    """After the renaming, no name repeats."""
    cat = ct.get_catalog(REPO)
    nomi = [r["name"] for r in cat]
    dup = {n for n in nomi if nomi.count(n) > 1}
    assert not dup, f"nomi duplicati: {sorted(dup)[:5]}"


# --------------------------------------------------------------
# Tempi (criterio item D)
# --------------------------------------------------------------
def test_cache_cold_under_3s():
    """Build without cache: median of 3 runs, limit 3 s.

    A single run is noisy on machines with few cores (Colab: 2).
    The median measures the typical value, not the outlier. The 3 s
    limit applies to this value.
    """
    tempi = []
    for _ in range(3):
        t0 = time.perf_counter()
        ct.build_catalog(REPO)
        tempi.append(time.perf_counter() - t0)
    mediana = sorted(tempi)[1]
    assert mediana < 3.0, (
        f"build fredda mediana: {mediana * 1e3:.0f} ms (limite 3000); "
        f"run: {[round(t * 1e3) for t in tempi]}"
    )


def test_cache_warm_under_500ms():
    ct.get_catalog(REPO, refresh=True)  # scrive la cache
    t0 = time.perf_counter()
    ct.get_catalog(REPO, refresh=False)
    t = time.perf_counter() - t0
    assert t < 0.5, f"cache calda: {t * 1e3:.0f} ms (limite 500)"


# --------------------------------------------------------------
# Cache invalidated when the SHA changes
# --------------------------------------------------------------
def test_cache_version_is_v8():
    """The cache version is 8 after the removal of mentions."""
    cache = ct.HERE / "catalog.json"
    if cache.exists():
        import json
        d = json.loads(cache.read_text(encoding="utf-8"))
        assert d.get("v") == 8, f"v: {d.get('v')} (atteso 8)"


def test_cache_invalidated_on_refresh():
    """refresh=True recomputes even if the cache is fresh."""
    ct.get_catalog(REPO, refresh=True)
    t0 = time.perf_counter()
    ct.get_catalog(REPO, refresh=True)
    t = time.perf_counter() - t0
    # must not use the cache (must recompute)
    assert t >= 0.05, f"refresh troppo veloce: {t * 1e3:.0f} ms (non ha ricalcolato?)"
