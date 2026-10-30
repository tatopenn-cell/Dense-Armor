import ast
import re
import subprocess
import sys

REF = sys.argv[1] if len(sys.argv) > 1 else "origin/master"
REPO = sys.argv[2] if len(sys.argv) > 2 else "."

STRONG = re.compile(
    r"welford|\bm2_?\b|self\._?m2|sum_sq|sumsq|\bvariance\b|self\._?var\b"
    r"|self\.var_|self\._?cov\b|self\.cov_|self\._?mean\b|self\.mean_"
    r"|self\._?sum\b|\b_?s2\b",
    re.I,
)


def git(*args):
    return subprocess.run(
        ["git", "-C", REPO, *args], capture_output=True, text=True
    ).stdout


files = [
    f
    for f in git("ls-tree", "-r", "--name-only", REF, "dense_armor").split()
    if f.endswith(".py") and "/neural/" not in f
]
rows = []
for f in files:
    src = git("show", f"{REF}:{f}")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        continue
    lines = src.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            seg = "\n".join(lines[node.lineno - 1 : node.end_lineno])
            n = len(STRONG.findall(seg))
            if n:
                rows.append(
                    (
                        f.replace("dense_armor/", ""),
                        node.name,
                        n,
                        "float32" in seg,
                        "float64" in seg or "dtype=float" in seg,
                    )
                )
rows.sort()
print("file | class | hits | names float32 | names float64")
for f, c, n, a, b in rows:
    print(f"{f} | {c} | {n} | {'F32' if a else '-'} | {'F64' if b else '-'}")
print("classes with accumulators (neural excluded):", len(rows))
print("of which name float32:", sum(1 for r in rows if r[3]))
