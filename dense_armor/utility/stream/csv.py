"""Read a CSV file as a stream of :class:`Signal` samples.

Examples:
    >>> import tempfile
    >>> from pathlib import Path
    >>> from dense_armor.utility.stream import iter_csv
    >>> d = tempfile.mkdtemp()
    >>> p = Path(d) / "t.csv"
    >>> _ = p.write_text("t,a,b\\n0,1,2\\n1,3,4\\n")
    >>> for sig, y in iter_csv(p, time_col="t"):
    ...     print(sig["a"], sig["b"], sig.t)
    1.0 2.0 0.0
    3.0 4.0 1.0
"""

import ast
import csv
from collections.abc import Iterator
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np

from dense_armor.roles import Signal


def _parse_array_cell(cell: Any) -> list[float] | None:
    if cell is None:
        return None
    s = str(cell).strip()
    if not s:
        return None
    try:
        val = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(val, (list, tuple)):
        return None
    out: list[float] = []
    for v in val:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            return None
    return out


def _parse_scalar_cell(cell: Any) -> float | None:
    if cell is None:
        return None
    s = str(cell).strip()
    if not s:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def iter_csv(
    path: str | Path,
    target: str | None = None,
    time_col: str | None = None,
    columns: list[str] | None = None,
    array_cols: list[str] | None = None,
    start: float | None = None,
    stop: float | None = None,
) -> Iterator[tuple[Signal, Any]]:
    """Iterate rows of a CSV file as ``(Signal, y)`` pairs.

    Args:
        path: path to the CSV file.
        target: label column, or ``None``.
        time_col: timestamp column in seconds, or ``None``.
        columns: scalar feature columns. ``None`` = every column but
            ``target``, ``time_col`` and ``array_cols``.
        array_cols: columns holding a list literal ``"[...]"``, expanded
            into ``<name>_0, ..., <name>_{k-1}`` channels; ``k`` from the
            first row.
        start: skip rows before this time.
        stop: stop at the first row past this time.

    Yields:
        ``(signal, label)`` pairs in file order.

    Raises:
        ValueError: on a requested column that is not in the file.
    """
    array_cols = [] if array_cols is None else list(array_cols)
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        for name in columns or []:
            if name not in header:
                raise ValueError(f"column {name!r} not in {header}")
        for name in array_cols:
            if name not in header:
                raise ValueError(f"column {name!r} not in {header}")
        if target is not None and target not in header:
            raise ValueError(f"target {target!r} not in {header}")
        if time_col is not None and time_col not in header:
            raise ValueError(f"time_col {time_col!r} not in {header}")
        if columns is None:
            col_list = [
                c
                for c in header
                if c != target and c != time_col and c not in array_cols
            ]
        else:
            col_list = list(columns)
        try:
            first = next(reader)
        except StopIteration:
            return
        array_lengths: dict[str, int] = {}
        for name in array_cols:
            parsed = _parse_array_cell(first.get(name))
            array_lengths[name] = 0 if parsed is None else len(parsed)
        feat_names: list[str] = list(col_list)
        for name in array_cols:
            clean = name.replace(" ", "_")
            for i in range(array_lengths[name]):
                feat_names.append(f"{clean}_{i}")
        feat_units: list[str] = ["" for _ in feat_names]
        for row in chain([first], reader):
            t_val: float | None = None
            if time_col is not None:
                t_val = _parse_scalar_cell(row.get(time_col))
            if t_val is not None:
                if start is not None and t_val < start:
                    continue
                if stop is not None and t_val > stop:
                    return
            vals: list[float] = []
            mask: list[bool] = []
            for name in col_list:
                v = _parse_scalar_cell(row.get(name))
                if v is None:
                    vals.append(float("nan"))
                    mask.append(True)
                else:
                    vals.append(v)
                    mask.append(False)
            for name in array_cols:
                k = array_lengths[name]
                parsed = _parse_array_cell(row.get(name))
                if parsed is None or len(parsed) != k:
                    for _ in range(k):
                        vals.append(float("nan"))
                        mask.append(True)
                else:
                    for v in parsed:
                        vals.append(v)
                        mask.append(False)
            if not vals:
                sig = Signal(values=np.zeros(0), names=[], units=[], t=t_val)
            else:
                sig = Signal(
                    values=np.asarray(vals, dtype=float),
                    names=feat_names,
                    units=feat_units,
                    t=t_val,
                    missing=np.asarray(mask, dtype=bool),
                )
            if target is not None:
                raw = row.get(target)
                if raw is None or str(raw).strip() == "":
                    label: Any = None
                else:
                    try:
                        label = float(raw)
                    except (TypeError, ValueError):
                        label = raw
            else:
                label = None
            yield sig, label
