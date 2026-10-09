"""CASPER dataset: a UR3e robot arm moving in a pick-and-place loop.

The dataset is from Kayan, H., Rana, O., Burnap, P., Perera, C. (2023),
"CASPER: Context-Aware Anomaly Detection System for Industrial Robotic
Arms", arXiv:2303.01300. A Universal Robots UR3e performs a repetitive
pick-and-place task; the data are logged at about 20 Hz, and the test
set is 24 hours long with about 50% anomalous samples (section III-A of
the paper). The user passes a folder containing a Kaggle copy of the
recording
(https://www.kaggle.com/datasets/hkayan/industrial-robotic-arm-anomaly-detection);
the library never redistributes the data.

Expected columns: ``Timestamp`` in seconds, plus joint columns such as
``"Actual Joint Velocities"`` whose cells hold a six-element Python list
literal, e.g. ``"[0.1, -0.2, 0.3, 0.0, 0.4, -0.5]"``. The joint columns
listed in :meth:`stream` are expanded into six channels each
(``<name>_0, ..., <name>_5``).

Licences: the code repository is MIT (Copyright (c) 2023 hkayann,
https://github.com/hkayann/CASPER-PerCom); the Kaggle dataset is CC BY-SA 4.0
(https://creativecommons.org/licenses/by-sa/4.0/), attribution to
Kayan et al. 2023.

Examples:
    >>> from dense_armor.utility.datasets import Casper
    >>> ds = Casper("/path/to/casper_folder")  # doctest: +SKIP
"""

import csv
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from dense_armor.roles import Signal
from dense_armor.utility.stream.csv import iter_csv


def _first_timestamp(path: Path, time_col: str) -> float | None:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        try:
            first = next(reader)
        except StopIteration:
            return None
        raw = first.get(time_col)
        if raw is None:
            return None
        try:
            return float(str(raw).strip())
        except (TypeError, ValueError):
            return None


class Casper:
    """CASPER UR3e joint stream.

    Args:
        path: folder containing the CSV files.
        time_col: name of the timestamp column, in seconds. Default
            ``"Timestamp"``.
        file: name of the CSV file inside ``path``. Default
            ``"right_arm.csv"``.
        target_col: name of the label column, if present. Default
            ``None``.

    Raises:
        FileNotFoundError: if the file does not exist.

    Examples:
        >>> from dense_armor.utility.datasets import Casper
        >>> ds = Casper("/path/to/casper_folder")  # doctest: +SKIP
    """

    def __init__(
        self,
        path: str | Path,
        time_col: str = "Timestamp",
        file: str = "right_arm.csv",
        target_col: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.file = file
        self.time_col = time_col
        self.target_col = target_col
        self._csv = self.path / file
        if not self._csv.exists():
            raise FileNotFoundError(
                f"{self._csv} not found; pass the folder of the Kaggle "
                f"copy of the CASPER dataset"
            )

    def stream(
        self,
        columns: list[str],
        start_h: float | None = None,
        stop_h: float | None = None,
    ) -> Iterator[tuple[Signal, Any]]:
        """Iterate the joint stream, in time order.

        Args:
            columns: names of the joint columns to expand; each cell
                holds a six-element list literal and becomes six
                channels.
            start_h: skip rows before ``start_h`` hours from the first
                timestamp.
            stop_h: stop as soon as a row goes past ``stop_h`` hours
                from the first timestamp.

        Yields:
            ``(signal, label)`` pairs with the expanded joint channels.
        """
        start_s: float | None = None
        stop_s: float | None = None
        if start_h is not None or stop_h is not None:
            t0 = _first_timestamp(self._csv, self.time_col)
            if t0 is not None:
                if start_h is not None:
                    start_s = t0 + start_h * 3600.0
                if stop_h is not None:
                    stop_s = t0 + stop_h * 3600.0
        yield from iter_csv(
            self._csv,
            target=self.target_col,
            time_col=self.time_col,
            columns=[],
            array_cols=columns,
            start=start_s,
            stop=stop_s,
        )

    def n_rows(self) -> int:
        """Number of data rows in the CSV file."""
        n = 0
        with self._csv.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for _ in reader:
                n += 1
        return max(0, n - 1)
