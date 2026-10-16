"""Provenance archive and text report for scientific traceability.

:class:`AIEngineVisualizer` writes SHA-256 signed JSON archives and compact
text reports of the filter behavior.
"""

import hashlib
import json
import os
import platform
import time

import numpy as np
import psutil

from .. import __version__ as _PKG_VERSION


class AIEngineVisualizer:
    """Export tools for scientific traceability.

    The ``ENGINE_SIGNATURE`` is derived from the package version instead of
    being hard-coded, so it stays in sync with the release.

    Args:
        output_dir: folder where exported archives and reports are written.
    """

    ENGINE_SIGNATURE = f"dense-armor-v{_PKG_VERSION}"

    def __init__(self, output_dir: str = ".") -> None:
        """Store the output folder for exported files."""
        self.output_dir = output_dir

    def export_provenance_archive(
        self,
        run_history: list,
        filename: str = "ai_provenance_archive.json",
    ) -> str:
        """Write a SHA-256 signed provenance archive.

        Args:
            run_history: the list of records to include in the archive.
            filename: name of the JSON file inside ``output_dir``.

        Returns:
            The SHA-256 hex digest of the archive body.
        """
        filepath = os.path.join(self.output_dir, filename)

        provenance_payload: dict = {
            "metadata": {
                "engine_signature": self.ENGINE_SIGNATURE,
                "export_timestamp_utc": time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.gmtime()
                ),
                "execution_environment": {
                    "os": platform.system(),
                    "architecture": platform.machine(),
                    "python": platform.python_version(),
                    "hardware": {
                        "cpu_cores_logical": psutil.cpu_count(logical=True),
                        "total_ram_gb": round(
                            psutil.virtual_memory().total / (1024**3), 2
                        ),
                    },
                },
            },
            "records": run_history,
        }

        raw_bytes = json.dumps(provenance_payload, sort_keys=True, indent=4).encode(
            "utf-8"
        )
        sha256 = hashlib.sha256(raw_bytes).hexdigest()
        provenance_payload["metadata"]["integrity_sha256"] = sha256

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(provenance_payload, f, indent=4)

        return sha256

    @staticmethod
    def export_trend_report_text(
        raw_signal: np.ndarray,
        filtered_signal: np.ndarray,
        filename: str = "ai_trend_report.txt",
    ):
        """Write a compact text report of the filter performance.

        Args:
            raw_signal: the raw signal, shape ``(N_scenarios, T_steps)``.
            filtered_signal: the filtered signal, same shape as ``raw_signal``.
            filename: name of the text file.
        """
        var_raw = float(np.var(raw_signal))
        var_filtered = float(np.var(filtered_signal))
        damping_pct = (var_raw - var_filtered) / var_raw * 100.0 if var_raw > 0 else 0.0
        with open(filename, "w", encoding="utf-8") as f:
            f.write("=== DENSE-ARMOR TELEMETRY REPORT ===\n")
            f.write(f"Engine:                  {AIEngineVisualizer.ENGINE_SIGNATURE}\n")
            f.write(
                f"Timestamp UTC:           "
                f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}\n"
            )
            f.write(f"Scenarios monitored:     {raw_signal.shape[0]}\n")
            f.write(f"Time steps:              {raw_signal.shape[1]}\n")
            f.write(f"Raw signal variance:     {var_raw:.6f}\n")
            f.write(f"Stabilized variance:     {var_filtered:.6f}\n")
            f.write(f"Noise damping:           {damping_pct:.2f}%\n")
