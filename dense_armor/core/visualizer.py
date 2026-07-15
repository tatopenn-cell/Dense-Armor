# -*- coding: utf-8 -*-
"""
core/visualizer.py
==================
AIEngineVisualizer — esportazione provenance con SHA-256 e report testuali.
"""

import hashlib
import json
import os
import platform
import time

import numpy as np
import psutil

# Importazione condizionale per leggere la versione del package
try:
    from . import __version__ as _PKG_VERSION
except ImportError:
    _PKG_VERSION = "1.0.0"


class AIEngineVisualizer:
    """
    Strumenti di esportazione per la tracciabilità scientifica.
    Genera archivi JSON firmati SHA-256 e report testuali del filtro.
    """

    ENGINE_SIGNATURE = f"TensorFlowEngine-Sentinel-v{_PKG_VERSION}"

    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir

    # ── Provenance archive ────────────────────────────────────────────────────

    def export_provenance_archive(
        self,
        run_history: list,
        filename:    str = "ai_provenance_archive.json",
    ) -> str:
        """
        Genera un archivio di tracciabilità scientifica con firma SHA-256.

        Returns
        -------
        sha256_hash : str
        """
        filepath = os.path.join(self.output_dir, filename)

        provenance_payload = {
            "metadata": {
                # FIX BUG: engine_signature non più hardcoded, usa la versione del package
                "engine_signature":    self.ENGINE_SIGNATURE,
                "export_timestamp_utc": time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
                "execution_environment": {
                    "os":           platform.system(),
                    "architecture": platform.machine(),
                    "python":       platform.python_version(),
                    "hardware": {
                        "cpu_cores_logical": psutil.cpu_count(logical=True),
                        "total_ram_gb":      round(
                            psutil.virtual_memory().total / (1024 ** 3), 2
                        ),
                    },
                },
            },
            "records": run_history,
        }

        raw_bytes  = json.dumps(provenance_payload, sort_keys=True, indent=4).encode('utf-8')
        sha256     = hashlib.sha256(raw_bytes).hexdigest()
        provenance_payload["metadata"]["integrity_sha256"] = sha256

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(provenance_payload, f, indent=4)

        return sha256

    # ── Trend report ──────────────────────────────────────────────────────────

    @staticmethod
    def export_trend_report_text(
        raw_signal:      np.ndarray,
        filtered_signal: np.ndarray,
        filename:        str = "ai_trend_report.txt",
    ):
        """Esporta un report testuale compatto delle performance del filtro."""
        var_raw      = float(np.var(raw_signal))
        var_filtered = float(np.var(filtered_signal))
        damping_pct  = (
            (var_raw - var_filtered) / var_raw * 100.0
            if var_raw > 0 else 0.0
        )
        with open(filename, "w", encoding="utf-8") as f:
            f.write("=== TENSORFLOWENGINE SENTINEL — TELEMETRY REPORT ===\n")
            f.write(f"Engine:                  {AIEngineVisualizer.ENGINE_SIGNATURE}\n")
            f.write(f"Timestamp UTC:           {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}\n")
            f.write(f"Scenari monitorati:      {raw_signal.shape[0]}\n")
            f.write(f"Passi temporali:         {raw_signal.shape[1]}\n")
            f.write(f"Varianza segnale grezzo: {var_raw:.6f}\n")
            f.write(f"Varianza stabilizzata:   {var_filtered:.6f}\n")
            f.write(f"Smorzamento rumore:      {damping_pct:.2f}%\n")