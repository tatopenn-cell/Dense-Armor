"""JSON and compact-console log formatters for Dense-Armor."""

import json
import logging
from datetime import datetime


class MinimalConsoleFormatter(logging.Formatter):
    """Compact console formatter ``[HH:MM:SS] [LEVEL] message``."""

    def format(self, record: logging.LogRecord) -> str:
        """Format one log record as ``[HH:MM:SS] [LEVEL] message``."""
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        return f"[{timestamp}] [{record.levelname}] {record.getMessage()}"


class CompactJsonFormatter(logging.Formatter):
    """One JSON object per log event, one line per event."""

    def format(self, record: logging.LogRecord) -> str:
        """Format one log record as a compact JSON line."""
        log_payload = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "filename": record.filename,
            "line_number": record.lineno,
            "message": record.getMessage(),
            "framework": "dense-armor",
        }
        if record.exc_info:
            log_payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_payload, ensure_ascii=False)


def get_json_file_logger(name: str = "dense_armor") -> logging.Logger:
    """Return a logger that writes JSON lines to ``dense_armor.log``.

    No console handler is attached: the training loop emits one record per
    step, and duplicating every line to stdout floods the terminal. The file
    is the single source for the dashboard and the log reader.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler("dense_armor.log", encoding="utf-8")
        file_handler.setFormatter(CompactJsonFormatter())
        logger.addHandler(file_handler)

    return logger
