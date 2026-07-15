# -*- coding: utf-8 -*-
import logging
import json
from datetime import datetime


class MinimalConsoleFormatter(logging.Formatter):

    def format(self, record):
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        return f"[{timestamp}] [{record.levelname}] {record.getMessage()}"


class CompactJsonFormatter(logging.Formatter):

    def format(self, record):
        log_payload = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "filename": record.filename,
            "line_number": record.lineno,
            "message": record.getMessage(),
            "framework": "Sentinel-TensorFlowEngine"
        }
        if record.exc_info:
            log_payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_payload, ensure_ascii=False)


def get_enterprise_logger(name: str = "sentinel") -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # niente console_handler: le carte girano ad ogni passo del train,
        # duplicare ogni riga anche su stdout inonda il terminale -- il
        # file resta la fonte per il dashboard/i log.
        file_handler = logging.FileHandler("sentinel_dashboard.log", encoding="utf-8")
        file_handler.setFormatter(CompactJsonFormatter())
        logger.addHandler(file_handler)

    return logger
