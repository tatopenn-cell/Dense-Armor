# -*- coding: utf-8 -*-
import json
import logging

from dense_armor.core.logger import (
    MinimalConsoleFormatter,
    CompactJsonFormatter,
    get_enterprise_logger,
)


def _make_record(msg="ciao", level=logging.INFO):
    return logging.LogRecord(
        name="test", level=level, pathname="modulo.py", lineno=42,
        msg=msg, args=(), exc_info=None,
    )


def test_minimal_console_formatter_include_livello_e_messaggio():
    formatted = MinimalConsoleFormatter().format(_make_record())
    assert "[INFO]" in formatted
    assert "ciao" in formatted


def test_compact_json_formatter_produce_json_valido_con_i_campi_attesi():
    formatted = CompactJsonFormatter().format(_make_record())
    payload = json.loads(formatted)
    assert payload["level"] == "INFO"
    assert payload["message"] == "ciao"
    assert payload["filename"] == "modulo.py"
    assert payload["line_number"] == 42
    assert payload["framework"] == "Sentinel-TensorFlowEngine"


def test_compact_json_formatter_include_traceback_se_presente():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys
        exc_info = sys.exc_info()
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="modulo.py", lineno=1,
            msg="errore", args=(), exc_info=exc_info,
        )
    payload = json.loads(CompactJsonFormatter().format(record))
    assert "exception" in payload
    assert "ValueError" in payload["exception"]


def test_get_enterprise_logger_scrive_file_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    logger_name = "sentinel_test_unico"
    logger = get_enterprise_logger(logger_name)
    logger.info("evento di prova")
    for handler in logger.handlers:
        handler.flush()

    log_file = tmp_path / "sentinel_dashboard.log"
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(lines[-1])
    assert payload["message"] == "evento di prova"


def test_get_enterprise_logger_non_duplica_handler_se_richiamato(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    logger_name = "sentinel_test_no_dup"
    logger_a = get_enterprise_logger(logger_name)
    n_handlers = len(logger_a.handlers)
    logger_b = get_enterprise_logger(logger_name)
    assert logger_a is logger_b
    assert len(logger_b.handlers) == n_handlers
