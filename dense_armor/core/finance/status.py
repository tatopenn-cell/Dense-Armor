"""Status command: one JSON object with the configuration state."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from dense_armor.core.finance.config import load_config
from dense_armor.core.finance.decision_log import DecisionLog


def collect_status(config_path: str) -> dict:
    """Return the status of a configuration as a dict.

    This is the function that the status command prints and that the
    dashboard of G2 exposes on ``GET /status``. It reads the configuration
    and three paths, and returns one dict. What :func:`load_config` raises
    (``FileNotFoundError``, ``ValueError``, ``RuntimeError``) is raised to
    the caller: the command turns it into an error object, the dashboard
    turns it into a ``ValueError`` at construction.

    Args:
        config_path: the path of the TOML configuration.

    Returns:
        A dict with ``config``, ``ok``, ``emergency_stop``, ``feed`` and
        ``log``, in the form of the status command.

    Raises:
        FileNotFoundError: if ``config_path`` does not exist.
        ValueError: if the configuration is invalid.
        RuntimeError: if ``tomllib`` is not available.
    """
    config = load_config(config_path)

    stop_exists = config.paths.emergency_stop.exists()
    feed_exists = config.paths.feed.exists()
    log_exists = config.paths.log.exists()

    log: dict[str, Any] = {
        "exists": log_exists,
        "path": str(config.paths.log),
        "records": 0,
        "last_hash": None,
        "verified": None,
    }
    if log_exists:
        dlog = DecisionLog(config.paths.log)
        log["records"] = len(dlog)
        log["last_hash"] = dlog.last_hash
        log["verified"] = dlog.verify()

    return {
        "config": config_path,
        "ok": True,
        "emergency_stop": stop_exists,
        "feed": {"exists": feed_exists, "path": str(config.paths.feed)},
        "log": log,
    }


def _inspect(config_path: str) -> tuple[int, dict]:
    try:
        return 0, collect_status(config_path)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        return 2, {"error": str(e), "ok": False}


def main(argv: list[str] | None = None) -> int:
    """Run the status command and return the exit code.

    The command prints one JSON object on stdout, sorted by key, and
    nothing else. A valid configuration gives exit 0, whatever the state
    of the stop file: the command reports, it does not decide. A missing
    or invalid configuration gives exit 2 with ``{"error": ..., "ok":
    false}``. A missing ``--config`` is argparse's own exit 2 on stderr.

    Args:
        argv: the argument list, or ``None`` to read :data:`sys.argv`.

    Returns:
        The exit code: 0 for a valid configuration, 2 for an error.

    Examples:
        >>> import contextlib, io, tempfile
        >>> from pathlib import Path
        >>> text = chr(10).join([
        ...     "[schedule]", "total = 10.0", "n = 2",
        ...     "[pretrade]", "max_notional = 1e6", "max_size = 10.0",
        ...     "price_band = 0.01", "max_orders = 5", "window_s = 1.0",
        ...     "duplicate_window_s = 0.1",
        ...     "[ring]", "capacity = 3", "width = 2",
        ...     "[paths]", 'log = "log.jsonl"', 'feed = "feed.bin"',
        ...     'emergency_stop = "STOP"', "",
        ... ])
        >>> with tempfile.TemporaryDirectory() as d:
        ...     cfg = Path(d) / "c.toml"
        ...     _ = cfg.write_text(text)
        ...     buf = io.StringIO()
        ...     with contextlib.redirect_stdout(buf):
        ...         code = main(["--config", str(cfg)])
        ...     payload = json.loads(buf.getvalue())
        ...     (code, payload["ok"], payload["emergency_stop"])
        (0, True, False)
    """
    parser = argparse.ArgumentParser(
        prog="python -m dense_armor.core.finance.status"
    )
    parser.add_argument(
        "--config", required=True, help="path of the TOML configuration"
    )
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return int(e.code) if e.code is not None else 2

    code, payload = _inspect(args.config)
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
