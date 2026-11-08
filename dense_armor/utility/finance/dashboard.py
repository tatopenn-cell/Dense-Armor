"""Flask dashboard: status, emergency stop and a chart of the decisions."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

from dense_armor.utility.finance.config import load_config
from dense_armor.utility.finance.status import collect_status

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _svg(rows: list[tuple[str, int]]) -> str:
    """Return an SVG bar chart of ``(reason, count)`` rows, sorted by reason.

    The chart has one ``<rect data-reason="..." data-count="...">`` per
    reason, a text label above each bar, and no external resource. The SVG
    is written by hand, so the dashboard does not need a plotting library.
    """
    if not rows:
        return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 100"></svg>'
    rows = sorted(rows, key=lambda r: r[0])
    n = len(rows)
    w = 400
    h = 200
    bar_w = max(4, (w - 40) // max(n, 1) - 10)
    max_count = max(c for _, c in rows) or 1
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
    for i, (reason, count) in enumerate(rows):
        bh = int((count / max_count) * (h - 60))
        x = 20 + i * (bar_w + 10)
        y = h - 20 - bh
        reason = html.escape(reason, quote=True)
        parts.append(
            f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bh}" '
            f'data-reason="{reason}" data-count="{count}"/>'
        )
        parts.append(
            f'<text x="{x + bar_w // 2}" y="{y - 5}" text-anchor="middle" '
            f'font-size="10">{count}</text>'
        )
        parts.append(
            f'<text x="{x + bar_w // 2}" y="{h - 5}" text-anchor="middle" '
            f'font-size="10">{reason}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _count_reasons(log_path: Path) -> list[tuple[str, int]]:
    """Count the ``payload.reason`` of each valid JSON line, ignore garbage."""
    counts: dict[str, int] = {}
    if not log_path.exists():
        return []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = obj.get("payload")
        if not isinstance(payload, dict):
            continue
        reason = payload.get("reason")
        if not isinstance(reason, str):
            continue
        counts[reason] = counts.get(reason, 0) + 1
    return list(counts.items())


def create_app(config_path: str) -> Any:
    """Return a Flask application for a configuration.

    The configuration is checked here, and read again at each request, as
    ``GET /status`` does, so that the three routes use the same file. A
    missing or invalid configuration raises ``ValueError`` (or
    ``FileNotFoundError``) here, not a 500 later. ``flask`` is imported inside this function, so that
    ``import dense_armor.utility.finance`` does not need it.

    Args:
        config_path: the path of the TOML configuration.

    Returns:
        A :class:`flask.Flask` application with three routes:
        ``GET /status`` (the JSON of :func:`collect_status`),
        ``POST /emergency-stop`` (create the stop file, return
        ``{"emergency_stop": true}``) and ``GET /decisions.svg`` (a bar
        chart of the reasons in the log).

    Raises:
        FileNotFoundError: if the configuration file does not exist.
        ValueError: if the configuration is invalid.
    """
    from flask import Flask, Response, jsonify

    load_config(config_path)
    app = Flask(__name__)

    @app.get("/")
    def index() -> Any:
        html_path = STATIC_DIR / "emergency.html"
        return Response(html_path.read_text(encoding="utf-8"), mimetype="text/html")

    @app.get("/status")
    def status() -> Any:
        return jsonify(collect_status(config_path))

    @app.post("/emergency-stop")
    def emergency_stop() -> Any:
        stop = load_config(config_path).paths.emergency_stop
        stop.parent.mkdir(parents=True, exist_ok=True)
        stop.write_bytes(b"")
        return jsonify({"emergency_stop": True})

    @app.get("/decisions.svg")
    def decisions_svg() -> Any:
        rows = _count_reasons(load_config(config_path).paths.log)
        return Response(_svg(rows), mimetype="image/svg+xml")

    return app


def main(argv: list[str] | None = None) -> int:
    """Run the dashboard and return the exit code.

    The command loads the configuration, starts the Flask development
    server on ``--host`` and ``--port``, and returns 0 when the server
    stops. An invalid configuration prints ``{"error": ..., "ok": false}``
    on stdout and returns 2.

    Args:
        argv: the argument list, or ``None`` to read :data:`sys.argv`.

    Returns:
        The exit code: 0 for a clean stop, 2 for an invalid configuration.

    Examples:
        >>> import contextlib, io
        >>> buf = io.StringIO()
        >>> with contextlib.redirect_stdout(buf):
        ...     code = main(["--config", "/nonexistent/dashboard.toml"])
        >>> code
        2
    """
    parser = argparse.ArgumentParser(
        prog="python -m dense_armor.utility.finance.dashboard"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return int(e.code) if e.code is not None else 2

    try:
        app = create_app(args.config)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        sys.stdout.write(
            json.dumps({"error": str(e), "ok": False}, sort_keys=True) + "\n"
        )
        return 2
    app.run(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
