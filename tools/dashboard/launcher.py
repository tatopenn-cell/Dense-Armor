"""Starts the Dense-Armor dashboard like a normal program: no console window, the browser opens by itself and the server stops when the page is closed.

The desktop icon runs ``pythonw launcher.py``. ``pythonw`` has no console, so nothing black appears; the output goes to ``run/dashboard.log``. Every open page pings ``/ping`` every few seconds; when no page has pinged for ``--idle-seconds`` the server exits. A second click on the icon while the dashboard runs only opens the browser again.
"""
from __future__ import annotations

import argparse
import os
import shutil
import socket
import sys
import threading
import time
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RUN = os.path.join(HERE, "run")


def idle_expired(now: float, started: float, last_ping: float | None,
                 idle_seconds: float, first_page_seconds: float) -> bool:
    """True when the server has to stop: no page ever came, or the last page went away."""
    if last_ping is None:
        return now - started > first_page_seconds
    return now - last_ping > idle_seconds


def port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def prepare_config() -> str:
    os.makedirs(RUN, exist_ok=True)
    cfg = os.path.join(RUN, "demo.toml")
    if not os.path.exists(cfg):
        shutil.copyfile(os.path.join(ROOT, "test", "fixtures", "finance", "demo.toml"), cfg)
    return cfg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Start the Dense-Armor dashboard.")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--idle-seconds", type=float, default=120.0,
                    help="stop when no page has pinged for this long")
    ap.add_argument("--first-page-seconds", type=float, default=180.0,
                    help="stop when no page has pinged this long after the start")
    args = ap.parse_args(argv)
    url = f"http://127.0.0.1:{args.port}/"

    if port_open(args.port):
        if not args.no_browser:
            webbrowser.open(url)
        return 0

    cfg = prepare_config()
    log = open(os.path.join(RUN, "dashboard.log"), "a", buffering=1, encoding="utf-8")
    if sys.stdout is None or sys.stderr is None or not sys.stdout.isatty():
        sys.stdout = sys.stderr = log

    sys.path[:0] = [os.path.join(ROOT, "tools"), ROOT]
    from dashboard.app import create_app

    app = create_app(cfg)
    started = time.time()

    def watch() -> None:
        while True:
            time.sleep(2.0)
            if idle_expired(time.time(), started, app.config.get("LAST_PING"),
                            args.idle_seconds, args.first_page_seconds):
                log.write("no page is open any more: stopping\n")
                log.flush()
                os._exit(0)

    def open_when_ready() -> None:
        for _ in range(120):
            if port_open(args.port):
                break
            time.sleep(0.5)
        if not args.no_browser:
            webbrowser.open(url)

    threading.Thread(target=watch, daemon=True).start()
    threading.Thread(target=open_when_ready, daemon=True).start()
    app.run(host="127.0.0.1", port=args.port, threaded=True, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
