"""Tests for the red button, in a real headless Chromium."""

from __future__ import annotations

import importlib.resources
import threading
from pathlib import Path

import pytest

pytestmark = pytest.mark.needs_browser

BASE_TOML = """\
[schedule]
total = 10.0
n = 2

[pretrade]
max_notional = 1000000.0
max_size = 10.0
price_band = 0.01
max_orders = 5
window_s = 1.0
duplicate_window_s = 0.1

[ring]
capacity = 3
width = 2

[paths]
log = "log.jsonl"
feed = "feed.bin"
emergency_stop = "STOP"
"""


def _cfg(tmp_path: Path, body: str = BASE_TOML) -> Path:
    p = tmp_path / "c.toml"
    p.write_text(body, encoding="utf-8", newline="\n")
    return p


def _serve(app):
    from werkzeug.serving import make_server

    server = make_server("127.0.0.1", 0, app)
    port = server.server_port
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, t, f"http://127.0.0.1:{port}/"


def test_1_click_raises_stop(tmp_path, chromium_available):
    from playwright.sync_api import sync_playwright

    from dashboard.app import create_app

    cfg = _cfg(tmp_path)
    app = create_app(str(cfg))
    server, thread, url = _serve(app)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(url + "emergency")
            page.click("#stop")
            page.wait_for_selector("#stopped", state="visible", timeout=5000)
            browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=2)
    assert (tmp_path / "STOP").exists()


def test_2_stop_already_raised(tmp_path, chromium_available):
    from playwright.sync_api import sync_playwright

    from dashboard.app import create_app

    cfg = _cfg(tmp_path)
    (tmp_path / "STOP").write_bytes(b"")
    app = create_app(str(cfg))
    server, thread, url = _serve(app)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(url + "emergency")
            page.wait_for_selector("#stopped", state="visible", timeout=5000)
            visible = page.eval_on_selector(
                "#stopped", "el => !el.hasAttribute('hidden')"
            )
            assert visible is True
            browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_3_server_down_shows_error(tmp_path, chromium_available):
    from playwright.sync_api import sync_playwright

    from dashboard.app import create_app

    cfg = _cfg(tmp_path)
    app = create_app(str(cfg))
    server, thread, url = _serve(app)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url + "emergency")
        # fermo il server, poi clicco: fetch deve fallire, #error visibile
        server.shutdown()
        thread.join(timeout=2)
        page.click("#stop")
        page.wait_for_selector("#error", state="visible", timeout=5000)
        stopped_hidden = page.eval_on_selector(
            "#stopped", "el => el.hasAttribute('hidden')"
        )
        assert stopped_hidden is True
        browser.close()


def test_4_package_data(chromium_available):
    ref = (
        importlib.resources.files("dashboard")
        / "static"
        / "emergency.html"
    )
    text = ref.read_text(encoding="utf-8")
    assert 'id="stop"' in text
    assert 'fetch("/emergency-stop"' in text


def test_5_the_web_application_is_not_in_the_wheel(chromium_available):
    """The web application is in tools/dashboard, as in Dense-Evolution, not in the package that goes to PyPI.

    ``packages.find`` lists what the wheel contains; ``dashboard`` and ``tools`` must not match it, so the
    templates and the static files are not shipped and nothing of the application sits inside ``dense_armor``.
    """
    from pathlib import Path

    import tomllib

    root = Path(__file__).resolve().parents[3]
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    include = data["tool"]["setuptools"]["packages"]["find"]["include"]
    assert all(not p.startswith(("dashboard", "tools", "webapp")) for p in include), include
    assert (root / "tools" / "dashboard" / "app.py").is_file()
    assert not (root / "dense_armor" / "webapp").exists() and not (root / "dense_armor" / "dashboard").exists()
