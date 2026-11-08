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

    from dense_armor.utility.finance.dashboard import create_app

    cfg = _cfg(tmp_path)
    app = create_app(str(cfg))
    server, thread, url = _serve(app)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(url)
            page.click("#stop")
            page.wait_for_selector("#stopped", state="visible", timeout=5000)
            browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=2)
    assert (tmp_path / "STOP").exists()


def test_2_stop_already_raised(tmp_path, chromium_available):
    from playwright.sync_api import sync_playwright

    from dense_armor.utility.finance.dashboard import create_app

    cfg = _cfg(tmp_path)
    (tmp_path / "STOP").write_bytes(b"")
    app = create_app(str(cfg))
    server, thread, url = _serve(app)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(url)
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

    from dense_armor.utility.finance.dashboard import create_app

    cfg = _cfg(tmp_path)
    app = create_app(str(cfg))
    server, thread, url = _serve(app)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
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
        importlib.resources.files("dense_armor.utility.finance")
        / "static"
        / "emergency.html"
    )
    text = ref.read_text(encoding="utf-8")
    assert 'id="stop"' in text
    assert 'fetch("/emergency-stop"' in text


def test_5_package_data_declared(chromium_available):
    """The pyproject.toml declares static/*.html as package data.

    importlib.resources reads from the source tree in editable mode, so it
    does not prove the wheel would include the file. This test checks the
    contract that the wheel honours.
    """
    from pathlib import Path

    import tomllib

    root = Path(__file__).resolve().parents[3]
    pyproject = root / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    pkg_data = data.get("tool", {}).get("setuptools", {}).get("package-data", {})
    assert "static/*.html" in pkg_data.get("dense_armor.utility.finance", [])
