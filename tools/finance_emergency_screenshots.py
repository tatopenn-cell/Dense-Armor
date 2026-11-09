"""Screenshots of the red button, before and after the click."""

from __future__ import annotations

import sys
import tempfile
import threading
from pathlib import Path

OUT = Path("docs/assets/finance/emergency")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from playwright.sync_api import sync_playwright
    from werkzeug.serving import make_server

    from dense_armor.utility.finance.dashboard import create_app

    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        work = Path(d)
        toml = (
            "[schedule]\ntotal = 10.0\nn = 2\n"
            "[pretrade]\nmax_notional = 1e6\nmax_size = 10.0\n"
            "price_band = 0.01\nmax_orders = 5\nwindow_s = 1.0\n"
            "duplicate_window_s = 0.1\n"
            "[ring]\ncapacity = 3\nwidth = 2\n"
            '[paths]\nlog = "log.jsonl"\nfeed = "feed.bin"\n'
            'emergency_stop = "STOP"\n'
        )
        cfg = work / "c.toml"
        cfg.write_text(toml, encoding="utf-8", newline="\n")
        app = create_app(str(cfg))
        server = make_server("127.0.0.1", 0, app)
        port = server.server_port
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        url = f"http://127.0.0.1:{port}/"
        try:
            with sync_playwright() as p:
                b = p.chromium.launch()
                page = b.new_page(viewport={"width": 640, "height": 360})
                page.goto(url + "emergency")
                page.screenshot(path=str(OUT / "before.png"))
                page.click("#stop")
                page.wait_for_selector("#stopped", state="visible", timeout=5000)
                page.screenshot(path=str(OUT / "after.png"))
                b.close()
        finally:
            server.shutdown()
            t.join(timeout=2)
    print(f"wrote {OUT / 'before.png'} and {OUT / 'after.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
