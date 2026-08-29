#!/usr/bin/env python3
"""
MCP server for Dense-Armor (dense_armor_mcp).

Unlike Dense-Evolution's MCP adapter (a thin HTTP proxy to a separately
running FastAPI kernel behind its Composer web UI), this one is DIRECT
in-process: Dense-Armor has no web UI/kernel to share, and its operations
(Orca.protect_and_forward, the standalone arbiter/robust_filters/healing
functions) are lightweight pure NumPy/JAX calls with no reason to live in
a separate long-running process. Every tool here imports and calls
dense_armor directly, in this same process.

Lives at `dense_armor.mcp_server`, NOT a bare top-level `mcp_server`
package (which is what Dense-Evolution's own adapter is called) -- a real
collision, not a hypothetical one: with both packages editable-installed
in the same environment, `from mcp_server.server import mcp` resolved to
whichever one's install happened to win, silently returning Dense-Evolution's
25 tools instead of these 5 depending on unrelated install order/cwd.
Namespacing under `dense_armor.` makes the import path unambiguous and
needs no `package-dir` override in pyproject.toml (`packages.find`
already discovers it as a normal subpackage of `dense_armor`).

Requires JAX 64-bit precision, same as any other dense_armor entry point
(see the package's own README) -- MUST be set before dense_armor (or
anything importing it) is first imported, not just before mcp.run().
Done as the very first lines below, before the `from .tools import ...`
line that transitively imports dense_armor -- setting it inside main()
instead would be too late, since that import already ran at module load.

`mcp` is created here BEFORE the `from .tools import ...` line below, and
tools.py does `from .server import mcp` -- Python resolves this correctly
despite looking circular: by the time that import runs, `dense_armor.mcp_server.server`
is already in `sys.modules` (registered before this file's body starts
executing) with `mcp` already assigned. Reordering `mcp = MCPServer(...)`
to after the tool import would break this (same convention as
Dense-Evolution's tools/mcp_server/server.py).
"""
import jax
jax.config.update("jax_enable_x64", True)

from mcp.server import MCPServer  # noqa: E402

mcp = MCPServer("dense_armor_mcp")

from .tools import (  # noqa: E402
    dense_armor_health, dense_armor_clean_signal, dense_armor_detect_anomalies,
    dense_armor_robust_filter, dense_armor_heal_series,
)


def main():
    """Console-script entry point (`dense-armor-mcp`, see `[project.scripts]`
    in pyproject.toml -- a separate script from `dense-armor` itself, which
    is Armatura's own number-cleaning CLI and stays untouched by this).
    stdio transport: this process is meant to be launched by an MCP client
    (Claude Code, Claude Desktop, ...) as a subprocess, not run standalone
    in a terminal."""
    mcp.run()  # pragma: no cover -- blocks on the real stdio transport loop


if __name__ == "__main__":
    main()
