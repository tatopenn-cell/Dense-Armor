# dense_armor_mcp

An MCP server that lets Claude (or any MCP client) call Dense-Armor's anomaly-shielding
functions directly -- clean a corrupted series, classify each point as an isolated spike vs a
genuine regime change, or run the classic robust-statistics detectors -- without writing Python.

Unlike Dense-Evolution's MCP adapter (a thin HTTP proxy to a separately running FastAPI kernel
behind its Composer web UI), this one is **direct, in-process**: Dense-Armor has no web UI/kernel
to share, and its operations are lightweight pure NumPy/JAX calls with no reason to live in a
separate long-running process. Every tool here imports and calls `dense_armor` directly, in this
same process -- no separate server to start first.

## 1. Install this server's dependencies

```bash
pip install -e ".[mcp]"           # if you have the repo checked out
pip install "dense-armor[mcp]"    # from PyPI, once published
# or, standalone without the extras mechanism:
pip install -r dense_armor/mcp_server/requirements.txt
```

## 2. Register it with your MCP client

**Claude Code:**

```bash
claude mcp add dense_armor -- dense-armor-mcp
```

`dense-armor-mcp` is the console-script entry point (`[project.scripts]` in `pyproject.toml`,
separate from `dense-armor` itself, Armatura's own CLI) -- it just calls this file's `main()`.
Running `python /absolute/path/to/dense_armor/mcp_server/server.py` directly still works too,
e.g. if you haven't installed the package.

Note: this lives at `dense_armor.mcp_server`, not a bare `mcp_server` -- Dense-Evolution's own
adapter is (confusingly) also just called `mcp_server`, and with both packages installed in the
same environment a bare top-level name is a real, observed collision, not a hypothetical one.

**Manual `.mcp.json` / `claude_desktop_config.json` entry:**

```json
{
  "mcpServers": {
    "dense_armor": {
      "command": "dense-armor-mcp"
    }
  }
}
```

## Tools

| Tool | What it does |
|---|---|
| `dense_armor_health` | Confirms `dense_armor` is importable, reports host RAM/backend -- call first if unsure. |
| `dense_armor_clean_signal` | Runs the full Orca shield over a raw series (Orca's own "simple data test" mode, no AI model in the loop). Optional `x_reference`; optional `use_arbiter` for per-point routing. |
| `dense_armor_detect_anomalies` | Classifies each point as `clean`/`spike`/`regime` without correcting anything -- the routing logic behind `use_arbiter`, exposed standalone. |
| `dense_armor_robust_filter` | One of the four classic detectors (Chauvenet, Tukey, Hampel, sigma-clipping) or their combined `pressure_valve` orchestrator. |
| `dense_armor_heal_series` | The neighbor-consensus `healing_filter` -- strong on pervasive noise and genuine sustained jumps, standalone. |

Every tool takes/returns plain JSON (`None`/`null` for a missing reading -- converted to/from
NaN internally, since raw JSON has no NaN literal). See `models.py` for the exact input schema
of each, or [`docs/api/arbiter.md`](https://tatopenn-cell.github.io/Dense-Armor/api/arbiter/) /
[`docs/api/orca.md`](https://tatopenn-cell.github.io/Dense-Armor/api/orca/) for the underlying
functions' full documentation.
