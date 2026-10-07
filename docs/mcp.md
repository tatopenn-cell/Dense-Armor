# MCP Server

Dense-Armor ships a Model Context Protocol server: a small process that exposes the
library's shields and detectors over the MCP protocol, so an LLM agent that already
speaks MCP can call them as tools without importing anything.

The page is a thin shell around the server's own `README.md`, which is the single
source of truth for installation, tool names, and configuration. It is pulled in
directly from `dense_armor/mcp_server/README.md`, so it never goes stale relative to
the code.

{% include-markdown "../dense_armor/mcp_server/README.md" %}

---

## What the server exposes

Three tool families, one for each shield entry point:

- **`shield_series`** — run `Armatura.analizza` on a 1D array and return the cleaned
  series, the anomaly count, and the list of replaced indices. Use this when an agent
  has a numeric series (a loss curve, a sensor channel) and wants it cleaned before
  reasoning about it.
- **`shield_model`** — call `Orca.protect_and_forward` on a small tensor. The model
  itself is passed as a reference to a callable that the MCP server holds, so the
  agent does not have to send model weights over the wire. Use this when the agent
  generates candidate inputs and wants to know whether they look corrupted.
- **`detect_drift`** — run the CUSUM detector on a stream and return the alarm indices
  and the ARL estimates for the parameters used. Use this when an agent is monitoring
  a stream of measurements and wants an early warning on a slow change.

The server does not run models. It runs the shields and the detectors on data the
agent provides, and returns the result. Nothing else.

## Why an MCP server at all

An agent that already has the library installed can call it directly. But in a
multi-agent setup, or in a hosted agent that has no local Python, an MCP server is
the standard way to expose a capability. The three tools above are the ones that make
sense to expose: an agent that wants to clean a series, protect a small forward pass,
or watch for a drift, without pulling in a JAX dependency just to do it.
