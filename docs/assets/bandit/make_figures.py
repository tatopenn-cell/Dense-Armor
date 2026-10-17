"""Regenerate the five figures of the bandit and active pages.

Each figure is a PNG in ``docs/assets/bandit/bandit/`` or
``docs/assets/bandit/active/``. Every number drawn comes from
``benchmarks/bandit_quality.json``; matplotlib only renders it.

Run: ``python docs/assets/bandit/make_figures.py``.
"""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
JSON = ROOT / "benchmarks" / "bandit_quality.json"

STYLE = {
    "figsize": (7.2, 3.6),
    "dpi": 150,
    "facecolor": "#fcfcfb",
    "data": "#2a78d6",
    "highlight": "#eb6834",
    "true": "#0b0b0b",
    "axes.grid": True,
    "grid.alpha": 0.3,
}


def _save(fig, sub, name):
    out = HERE / sub / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=STYLE["dpi"], facecolor=STYLE["facecolor"],
                bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.relative_to(ROOT)}  ({out.stat().st_size} B)")
    return out


def fig_regret_per_policy(data):
    """(a) regret_oracle median per policy, per seed."""
    policies = data["context_free"]["policies"]
    names = list(policies)
    fig, ax = plt.subplots(figsize=STYLE["figsize"], dpi=STYLE["dpi"],
                           facecolor=STYLE["facecolor"])
    y = np.arange(len(names))
    med = [policies[n]["regret_oracle_median"] for n in names]
    lo = [policies[n]["regret_oracle_median"]
          - policies[n]["regret_oracle_iqr"] / 2 for n in names]
    hi = [policies[n]["regret_oracle_median"]
          + policies[n]["regret_oracle_iqr"] / 2 for n in names]
    ax.barh(y, med, color=STYLE["data"],
            xerr=[np.array(med) - np.array(lo), np.array(hi) - np.array(med)],
            capsize=4)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("regret against the oracle (median, 20 seeds)")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, "bandit", "regret_per_policy")


def fig_pull_count(data):
    """(b) pulls per arm, per policy."""
    policies = data["context_free"]["policies"]
    names = list(policies)
    fig, ax = plt.subplots(figsize=STYLE["figsize"], dpi=STYLE["dpi"],
                           facecolor=STYLE["facecolor"])
    x = np.arange(len(names))
    med = [policies[n]["regret_static_median"] for n in names]
    ax.bar(x, med, color=STYLE["data"], width=0.7)
    ax.axhline(0.0, color=STYLE["true"], linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("regret against the best fixed arm")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, "bandit", "static_regret")


def fig_contextual(data):
    """(c) contextual vs context-free UCB1."""
    ctx = data["contextual"]["policies"]
    names = list(ctx)
    fig, ax = plt.subplots(figsize=STYLE["figsize"], dpi=STYLE["dpi"],
                           facecolor=STYLE["facecolor"])
    x = np.arange(len(names))
    med = [ctx[n]["regret_oracle_median"] for n in names]
    colors = [STYLE["data"], STYLE["highlight"]]
    ax.bar(x, med, color=colors[:len(names)])
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("regret against the oracle (median, 20 seeds)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, "bandit", "contextual")


def fig_active_rate(data):
    """(d) labelling rate per strategy vs budget."""
    active = data["active"]
    budget = data["active_budget"]
    names = list(active)
    fig, ax = plt.subplots(figsize=STYLE["figsize"], dpi=STYLE["dpi"],
                           facecolor=STYLE["facecolor"])
    x = np.arange(len(names))
    med = [active[n]["rate_median"] for n in names]
    iqr = [active[n]["rate_iqr"] for n in names]
    ax.bar(x, med, yerr=iqr, capsize=4, color=STYLE["data"])
    ax.axhline(budget, color=STYLE["true"], linestyle="--",
               label=f"budget = {budget}")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("labelling rate (median, 20 seeds)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, "active", "labelling_rate")


def fig_robot(data):
    """(e) chosen kp per seed."""
    robot = data["robot"]
    picks = robot["picks"]
    arms = data["robot_arms"]
    fig, ax = plt.subplots(figsize=STYLE["figsize"], dpi=STYLE["dpi"],
                           facecolor=STYLE["facecolor"])
    counts = {a: picks.count(a) for a in arms}
    x = np.arange(len(arms))
    ax.bar(x, [counts[a] for a in arms], color=STYLE["data"])
    ax.axhline(np.median(list(counts.values())), color=STYLE["true"],
               linestyle="--", label="median")
    ax.set_xticks(x)
    ax.set_xticklabels([f"kp={a}" for a in arms])
    ax.set_ylabel("seeds picking the gain")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, "active", "robot")


def main():
    if not JSON.exists():
        raise SystemExit(f"{JSON} missing. Run benchmarks/bandit_quality.py first.")
    data = json.loads(JSON.read_text())
    print(f"reading {JSON.relative_to(ROOT)}")
    fig_regret_per_policy(data)
    fig_pull_count(data)
    fig_contextual(data)
    fig_active_rate(data)
    fig_robot(data)

    # salvo i numeri citati nelle caption in figures.json
    out = {
        "regret_oracle_median": {
            n: d["regret_oracle_median"]
            for n, d in data["context_free"]["policies"].items()
        },
        "contextual_regret_oracle_median": {
            n: d["regret_oracle_median"]
            for n, d in data["contextual"]["policies"].items()
        },
        "active_rate_median": {
            n: d["rate_median"] for n, d in data["active"].items()
        },
        "active_budget": data["active_budget"],
        "robot_picks": data["robot"]["picks"],
        "robot_regret_median": data["robot"]["regret_median"],
        "robot_seconds_per_decision_median": data["robot"]["seconds_per_decision_median"],
        "robot_arms": data["robot_arms"],
    }
    (HERE / "figures.json").write_text(json.dumps(out, indent=2))
    print(f"  docs/assets/bandit/figures.json  ({(HERE / 'figures.json').stat().st_size} B)")


if __name__ == "__main__":
    main()
