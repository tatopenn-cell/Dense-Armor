"""Dark figures for the robot pages of the online dynamics, drawn with the library itself, on the 2-link arm
of ``write_minimal_urdf`` (a payload that is not in the model).

Generates in the same folder:
- residual_learning.png   the torque error of the model alone and of the model plus the learner, sample by sample
- drift_alarms.png        when each trigger of the drift-aware learner rings, with a payload change at sample 200

Run from the repo root:
    python docs/assets/learn/make_dynamics_robot_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dense_armor.core.online.learn.online_dynamics import (
    DriftAwareResidualDynamicsLearner,
    ResidualDynamicsLearner,
    write_minimal_urdf,
)
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel

HERE = Path(__file__).resolve().parent
BG, PANEL, GRID = "#0b1118", "#111a24", "#26364a"
INK, MUTED = "#e8fbff", "#8fa3b8"
CYAN, ORANGE, PINK = "#3ec6e8", "#ffa21f", "#ff2d87"

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "grid.color": GRID,
        "grid.alpha": 0.6,
        "font.size": 10,
    }
)


def heading(fig, title, sub, y=0.945):
    fig.text(0.04, y, title, fontsize=20, fontweight="bold", color=INK)
    fig.text(0.04, y - 0.06, sub, fontsize=11.5, color=CYAN)


def tau(m, s):
    return np.asarray(m.mass_matrix(s[0])) @ s[2] + np.asarray(
        m.bias_forces(s[0], s[1])
    )


def residual_learning():
    nom = RigidBodyModel(write_minimal_urdf())
    real = RigidBodyModel(write_minimal_urdf(payload_mass=0.5))
    learner = ResidualDynamicsLearner(nom)
    rng = np.random.default_rng(0)
    train = rng.uniform(-1, 1, (100, 3, 2))
    test = rng.uniform(-1, 1, (20, 3, 2))
    true = [tau(real, s) for s in test]
    base = np.sqrt(np.mean([(true[i] - tau(nom, s)) ** 2 for i, s in enumerate(test)]))
    marks = [0, 1, 2, 3, 5, 8, 12, 20, 30, 40, 60, 80, 100]
    errs = []
    for k in range(101):
        if k in marks:
            errs.append(
                np.sqrt(
                    np.mean(
                        [
                            (true[i] - learner.predict_torque(*s)) ** 2
                            for i, s in enumerate(test)
                        ]
                    )
                )
            )
        if k < 100:
            learner.learn_one(*train[k], tau(real, train[k]))
    fig, ax = plt.subplots(figsize=(16, 5.6))
    fig.subplots_adjust(top=0.74, bottom=0.14, left=0.06, right=0.985)
    heading(
        fig,
        "THE MANUAL, AND THE EXTRA STRENGTH IT DOES NOT KNOW",
        "torque error (RMSE, N·m) on 20 held-out moves of a 2-link arm with a 0.5 kg payload that the URDF does not have",
    )
    ax.axhline(base, color=ORANGE, lw=2.4, label=f"the URDF alone: {base:.3f}")
    xs = [max(m, 0.7) for m in marks]
    ax.plot(
        xs,
        errs,
        "o-",
        color=CYAN,
        lw=2.6,
        ms=6,
        label="the URDF plus ResidualDynamicsLearner",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("samples learned (0 is drawn at 0.7)")
    ax.set_ylabel("torque error [N·m]")
    ax.legend(loc="center right", facecolor=PANEL, edgecolor=GRID, labelcolor=INK)
    ax.grid(True)
    fig.savefig(HERE / "residual_learning.png", dpi=150, facecolor=BG)
    plt.close(fig)
    i40 = marks.index(40)
    print(
        "residual: nominal",
        round(float(base), 3),
        "| learner at 40 samples",
        round(float(errs[i40]), 3),
        "| at 100",
        round(float(errs[-1]), 3),
    )


def drift_alarms():
    nom = RigidBodyModel(write_minimal_urdf())
    light = RigidBodyModel(write_minimal_urdf(payload_mass=0.5))
    heavy = RigidBodyModel(write_minimal_urdf(payload_mass=1.5))
    rng = np.random.default_rng(0)
    states = rng.uniform(-1, 1, (400, 3, 2))
    configs = [
        ("Hampel", {"trigger": "hampel"}),
        ("CUSUM h = 20", {"trigger": "cusum", "drift_h": 20.0}),
        ("CUSUM h = 10", {"trigger": "cusum", "drift_h": 10.0}),
        ("CUSUM h = 5", {"trigger": "cusum", "drift_h": 5.0}),
    ]
    events = {}
    for name, kw in configs:
        learner = DriftAwareResidualDynamicsLearner(nom, **kw)
        ev = []
        for i, s in enumerate(states):
            learner.learn_one(*s, tau(light if i < 200 else heavy, s))
            if learner._countdown == learner.warmup_steps - 1:
                ev.append(i)
        events[name] = ev
    fig, ax = plt.subplots(figsize=(16, 5.2))
    fig.subplots_adjust(top=0.74, bottom=0.14, left=0.12, right=0.985)
    heading(
        fig,
        "WHEN THE ALARM RINGS",
        "the payload goes from 0.5 to 1.5 kg at sample 200; each tick is a sample at which the trigger of DriftAwareResidualDynamicsLearner rang",
    )
    cols = [CYAN, ORANGE, PINK, "#6a5acd"]
    for k, (name, ev) in enumerate(events.items()):
        ax.vlines(ev, k - 0.35, k + 0.35, color=cols[k], lw=2.4)
    ax.axvline(200, color=INK, lw=1.4, ls=":")
    ax.text(204, len(events) - 0.55, "the payload changes", color=INK, fontsize=10.5)
    ax.set_yticks(range(len(events)))
    ax.set_yticklabels(list(events))
    ax.set_ylim(-0.7, len(events) - 0.3)
    ax.set_xlim(0, 400)
    ax.set_xlabel("sample")
    ax.grid(True, axis="x")
    fig.savefig(HERE / "drift_alarms.png", dpi=150, facecolor=BG)
    plt.close(fig)
    for name, ev in events.items():
        print("alarms", name, ":", len(ev), ev[:12])


if __name__ == "__main__":
    residual_learning()
    drift_alarms()
