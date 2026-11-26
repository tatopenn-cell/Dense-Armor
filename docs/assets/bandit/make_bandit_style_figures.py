"""The figures of the Bandits pages (3D at the start and finance figure), drawn with the library itself.

A desk picks one of five execution algorithms each day (the arms) and learns from the reward (1 when the order is filled
well, 0 when not). The success rates of the arms are 0.2, 0.35, 0.5, 0.65 and 0.8 unless a page says another. The surface
of each page is the mean regret per day so far (the best rate minus the rate of the arm that was pulled) over 1000 days,
for 25 values of one parameter of the policy.

Run from the repo root:
    python docs/assets/bandit/make_bandit_style_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from dense_armor.core.online.bandit import (
    UCB1,
    BayesUCB,
    EpsilonGreedy,
    Exp3,
    LinUCBDisjoint,
    ThompsonSampling,
    UCB1Tuned,
    offline_replay,
)

HERE = Path(__file__).resolve().parent
BG, PANEL, GRID = "#0b1118", "#111a24", "#26364a"
INK, MUTED = "#e8fbff", "#8fa3b8"
CYAN, ORANGE, PINK = "#3ec6e8", "#ffa21f", "#ff2d87"
CMAP = LinearSegmentedColormap.from_list("s", [CYAN, "#6a5acd", PINK])
T = 1000
P = np.array([0.2, 0.35, 0.5, 0.65, 0.8])
ARMS = [0, 1, 2, 3, 4]


def run(make, p=P, steps=T, seed=5):
    """Mean regret per day so far, and the arm counts over time."""
    rng = np.random.default_rng(seed)
    pol = make()
    reg = np.zeros(steps)
    counts = np.zeros((steps, len(p)))
    c = np.zeros(len(p))
    best = p.max()
    tot = 0.0
    for t in range(steps):
        a = pol.pull()
        r = float(rng.random() < p[a])
        pol.update(a, r)
        tot += best - p[a]
        reg[t] = tot / (t + 1)
        c[a] += 1
        counts[t] = c
    return reg, counts


def run_ctx(make, steps=T, seed=6):
    rng = np.random.default_rng(seed)
    pol = make()
    reg = np.zeros(steps)
    tot = 0.0
    for t in range(steps):
        s = float(rng.uniform(-1, 1))
        x = {"trend": s, "bias": 1.0}
        a = pol.pull(context=x)
        mean = {"long": (1 + s) / 2, "short": (1 - s) / 2}
        pol.update(a, float(rng.random() < mean[a]), context=x)
        tot += max(mean.values()) - mean[a]
        reg[t] = tot / (t + 1)
    return reg


def surface(Z, ys, name, title, subtitle, yl, zl, zlim=None, second=None):
    X, Y = np.meshgrid(np.arange(20, T + 1), ys)
    fig = plt.figure(figsize=(12, 8.2))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(top=0.93, bottom=0.0, left=-0.05, right=1.05)
    ax.set_facecolor(BG)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.07, 0.10, 0.14, 1.0))
        axis._axinfo["grid"]["color"] = GRID
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((1.35, 1.2, 0.7), zoom=1.1)
    ax.set_xlabel("days seen", color=CYAN, labelpad=8)
    ax.set_ylabel(yl, color=CYAN, labelpad=8)
    ax.set_zlabel(zl, color=CYAN, labelpad=6)
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.text(0.05, 0.955, title, fontsize=21, fontweight="bold", color=INK)
    fig.text(0.05, 0.915, subtitle, fontsize=11.5, color=CYAN)
    if second is None:
        top = zlim[1] if zlim else max(Z.max(), 1e-9)
        ax.plot_surface(X, Y, Z, facecolors=CMAP(np.clip(Z / top, 0, 1)), rstride=1, cstride=3, linewidth=0, antialiased=False, alpha=0.93)
    else:
        ax.plot_surface(X, Y, Z, color=CYAN, rstride=1, cstride=3, linewidth=0, antialiased=False, alpha=0.8)
        ax.plot_surface(X, Y, second, color=PINK, rstride=1, cstride=3, linewidth=0, antialiased=False, alpha=0.8)
    if zlim:
        ax.set_zlim(*zlim)
    fig.savefig(HERE / f"{name}_3d.png", dpi=150, facecolor=BG)
    plt.close(fig)


def lines(curves, name, title, ylabel, ylim=None, loc="upper right"):
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.set_xlabel("days seen", color=CYAN)
    ax.set_ylabel(ylabel, color=CYAN)
    ax.tick_params(colors=MUTED)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(color=GRID, lw=0.5)
    ax.set_title(title, color=INK, loc="left", fontsize=13, fontweight="bold")
    for (lab, v), col in zip(curves, (CYAN, ORANGE, PINK, MUTED, GREEN if False else "#46d37f")):
        ax.plot(np.arange(1, len(v) + 1), v, color=col, lw=2.0, label=lab)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=INK, fontsize=9, loc=loc)
    fig.tight_layout()
    fig.savefig(HERE / f"{name}_finance.png", dpi=150, facecolor=BG)
    plt.close(fig)


def at(v, days):
    return [round(float(v[d - 1]), 3) for d in days]


def sweep(make_for, params, p=P):
    return np.array([run(lambda v=v: make_for(v), p=p)[0] for v in params])


def main():
    D = (50, 200, 1000)
    # epsilon greedy: epsilon
    eps = np.linspace(0.0, 0.5, 25)
    Z = sweep(lambda e: EpsilonGreedy(ARMS, epsilon=float(e), seed=0), eps)
    surface(Z[:, 19:], eps, "bandit_epsilon_greedy", "A FIXED EPSILON NEVER STOPS EXPLORING", "five execution algorithms over 1000 days: mean regret per day of EpsilonGreedy for 25 values of epsilon", "epsilon", "mean regret per day", (0, 0.4))
    for e in (0.0, 0.1, 0.5):
        k = int(np.argmin(np.abs(eps - e)))
        print("epsilon", round(float(eps[k]), 3), "regret/day at 50, 200, 1000:", at(Z[k], D))
    lines([("epsilon 0", run(lambda: EpsilonGreedy(ARMS, epsilon=0.0, seed=0))[0]), ("epsilon 0.1", run(lambda: EpsilonGreedy(ARMS, epsilon=0.1, seed=0))[0]), ("epsilon 0.3", run(lambda: EpsilonGreedy(ARMS, epsilon=0.3, seed=0))[0])], "bandit_epsilon_greedy", "Five execution algorithms: the regret per day of three epsilons", "mean regret per day", (0, 0.4))

    # ucb: gap between the best arm and the others
    gaps = np.linspace(0.02, 0.4, 25)
    pg = lambda g: np.array([0.5, 0.5, 0.5, 0.5, 0.5 + g])
    Z1 = np.array([run(lambda: UCB1(ARMS, seed=0), p=pg(g))[0] for g in gaps])
    Z2 = np.array([run(lambda: UCB1Tuned(ARMS, seed=0), p=pg(g))[0] for g in gaps])
    surface(Z1[:, 19:], gaps, "bandit_ucb", "UCB1 AND UCB1-TUNED AGAINST THE GAP", "one execution algorithm better than four equal ones by a gap: mean regret per day of UCB1 (cyan) and UCB1Tuned (pink)", "gap of the best arm", "mean regret per day", (0, 0.3), second=Z2[:, 19:])
    for g in (0.02, 0.2, 0.4):
        k = int(np.argmin(np.abs(gaps - g)))
        print("ucb gap", round(float(gaps[k]), 3), "UCB1", at(Z1[k], D), "UCB1Tuned", at(Z2[k], D))
    lines([("UCB1", run(lambda: UCB1(ARMS, seed=0))[0]), ("UCB1Tuned", run(lambda: UCB1Tuned(ARMS, seed=0))[0]), ("EpsilonGreedy 0.1", run(lambda: EpsilonGreedy(ARMS, epsilon=0.1, seed=0))[0])], "bandit_ucb", "Five execution algorithms: UCB1, UCB1-Tuned and epsilon-greedy", "mean regret per day", (0, 0.4))

    # thompson: gap
    Zt = np.array([run(lambda: ThompsonSampling(ARMS, family="bernoulli", seed=0), p=pg(g))[0] for g in gaps])
    surface(Zt[:, 19:], gaps, "bandit_thompson", "THOMPSON SAMPLING AGAINST THE GAP", "one execution algorithm better than four equal ones by a gap: mean regret per day of ThompsonSampling", "gap of the best arm", "mean regret per day", (0, 0.3))
    for g in (0.02, 0.2, 0.4):
        k = int(np.argmin(np.abs(gaps - g)))
        print("thompson gap", round(float(gaps[k]), 3), at(Zt[k], D))
    lines([("ThompsonSampling", run(lambda: ThompsonSampling(ARMS, family="bernoulli", seed=0))[0]), ("UCB1", run(lambda: UCB1(ARMS, seed=0))[0])], "bandit_thompson", "Five execution algorithms: Thompson sampling and UCB1", "mean regret per day", (0, 0.4))

    # bayes ucb: c
    cs = np.linspace(0.0, 3.0, 25)
    Zb = sweep(lambda c: BayesUCB(ARMS, family="bernoulli", horizon=T, c=float(c), seed=0), cs)
    surface(Zb[:, 19:], cs, "bandit_bayes_ucb", "BAYES-UCB AGAINST THE EXPONENT c", "five execution algorithms over 1000 days: mean regret per day of BayesUCB for 25 values of c", "exponent c", "mean regret per day", (0, 0.3))
    for c in (0.0, 1.0, 3.0):
        k = int(np.argmin(np.abs(cs - c)))
        print("bayes c", round(float(cs[k]), 3), at(Zb[k], D))
    lines([("BayesUCB c=0", run(lambda: BayesUCB(ARMS, family="bernoulli", horizon=T, c=0.0, seed=0))[0]), ("UCB1", run(lambda: UCB1(ARMS, seed=0))[0])], "bandit_bayes_ucb", "Five execution algorithms: Bayes-UCB and UCB1", "mean regret per day", (0, 0.4))

    # exp3: gamma
    gam = np.linspace(0.01, 0.5, 25)
    Ze = sweep(lambda g: Exp3(ARMS, gamma=float(g), seed=0), gam)
    surface(Ze[:, 19:], gam, "bandit_exp3", "EXP3 AGAINST THE MIXING COEFFICIENT", "five execution algorithms over 1000 days: mean regret per day of Exp3 for 25 values of gamma", "gamma", "mean regret per day", (0, 0.4))
    for g in (0.01, 0.1, 0.5):
        k = int(np.argmin(np.abs(gam - g)))
        print("exp3 gamma", round(float(gam[k]), 3), at(Ze[k], D))
    lines([("Exp3 gamma 0.1", run(lambda: Exp3(ARMS, gamma=0.1, seed=0))[0]), ("UCB1", run(lambda: UCB1(ARMS, seed=0))[0])], "bandit_exp3", "Five execution algorithms: Exp3 and UCB1", "mean regret per day", (0, 0.4))

    # linucb: alpha, with a context
    al = np.linspace(0.02, 2.0, 25)
    Zl = np.array([run_ctx(lambda a=a: LinUCBDisjoint(["long", "short"], alpha=float(a), feature_keys=("trend", "bias"), seed=0)) for a in al])
    surface(Zl[:, 19:], al, "bandit_linucb", "LINUCB AGAINST ITS EXPLORATION WEIGHT", "buy or sell with a trend signal as context over 1000 days: mean regret per day of LinUCBDisjoint for 25 values of alpha", "alpha", "mean regret per day", (0, 0.3))
    for a in (0.02, 0.3, 2.0):
        k = int(np.argmin(np.abs(al - a)))
        print("linucb alpha", round(float(al[k]), 3), at(Zl[k], D))
    lines([("LinUCBDisjoint alpha 0.3", run_ctx(lambda: LinUCBDisjoint(["long", "short"], alpha=0.3, feature_keys=("trend", "bias"), seed=0))), ("UCB1, no context", run_ctx(lambda: UCB1(["long", "short"], seed=0)))], "bandit_linucb", "Buy or sell with a trend signal: LinUCB and UCB1 without context", "mean regret per day", (0, 0.4))

    # offline replay: log length and true value of the evaluated policy
    lens = np.linspace(100, 3000, 25).astype(int)
    vals = np.linspace(0.1, 0.9, 25)
    err = np.zeros((len(vals), len(lens)))
    for i, v in enumerate(vals):
        rng = np.random.default_rng(7)
        log = [({"x": float(rng.random())}, int(rng.integers(2)), None) for _ in range(3000)]
        rr = rng.random(3000)
        log = [(c, a, float(rr[k] < (v if a == 0 else 0.3))) for k, (c, a, _) in enumerate(log)]
        for j, n in enumerate(lens):
            out = offline_replay(log[: int(n)], lambda c: 0, n_bootstrap=20, seed=0)
            err[i, j] = abs(out["mean"] - v)
    X, Y = np.meshgrid(lens, vals)
    fig = plt.figure(figsize=(12, 8.2))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(top=0.93, bottom=0.0, left=-0.05, right=1.05)
    ax.set_facecolor(BG)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.07, 0.10, 0.14, 1.0))
        axis._axinfo["grid"]["color"] = GRID
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((1.35, 1.2, 0.7), zoom=1.1)
    ax.set_xlabel("events in the log", color=CYAN, labelpad=8)
    ax.set_ylabel("true success rate of the policy", color=CYAN, labelpad=8)
    ax.set_zlabel("error of the estimate", color=CYAN, labelpad=6)
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.text(0.05, 0.955, "THE REPLAY ESTIMATE CONVERGES WITH THE LOG", fontsize=21, fontweight="bold", color=INK)
    fig.text(0.05, 0.915, "a log of executions with a random algorithm: error of offline_replay for a policy that always picks arm 0, for 25 log sizes and 25 success rates", fontsize=11.5, color=CYAN)
    ax.plot_surface(X, Y, err, facecolors=CMAP(np.clip(err / max(err.max(), 1e-9), 0, 1)), rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=0.93)
    fig.savefig(HERE / "bandit_offline_replay_3d.png", dpi=150, facecolor=BG)
    plt.close(fig)
    for v, n in ((0.5, 100), (0.5, 1000), (0.5, 3000), (0.9, 3000)):
        i, j = int(np.argmin(np.abs(vals - v))), int(np.argmin(np.abs(lens - n)))
        print("replay value", round(float(vals[i]), 2), "log", int(lens[j]), "error", round(float(err[i, j]), 4))
    rng = np.random.default_rng(8)
    rr = rng.random(3000)
    arms_l = rng.integers(2, size=3000)
    log = [({"x": float(rng.random())}, int(arms_l[k]), float(rr[k] < (0.7 if arms_l[k] == 0 else 0.2))) for k in range(3000)]
    pts = [100, 300, 1000, 3000]
    est, lo, hi = [], [], []
    for n in range(100, 3001, 100):
        o = offline_replay(log[:n], lambda c: 0, n_bootstrap=50, seed=0)
        est.append(o["mean"]); lo.append(o["ci_low"]); hi.append(o["ci_high"])
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=BG)
    ax.set_facecolor(PANEL)
    xs = np.arange(100, 3001, 100)
    ax.fill_between(xs, lo, hi, color=CYAN, alpha=0.25, label="95% bootstrap interval")
    ax.plot(xs, est, color=CYAN, lw=2.0, label="replay estimate")
    ax.axhline(0.7, color=ORANGE, lw=1.4, ls="--", label="true value 0.7")
    ax.set_xlabel("events in the log", color=CYAN)
    ax.set_ylabel("value of the policy", color=CYAN)
    ax.tick_params(colors=MUTED)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(color=GRID, lw=0.5)
    ax.set_title("A log of executions: the replay estimate of 'always the first algorithm'", color=INK, loc="left", fontsize=13, fontweight="bold")
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=INK, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(HERE / "bandit_offline_replay_finance.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("replay estimate/interval at log 100, 300, 1000, 3000:", [(round(est[k // 100 - 1], 3), round(lo[k // 100 - 1], 3), round(hi[k // 100 - 1], 3)) for k in pts])

    # policy page: UCB1 with K arms
    ks = np.arange(2, 13)
    Zk = []
    for k in ks:
        p = np.linspace(0.2, 0.8, k)
        Zk.append(run(lambda: UCB1(list(range(int(k))), seed=0), p=p)[0])
    Zk = np.array(Zk)
    surface(Zk[:, 19:], ks.astype(float), "bandit_policy", "ONE POLICY, MORE ARMS, MORE TO LEARN", "execution algorithms with rates from 0.2 to 0.8: mean regret per day of UCB1 for 2 to 12 arms", "number of arms", "mean regret per day", (0, 0.5))
    for k in (2, 5, 12):
        i = int(np.argmin(np.abs(ks - k)))
        print("policy arms", int(ks[i]), at(Zk[i], D))
    reg, counts = run(lambda: UCB1(ARMS, seed=0))
    lines([("arm " + str(a) + " (rate " + str(P[a]) + ")", counts[:, a]) for a in range(5)], "bandit_policy", "UCB1 on five execution algorithms: how often each arm was pulled", "pulls so far", None, "upper left")
    print("policy counts at 1000:", [int(c) for c in counts[-1]])


if __name__ == "__main__":
    main()
