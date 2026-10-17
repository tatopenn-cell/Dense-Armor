"""Quality benchmark: the bandit and active packages of Task 15.

Four blocks, each written to ``benchmarks/bandit_quality.json``:

1. Context-free bandits on a stream with one sudden change. Four arms
   with Bernoulli rewards whose distributions swap at half. Six policies
   (EpsilonGreedy, UCB1, UCB1Tuned, ThompsonSampling, BayesUCB, Exp3),
   the oracle regret (best arm at every step) and the static regret
   (best fixed arm). Median, IQR and ms per decision over the seeds; a
   paired two-sided Wilcoxon between each policy and the baseline.

2. Contextual bandit on the same stream, context ``phase`` (0 before
   the change, 1 after) and a constant. LinUCBDisjoint against UCB1,
   which ignores the context.

3. Active learning: four strategies on the same drifting stream, with
   OnlineGaussianNB as the classifier. Median labelling rate over the
   seeds, the budget, and the ratio.

4. Robot test: UCB chooses the gain ``kp`` of the kinematic tracking
   controller of SyntheticArm among five values. The chosen gain, the
   regret against the best fixed gain and the time per decision.

Machine and Python are in the JSON header. ``--quick`` shortens the
run for a smoke test. The run takes about two minutes on a 2-core CPU.

Run: ``python benchmarks/bandit_quality.py``.
"""

import argparse
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from dense_armor.utility.active import (
    FixedUncertainty,
    RandomStrategy,
    VariableRandomizedUncertainty,
    VariableUncertainty,
)
from dense_armor.utility.bandit import (
    UCB1,
    BayesUCB,
    EpsilonGreedy,
    Exp3,
    LinUCBDisjoint,
    RobotKpSelector,
    ThompsonSampling,
    UCB1Tuned,
)
from dense_armor.utility.datasets import DriftStream
from dense_armor.utility.learn.online_classifiers import OnlineGaussianNB

N_STEPS = 2000
DRIFT_AT = 1000
ARMS = [0, 1, 2, 3]
P_BEFORE = [0.3, 0.7, 0.5, 0.4]
P_AFTER = [0.7, 0.3, 0.5, 0.4]
POLICY_NAMES = [
    "EpsilonGreedy",
    "UCB1",
    "UCB1Tuned",
    "ThompsonSampling",
    "BayesUCB",
    "Exp3",
]
CTX_POLICY_NAMES = ["UCB1", "LinUCBDisjoint"]
ACTIVE_NAMES = [
    "RandomStrategy",
    "FixedUncertainty",
    "VariableUncertainty",
    "VariableRandomizedUncertainty",
]
ACTIVE_BUDGET = 0.10
ROBOT_ARMS = [0.5, 8.0, 30.0]


def _machine():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }


def _policy(name, seed, contextual=False):
    if name == "EpsilonGreedy":
        return EpsilonGreedy(arms=ARMS, epsilon=0.1, seed=seed)
    if name == "UCB1":
        return UCB1(arms=ARMS, seed=seed)
    if name == "UCB1Tuned":
        return UCB1Tuned(arms=ARMS, seed=seed)
    if name == "ThompsonSampling":
        return ThompsonSampling(arms=ARMS, family="bernoulli", seed=seed)
    if name == "BayesUCB":
        return BayesUCB(
            arms=ARMS, family="bernoulli", horizon=N_STEPS, seed=seed
        )
    if name == "Exp3":
        return Exp3(arms=ARMS, gamma=0.1, seed=seed)
    if name == "LinUCBDisjoint":
        return LinUCBDisjoint(
            arms=ARMS, alpha=0.3, feature_keys=("phase", "bias")
        )
    raise ValueError(f"unknown policy {name}")


def _reward(rng, t, arm):
    p = P_BEFORE if t < DRIFT_AT else P_AFTER
    return float(rng.binomial(1, p[arm]))


def _oracle(t):
    p = P_BEFORE if t < DRIFT_AT else P_AFTER
    return max(p)


def _static_best():
    return max(
        (
            DRIFT_AT * P_BEFORE[a] + (N_STEPS - DRIFT_AT) * P_AFTER[a]
        )
        / N_STEPS
        for a in ARMS
    )


def _context(t):
    return {"phase": float(t >= DRIFT_AT), "bias": 1.0}


def _run_policy(name, seed, contextual=False):
    policy = _policy(name, seed, contextual)
    rng = np.random.default_rng(seed * 7 + 13)
    reg_oracle = 0.0
    reg_static = 0.0
    total = 0.0
    latencies = []
    for t in range(N_STEPS):
        ctx = _context(t) if contextual else None
        t0 = time.perf_counter()
        arm = policy.pull(context=ctx)
        dt = time.perf_counter() - t0
        latencies.append(dt)
        r = _reward(rng, t, arm)
        if contextual:
            _ = policy.update(arm, r, context=ctx)
        else:
            _ = policy.update(arm, r)
        reg_oracle += _oracle(t) - r
        total += r
    best = _static_best()
    reg_static = N_STEPS * best - total
    return {
        "seed": seed,
        "regret_oracle": reg_oracle,
        "regret_static": reg_static,
        "ms_per_decision": 1e3 * float(np.median(latencies)),
        "total_reward": total,
    }


def _run_block(names, seeds, contextual=False):
    rows = defaultdict(list)
    for name in names:
        for seed in range(seeds):
            r = _run_policy(name, seed, contextual)
            rows[name].append(r)
    out = {}
    for name in names:
        data = rows[name]
        ro = np.array([r["regret_oracle"] for r in data])
        rs = np.array([r["regret_static"] for r in data])
        ms = np.array([r["ms_per_decision"] for r in data])
        out[name] = {
            "regret_oracle_median": float(np.median(ro)),
            "regret_oracle_iqr": float(np.percentile(ro, 75) - np.percentile(ro, 25)),
            "regret_static_median": float(np.median(rs)),
            "ms_per_decision_median": float(np.median(ms)),
            "per_seed_oracle": ro.tolist(),
        }
    tests = []
    base = names[0]
    base_vals = np.array(out[base]["per_seed_oracle"])
    for name in names[1:]:
        vals = np.array(out[name]["per_seed_oracle"])
        try:
            p = float(wilcoxon(base_vals, vals, alternative="two-sided").pvalue)
        except ValueError:
            p = 1.0
        tests.append({"a": base, "b": name, "p_two_sided": p})
    return {"policies": out, "pairwise_vs": base, "tests": tests}


def _run_active(seeds):
    out = {}
    for name in ACTIVE_NAMES:
        rates = []
        for seed in range(seeds):
            if name == "RandomStrategy":
                strat = RandomStrategy(budget=ACTIVE_BUDGET, seed=seed)
            elif name == "FixedUncertainty":
                strat = FixedUncertainty(
                    theta=0.7, budget=ACTIVE_BUDGET, seed=seed
                )
            elif name == "VariableUncertainty":
                strat = VariableUncertainty(
                    budget=ACTIVE_BUDGET, s=0.001, seed=seed
                )
            else:
                strat = VariableRandomizedUncertainty(
                    budget=ACTIVE_BUDGET, s=0.001, delta=0.05, seed=seed
                )
            model = OnlineGaussianNB()
            ds = DriftStream(
                n_samples=N_STEPS, n_features=4, n_drifts=1,
                kind="sudden", seed=seed,
            )
            for sig, y in ds.stream():
                x = sig.to_dict()
                ask = strat.should_label(x, model)
                if ask:
                    model.learn_one(x, int(y))
            rates.append(strat.n_labelled_ / max(strat.n_seen_, 1))
        out[name] = {
            "rate_median": float(np.median(rates)),
            "rate_iqr": float(np.percentile(rates, 75) - np.percentile(rates, 25)),
            "per_seed": rates,
        }
    return out


def _run_robot(seeds, rounds):
    picks = []
    regrets = []
    latencies = []
    for seed in range(seeds):
        sel = RobotKpSelector(
            arms=ROBOT_ARMS, period_s=0.5, n_cycles=2, seed=seed
        )
        t0 = time.perf_counter()
        for _ in range(rounds):
            sel.run_round()
        dt = time.perf_counter() - t0
        latencies.append(dt / rounds)
        picks.append(sel.best_gain())
        # regret vs the best fixed gain, measured on the same dataset
        from dense_armor.utility.bandit.robot import _episode
        best_fixed = max(
            (a for a in ROBOT_ARMS),
            key=lambda a: _episode(sel.ds_, a),
        )
        best_reward = _episode(sel.ds_, best_fixed)
        chosen_reward = sel.rewards_.get(sel.best_gain(), -np.inf)
        regrets.append(best_reward - chosen_reward)
    return {
        "picks": picks,
        "regret_median": float(np.median(regrets)),
        "regret_max": float(np.max(regrets)),
        "seconds_per_decision_median": float(np.median(latencies)),
    }


def main(seeds, quick, out_path):
    global N_STEPS, DRIFT_AT
    rounds_robot = 5 if quick else 15
    if quick:
        N_STEPS = 300
        DRIFT_AT = 150

    payload = {"machine": _machine(), "python": platform.python_version(),
               "seeds": seeds, "quick": quick, "n_steps": N_STEPS,
               "drift_at": DRIFT_AT}

    print("=== context-free bandits ===")
    payload["context_free"] = _run_block(POLICY_NAMES, seeds, contextual=False)
    for name, d in payload["context_free"]["policies"].items():
        print(f"  {name:18s} regret_oracle_med={d['regret_oracle_median']:7.2f}  "
              f"regret_static_med={d['regret_static_median']:7.2f}  "
              f"ms={d['ms_per_decision_median']*1000:.1f}us")

    print("=== contextual bandit ===")
    payload["contextual"] = _run_block(CTX_POLICY_NAMES, seeds, contextual=True)
    for name, d in payload["contextual"]["policies"].items():
        print(f"  {name:18s} regret_oracle_med={d['regret_oracle_median']:7.2f}  "
              f"regret_static_med={d['regret_static_median']:7.2f}")

    print("=== active learning ===")
    payload["active"] = _run_active(seeds)
    for name, d in payload["active"].items():
        print(f"  {name:30s} rate_med={d['rate_median']:.4f}  "
              f"budget={ACTIVE_BUDGET}")

    print("=== robot test ===")
    payload["robot"] = _run_robot(seeds, rounds_robot)
    print(f"  picks: {payload['robot']['picks']}")
    print(f"  regret_median={payload['robot']['regret_median']:.4f}")

    payload["active_budget"] = ACTIVE_BUDGET
    payload["robot_arms"] = ROBOT_ARMS
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print()
    print(f"JSON: {out_path}  ({out_path.stat().st_size} B)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="benchmarks/bandit_quality.json")
    args = ap.parse_args()
    main(args.seeds, args.quick, Path(args.out))
