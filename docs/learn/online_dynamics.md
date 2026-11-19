# Online dynamics: learn the difference between the formula and reality

A formula from a manual is almost right. A bank's manual says that a payment costs 2 % of its amount; the real fee has a fixed charge on top, and the commission changes in the middle of the year. A robot's manual says how much torque a move needs; the real robot has friction, wear and a package in its hand. The formula is not wrong, it is missing a piece, and the piece changes. The tools of this section learn that missing piece, one sample at a time, while the system runs, and forget it when it changes.

## The map of online dynamics

Click a page to open it, or the job on the left (or the button at the top) to go back to the [map by job](../index.md#find-a-tool-by-what-you-want-to-do).

<object type="image/svg+xml" data="../../assets/maps/online_dynamics.svg" style="width:100%"></object>

## Gamestory

London Theater, deep night. The young detective has to move a mechanical puppet from afar by pulling its strings, and the puppet has an arm made of two pieces joined by hinges.

The detective holds the official instruction manual (the URDF file), which says exactly how much strength is needed to lift the arm. But as soon as the show begins, the arm always stays a bit too low and misses the target! In fact, the manual doesn't know three real-world things: the wooden hinges have friction, the puppet has worn down, and above all someone put a heavy package in its hand! To fix his aim live without stopping the show, the detective turns on Online Dynamics, a helper program with three special tricks:

- **1. Measuring the missing strength (ResidualDynamicsLearner):** The program compares how much strength the manual says with how much strength is actually needed to move the puppet's arm. It calculates the difference (the "extra" force) on the fly and adds it to every move.
- **2. The 5 clues of the hinge (RecursiveLeastSquares):** To figure out where that extra force comes from, the program checks 5 things on each hinge of the puppet, one move at a time: how fast it speeds up, how fast it moves, which way it turns, how much gravity pulls, and a fixed weight. That way it immediately finds out if the arm is heavier or if the hinges are rusty!
- **3. Forgetting the old weight (lambda and DriftAwareResidualDynamicsLearner):** What if halfway through the show the puppet puts down the heavy package or picks up an even bigger one? Thanks to a trick for forgetting the past (lambda), or an alarm (DriftAware) that triggers a super-fast memory as soon as the weight changes, the puppeteer readjusts the strength of the strings in a flash!

In the end, what is Online Dynamics? It is a program that learns in real time the difference between theory (the perfect instruction manual) and reality (extra weight, friction, and wear), allowing the puppeteer — or a real robotic arm — to never miss the target!

## A first example: a fee formula that changes

The manual says that the fee is 2 % of the amount. In reality there is a fixed 0.50 on top, and at payment 300 the commission becomes 3.5 %. Two learners see the same 600 payments, one that weighs every payment the same (`lam=1`) and one that forgets a little all the time (`lam=0.99`). At payment 300 and at payment 600 each is asked for the fee of a payment of 100 (the real fee is 2.50 and then 4.00).

```python
import numpy as np
from dense_armor.core.online.learn.online_dynamics import RecursiveLeastSquares

rng = np.random.default_rng(0)
models = {
    "lam=1": RecursiveLeastSquares(lam=1.0),
    "lam=0.99": RecursiveLeastSquares(lam=0.99),
}
for i in range(600):
    amount = rng.uniform(10, 500)
    fee = (0.02 if i < 300 else 0.035) * amount + 0.5 + rng.normal(0, 0.05)
    x = {"amount": amount, "one": 1.0}
    for m in models.values():
        m.learn_one(x, fee)
    if i + 1 in (300, 600):
        for name, m in models.items():
            print(
                i + 1,
                name,
                round(m.predict_one({"amount": 100.0, "one": 1.0}), 2),
            )
```

```
300 lam=1 2.5
300 lam=0.99 2.49
600 lam=1 3.3
600 lam=0.99 3.93
```

At payment 300 both have found the fixed charge that the manual does not know (2.50 and 2.49, not 2.00). At payment 600 the one that forgets is at 3.93, close to the new fee of 4.00; the one that weighs every payment the same is at 3.30, held back by the first 300 payments. The same idea, applied to the torque of a robot arm, is on the [ResidualDynamicsLearner](residual_dynamics.md) page.

![The fee that a recursive least squares learner predicts for a payment of 100 over 600 payments, with and without forgetting: both find the real fee of 2.50, which is 0.50 above what the manual says; at payment 300 the real fee jumps to 4.00, lam 0.99 follows and lam 1 is still at 3.30 at payment 600.](../assets/learn/rls_tracking.png)

## The three classes, one page each

| Page | Class | What it is for |
|---|---|---|
| [RecursiveLeastSquares](recursive_least_squares.md) | `RecursiveLeastSquares` | a linear model that keeps up with the data, with a forgetting factor `lam`: the engine of the other two |
| [ResidualDynamicsLearner](residual_dynamics.md) | `ResidualDynamicsLearner` | the torque that the model of a robot (its URDF) misses: friction, a payload, wear |
| [DriftAwareResidualDynamicsLearner](drift_aware.md) | `DriftAwareResidualDynamicsLearner` | the same, with an alarm that makes the learner forget faster when the weight changes |

## The robot example

A robot's digital description file is almost always slightly inaccurate because it misses real-world factors like joint friction or a newly picked-up object. This tool continuously learns those missing dynamics in real time, sample by sample, so the control system can add an estimated corrective torque to its command.

A robot's URDF file describes its rigid-body dynamics: mass matrix, Coriolis terms,
gravity. That model is *almost* right. But a real robot also carries a payload in its
gripper, has friction in its joints, wears down over months. None of that is in the
file. The controller that trusts the URDF alone will be slightly wrong, always in the
same direction, and on a heavy payload that "slightly" is enough to miss the target.

`ResidualDynamicsLearner` learns that missing piece, one sample at a time, using one
**Recursive Least Squares** per joint on a five-feature basis. The nominal model comes
from [`urdf_dynamics`](../dynamics/urdf_dynamics.md); the learner adds a residual torque
on top so the controller does not need exact masses or friction coefficients.

The details of the robot version (the two torques, the five features of each joint, a simulated arm with a payload, what happens when a parameter changes in the middle of the run) are on the [ResidualDynamicsLearner](residual_dynamics.md) page, and the variant with an alarm on the [DriftAwareResidualDynamicsLearner](drift_aware.md) page.

## API reference

The API of each class is on its page: [RecursiveLeastSquares](recursive_least_squares.md), [ResidualDynamicsLearner](residual_dynamics.md) and [DriftAwareResidualDynamicsLearner](drift_aware.md).

---

## Details

**Two-step promotion** for the underlying dynamics model — Dense-Evolution-Discovery
Experiment 61 built the same Euler-Lagrange dynamics but hardcoded to one Kinova Gen3;
Experiment 62 replaced the hardcoded tables with a real URDF parser and re-validated.
The learner itself was re-validated on the 2-link benchmark after that.

**RLS references**: Ljung, L., Soderstrom, T. (1983). *Theory and Practice of Recursive
Identification*. MIT Press. RLS with exponential forgetting.

**Triggers**: Page, E. S. (1954). Continuous inspection schemes. *Biometrika* 41,
100–114. Hampel, F. R. (1974). The influence curve and its role in robust estimation.
*JASA* 69(346), 383–393.

**See also**: [Rigid-body dynamics](../dynamics/urdf_dynamics.md) — the nominal model
the learner builds on. [Calibration](calibration.md) — if the residual is used as a
score, calibration makes the score interpretable.
