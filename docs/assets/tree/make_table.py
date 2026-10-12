"""Generate the comparison table in docs/tree/index.md section 6.

Stream specification (must match the text in the page):
- 4000 samples, tested before each learn_one.
- Classifiers: features a, b. Target y in {0, 1}.
  Stationary: a ~ N(3y, 0.5), b ~ N(0, 0.5).
  Change at sample 2000: a ~ N(3(1-y), 0.5), b unchanged.
- Regressors: two-link gravity torque, q1, q2 ~ U(-pi, pi).
  tau = 9.81 * (2 cos q1 + cos(q1 + q2)) * scale
  Stationary scale = 1.0, change at sample 2000 -> scale = 1.5.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import time
import numpy as np

from dense_armor.utility.metrics import Accuracy, MeanAbsoluteError
from dense_armor.utility.drift.adwin import ADWIN
from dense_armor.utility.tree.hoeffding import HoeffdingTreeClassifier
from dense_armor.utility.tree.efdt import HoeffdingAnytimeTreeClassifier
from dense_armor.utility.tree.adaptive import HoeffdingAdaptiveTreeClassifier
from dense_armor.utility.tree.sgt import SGTClassifier, SGTRegressor
from dense_armor.utility.tree.mondrian import (
    MondrianForestClassifier, MondrianForestRegressor,
)

N = 4000
CHANGE_AT = 2000


def cls_stream(change):
    rng = np.random.default_rng(42)
    for i in range(N):
        y = int(rng.random() < 0.5)
        if change and i >= CHANGE_AT:
            a = rng.normal(3.0 * (1 - y), 0.5)
        else:
            a = rng.normal(3.0 * y, 0.5)
        b = rng.normal(0.0, 0.5)
        yield {"a": float(a), "b": float(b)}, y


def reg_stream(change):
    rng = np.random.default_rng(42)
    for i in range(N):
        q1, q2 = rng.uniform(-np.pi, np.pi, 2)
        scale = 1.5 if (change and i >= CHANGE_AT) else 1.0
        tau = scale * 9.81 * (2.0 * np.cos(q1) + np.cos(q1 + q2))
        yield {"q1": float(q1), "q2": float(q2)}, float(tau)


def score_cls(model, stream):
    acc = Accuracy()
    t0 = time.perf_counter()
    for x, y in stream:
        p = model.predict_one(x)
        if p is not None:
            acc.update(y, p)
        model.learn_one(x, y)
    return acc.get(), (time.perf_counter() - t0) / N * 1000


def score_reg(model, stream):
    mae = MeanAbsoluteError()
    t0 = time.perf_counter()
    for x, y in stream:
        p = model.predict_one(x)
        if p is not None:
            mae.update(y, p)
        model.learn_one(x, y)
    return mae.get(), (time.perf_counter() - t0) / N * 1000


rows = []
for name, factory in [
    ("HoeffdingTreeClassifier",        lambda: HoeffdingTreeClassifier(grace_period=100)),
    ("HoeffdingAnytimeTreeClassifier", lambda: HoeffdingAnytimeTreeClassifier(grace_period=100)),
    ("HoeffdingAdaptiveTreeClassifier",lambda: HoeffdingAdaptiveTreeClassifier(grace_period=100, drift_detector=ADWIN())),
    ("SGTClassifier",                  lambda: SGTClassifier()),
    ("MondrianForestClassifier",       lambda: MondrianForestClassifier(n_trees=10, seed=0)),
]:
    s, ms_s = score_cls(factory(), cls_stream(False))
    c, ms_c = score_cls(factory(), cls_stream(True))
    ms = (ms_s + ms_c) / 2
    rows.append((name, "accuracy", s, c, ms))

for name, factory in [
    ("SGTRegressor",             lambda: SGTRegressor()),
    ("MondrianForestRegressor",  lambda: MondrianForestRegressor(n_trees=10, seed=0)),
]:
    s, ms_s = score_reg(factory(), reg_stream(False))
    c, ms_c = score_reg(factory(), reg_stream(True))
    ms = (ms_s + ms_c) / 2
    rows.append((name, "mean abs. error", s, c, ms))

print("| model | task | stationary | with a change | ms per sample |")
print("|---|---|---|---|---|")
for name, task, s, c, ms in rows:
    print(f"| {name} | {task} | {s:.3f} | {c:.3f} | {ms:.3f} |")
