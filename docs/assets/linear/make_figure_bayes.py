import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dense_armor.utility.linear.bayes import BayesianLinearRegression

out = Path(__file__).parent
rng = np.random.default_rng(0)

n = 10
X = rng.uniform(-1, 1, size=(n, 1))
y = 2.0 * X[:, 0] + 0.05 * rng.normal(size=n)

model = BayesianLinearRegression(alpha=1.0, beta=1.0 / 0.01,
                                  intercept=False, feature_keys=["x"])
for i in range(n):
    model.learn_one({"x": float(X[i, 0])}, float(y[i]))

xs = np.linspace(-3.0, 3.0, 200)
means = np.array([model.predict_one({"x": float(x)}) for x in xs])
stds = np.array([
    np.sqrt(1.0 / (1.0 / 0.01)
            + float(np.array([x]) @ model._S @ np.array([x])))
    for x in xs
])

fig, ax = plt.subplots(figsize=(8, 3.2))
ax.scatter(X[:, 0], y, color="black", s=18, label="samples")
ax.plot(xs, means, color="tab:blue", lw=1.5, label="posterior mean")
ax.fill_between(xs, means - 2 * stds, means + 2 * stds,
                color="tab:blue", alpha=0.18,
                label="mean +/- 2 std")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(out / "bayes_var.png", dpi=130)
print("scritto:", out / "bayes_var.png")
