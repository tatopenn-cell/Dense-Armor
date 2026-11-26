"""Generate figures for the Metrics section pages.

Run from the repo root:
    python docs/assets/metrics/make_metrics_pages.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dense_armor.core.evaluation.metrics import (
    Accuracy, Precision, Recall, F1, FBeta, BalancedAccuracy, 
    ConfusionMatrix, CohenKappa, LogLoss, BrierScore
)

OUT = Path(__file__).parent

# ============== BASICS FIGURE ==============
# Generate 100 pairs with 80% correct predictions
rng = np.random.default_rng(42)
n = 100
y_true = rng.integers(0, 2, size=n).tolist()

# 80% correct
y_pred = y_true.copy()
correct_indices = rng.choice(n, size=int(0.8 * n), replace=False)
for idx in correct_indices:
    y_pred[idx] = y_true[idx]

# For incorrect ones, flip the prediction
incorrect_indices = [i for i in range(n) if i not in correct_indices]
for idx in incorrect_indices:
    y_pred[idx] = 1 - y_true[idx]

# Compute accuracy after each pair
acc = Accuracy()
acc_history = []
for yt, yp in zip(y_true, y_pred):
    acc.update(yt, yp)
    acc_history.append(acc.get())

# Final accuracy
final_acc = acc.get()
print(f"Final accuracy: {final_acc:.3f} ({int(final_acc * 100)}%)")

# Plot for basics
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.plot(range(1, n + 1), acc_history, "k-", lw=1.6, label="Accuracy")
ax.axhline(final_acc, ls="--", color="red", lw=1.5, label=f"final accuracy = {final_acc:.2f}")
ax.set_xlim(1, n)
ax.set_ylim(0, 1)
ax.set_xlabel("sample")
ax.set_ylabel("accuracy")
ax.legend(fontsize=9)
ax.set_title(f"Accuracy updated after each pair (final {int(final_acc*100)}%)")
fig.tight_layout()
fig.savefig(OUT / "acc_example.png", dpi=130)

print(f"Figure saved to {OUT / 'acc_example.png'}")

# ============== CLASSIFICATION FIGURE ==============
# Generate data: 200 pairs
rng = np.random.default_rng(42)
n = 200

y_true = [1] * 100 + [0] * 100
y_pred = []
for i in range(200):
    if i < 100:  # class 1
        y_pred.append(1 if rng.random() < 0.85 else 0)
    else:  # class 0
        y_pred.append(0 if rng.random() < 0.75 else 1)

# Compute all metrics
acc = Accuracy()
prec = Precision(positive=1)
rec = Recall(positive=1)
f1 = F1(positive=1)
bal = BalancedAccuracy()
cm = ConfusionMatrix()

for yt, yp in zip(y_true, y_pred):
    acc.update(yt, yp)
    prec.update(yt, yp)
    rec.update(yt, yp)
    f1.update(yt, yp)
    bal.update(yt, yp)
    cm.update(yt, yp)

print(f"Classification: Acc={acc.get():.3f}, Prec={prec.get():.3f}, Rec={rec.get():.3f}, F1={f1.get():.3f}, BalancedAcc={bal.get():.3f}")

# Confusion matrix figure
cm_data = cm.get()
labels = [0, 1]
cm_array = np.array([[cm_data[l][p] for p in labels] for l in labels])

fig_cm, ax_cm = plt.subplots(figsize=(5, 3))
im = ax_cm.imshow(cm_array, cmap="Blues")
for i in range(2):
    for j in range(2):
        text = ax_cm.text(j, i, f"{cm_array[i, j]:.0f}", ha="center", va="center")
ax_cm.set_xticks([0, 1])
ax_cm.set_yticks([0, 1])
ax_cm.set_xticklabels(["pred 0", "pred 1"])
ax_cm.set_yticklabels(["true 0", "true 1"])
ax_cm.set_title("Confusion Matrix")
fig_cm.tight_layout()
fig_cm.savefig(OUT / "confusion_matrix.png", dpi=130)

# Metrics comparison bars
class_counts = ["Precision", "Recall", "F1", "BalancedAccuracy"]
values = [prec.get(), rec.get(), f1.get(), bal.get()]

fig_bar, ax_bar = plt.subplots(figsize=(6, 3.2))
bars = ax_bar.bar(class_counts, values, color=["steelblue", "steelblue", "steelblue", "steelblue"])
ax_bar.axhline(0.5, ls="--", color="red", lw=1, label="random")
ax_bar.set_ylim(0, 1)
ax_bar.set_ylabel("score")
ax_bar.legend(fontsize=8)
ax_bar.set_title(f"Classification metrics (final values)")
fig_bar.tight_layout()
fig_bar.savefig(OUT / "class_metrics.png", dpi=130)

print(f"Figures saved to {OUT}")

# ============== PROBABILITIES FIGURE ==============
# Generate log loss and Brier score vs probability
probs = np.linspace(0.01, 0.99, 50)

ll_values = [-np.log(p) for p in probs]  # when true=1
br_values = [(1-p)**2 for p in probs]  # when true=1

fig_prob, ax_prob = plt.subplots(figsize=(6, 3.2))
ax_prob.plot(probs, ll_values, "b-", lw=1.5, label="Log loss")
ax_prob.plot(probs, br_values, "r-", lw=1.5, label="Brier score")
ax_prob.axvline(0.5, ls=":", color="gray", alpha=0.5, label="50% confidence")
ax_prob.set_xlabel("assigned probability")
ax_prob.set_ylabel("score")
ax_prob.legend(fontsize=8)
ax_prob.set_title("Log loss and Brier score vs assigned probability")
fig_prob.tight_layout()
fig_prob.savefig(OUT / "probabilities.png", dpi=130)

print(f"Figure saved to {OUT / 'probabilities.png'}")
print("All figures generated successfully!")