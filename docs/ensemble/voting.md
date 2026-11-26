# Voting: the majority decides

[← Ensembles](index.md)

`VotingClassifier` trains a list of classifiers on the same stream and returns the class that
most members predict. In `"hard"` mode each member is one vote, whatever confidence it has; in
`"soft"` mode each member returns a probability for every class and the ensemble averages them.

```python
from dense_armor.core.online.ensemble import VotingClassifier
from dense_armor.core.online.tree.hoeffding import HoeffdingTreeClassifier

v = VotingClassifier(
    [HoeffdingTreeClassifier(grace_period=20),
     HoeffdingTreeClassifier(grace_period=50),
     HoeffdingTreeClassifier(grace_period=100)],
    mode="hard")
for i in range(200):
    x = (i % 20) / 19.0
    v.learn_one({"x": x}, 1 if x > 0.5 else 0)
print(v.predict_one({"x": 0.9}))
```

```
1
```

The three members differ in how often they reconsider a split, so they make different mistakes,
and the majority cancels some of them. A vote of three identical members cannot help: they would
all be wrong together. The members here differ by `grace_period`, which is enough on this
stream.

---

## Details

### Choices

- **The members of voting and stacking.** The papers give no rule for how different the members
  should be. Here they differ by `grace_period`; three identical members cannot help, and a test
  checks that a vote of diverse members is not worse than a single tree.

### Source

Voting and stacking have no paper in the task folder. Their rules are stated in the class docstrings and tested.
