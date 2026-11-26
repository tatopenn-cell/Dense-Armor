# Stacking: a second model on top

[← Ensembles](index.md)

`StackingClassifier` trains a list of base classifiers and a second classifier, the meta-model,
on the predictions of the base ones. For each sample the base members predict first, their
predictions become the input of the meta-model, and only then the members see the label. The
order matters: if the meta-model saw the labels before the members had predicted, it would learn
from information the members did not have, and the ensemble would look better on the stream than
it is.

```python
import numpy as np
from dense_armor.core.online.ensemble import StackingClassifier
from dense_armor.core.online.tree.hoeffding import HoeffdingTreeClassifier

rng = np.random.default_rng(0)
stack = StackingClassifier(
    [HoeffdingTreeClassifier(grace_period=20),
     HoeffdingTreeClassifier(grace_period=100)],
    HoeffdingTreeClassifier(grace_period=20))
for _ in range(500):
    x = float(rng.normal(0.0 if rng.random() < 0.5 else 4.0, 0.5))
    stack.learn_one({"x": x}, int(x > 2.0))
print(stack.predict_one({"x": 0.1}), stack.predict_one({"x": 4.0}))
```

```
0 1
```

The meta-model sees only the members' predictions, never the raw `x`. On the drifted stream of
the cost section of the [index](index.md#what-the-ensembles-cost) stacking is on par with voting and both are above the single tree.

---

## Details

### Source

Stacking has no paper in the task folder. Its rule is stated in the class docstring and tested.

### Choices

- **The meta-model of stacking.** The paper gives no rule for what the meta-model sees. Here it
  sees the members' predicted labels, one key per member, and never the raw input.

### Limits

- The stacking meta-model receives the members' predicted labels, not their probabilities. A
  meta-model that could see the probabilities would have more to work with; the class does not
  pass them.

### Checks

- The stacking order (meta-model learns before the members see the label) is verified by a test
  with a spy classifier.
