# The input shape of the linear models

`LinearRegression` and `LogisticRegression` read one sample at a time as a dict. This page is about what the dict has to look like and what the models do when it does not. The overview is on [Linear models](index.md); the models are on the [GLM](glm.md) page.

## Gamestory

London, deep night. One at a time, clue cards (which in the code are called dict) arrive at the young detective's table. On each card there are boxes with a name and a number (for example: time: 2, steps: 3) and on the back the secret answer (y) is written.

So as not to get confused while working super fast, the detective gives three simple rules to his assistant:

- **1. The order of the boxes never changes (feature_keys):** As soon as the first card arrives, the assistant locks in the order of the boxes (or uses an already-made list called feature_keys). From that moment on, all following cards must have the exact same boxes in the exact same order, otherwise the detective won't read them!
- **2. Guess first, flip the card second (step and y):** When a new card arrives, the detective looks at the numbers on the front and immediately makes his guess (the step command gives the answer before updating). Only afterward does he flip the card to read the real answer (y) and learn from his mistake.
- **3. Blank boxes don't stop the game (None and n_missing):** Sometimes a card arrives with an empty box (None) or a word instead of a number. Instead of stopping everything and getting mad, the assistant adds "+1" on the n_missing ("missing clues") sheet, skips that card, and moves right on to the next one!

In the end, what is this rule of GLM models (The input shape)? It is the tidy way the program reads data one piece at a time: it locks in the box order (feature_keys) on the first try, always guesses before looking at the answer (y), and if it finds a gap (None) it counts it in n_missing without ever crashing!

## The dict

Both models read a dict at every sample. The keys of the dict are the feature
names; the values are the numeric features; the target is passed separately
as the second argument of `learn_one`. The first sample fixes the order of
the features: either the explicit `feature_keys` argument, or the sorted
keys of the first dict. Subsequent samples must have the same keys.

```python
import numpy as np
from dense_armor.core.online.optim import SGD
from dense_armor.core.online.linear.glm import LinearRegression

reg = LinearRegression(optimizer=SGD(lr=0.1), intercept=False, feature_keys=["x"])
state = reg.init_state(1)
state, out = reg.step(state, {"x": np.asarray(1.0), "y": np.asarray(3.0)})
print(out)
```

```
0.0
```

The pure form takes the same dict, with the target under the key `"y"`. The
output is the prediction on the current sample, before the update.

## Missing values

A sample with a hole does not stop the loop. A feature that is `NaN` or infinite, or a target that is `None`, a word or `NaN`, is counted in `n_missing` and the sample is skipped, without raising. A key that is absent from the dict counts as `0.0` (a sparse input) and the sample is learned. Of the six samples below, one is learned with the key missing, and three have a hole:

```python
from dense_armor.core.online.optim import SGD
from dense_armor.core.online.linear.glm import LinearRegression

reg = LinearRegression(optimizer=SGD(lr=0.1), intercept=False, feature_keys=["x"])
reg.learn_one({"x": 1.0}, 3.0)
reg.learn_one({"x": float("nan")}, 3.0)
reg.learn_one({"x": 2.0}, None)
reg.learn_one({"x": 2.0}, "six")
reg.learn_one({}, 3.0)
reg.learn_one({"x": 2.0}, 6.0)
print(reg.n_missing)
```

```
3
```

The `NaN` feature, the `None` target and the word `"six"` are counted and skipped (3); the sample with no key (`{}`, learned with `x = 0.0`) and the two normal ones are learned. A `None` as the value of a feature, or `None` instead of the whole dict, is not counted: it raises a `TypeError`.
