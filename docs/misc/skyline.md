# Skyline: the best points in a sliding window

A laptop shop lists its models on two numbers: price and weight. A buyer who wants both cheap and light has no single "best" model: the cheapest is heavy, the lightest is expensive. The models that are not beaten on both at once are the *skyline*. This page explains what it is, why a window is needed when the points arrive one at a time, and how the Skyline class keeps it up to date.

## The scene

Suppose three models arrive on the shelf, one after the other. A: 800 EUR, 1.2 kg. B: 1000 EUR, 1.2 kg. C: 1000 EUR, 1.0 kg.

A beats B (same weight, cheaper). A does not beat C on weight (1.2 > 1.0). C does not beat A on price (1000 > 800). So the skyline is {A, C}: A is the cheap one, C is the light one, neither is beaten on both.

The class answers this question for a stream of points, one at a time, with a window: only the last W points count. A model that was on the skyline three days ago may leave the shelf, and its line disappears.

## What is in the box

| Tool | Its one job | Example |
|---|---|---|
| Skyline(window=W) | Keep the last W points, report the non-dominated ones | With three laptops, returns {A, C} |
| update(p) | Push one point, drop the oldest when the window is full | `s.update({'price': 800, 'weight_kg': 1.2})` |
| skyline() | Return the indices of the non-dominated points in the window | `[0, 2]` |

## 1. The definition, in one line

A point `p` dominates a point `q` when `p` is at least as good as `q` on every dimension and strictly better on at least one. Ties (equal on every dimension) are not dominance: neither point wins. The skyline is the set of points of the window that no other point of the window dominates.

Börzsönyi, Kossmann and Stocker gave this definition in 2001 (Section 2, page 2), and Tao and Papadias extended it to a sliding window in 2006 (Section 3, page 3). In Skyline, "better" means smaller on every dimension. To have larger-is-better, negate the inputs.

## 2. One point at a time

The class stores the points in a deque with a maximum length. When a new point arrives and the window is already full, the oldest leaves first. Then `skyline()` recomputes the non-dominated set with the block nested loops algorithm (Börzsönyi et al. 2001, Section 3.3, page 4): for each point, compare against every other point of the window; if no other point dominates it, keep it.

The recomputation is O(W^2) per update, where W is the window size. The paper also gives an incremental algorithm (Tao and Papadias 2006), which is not implemented here: the module docstring says so explicitly, and the choice is restated in Details.

```python
from dense_armor.utility.misc import Skyline

s = Skyline(window=3)
for p in [
    {"price": 800.0, "weight_kg": 1.2},
    {"price": 1000.0, "weight_kg": 1.2},
    {"price": 1000.0, "weight_kg": 1.0},
]:
    _ = s.update(p)
print(sorted(s.skyline()))
```

```
[0, 2]
```

The three points fit in the window. Point 1 (1000 EUR, 1.2 kg) is dominated by point 0 (800 EUR, 1.2 kg). Points 0 and 2 are both on the skyline: A is cheaper, C is lighter, nobody is better on both.

## 3. The window, and what leaves it

A sliding window forgets. A point that was on the skyline five minutes ago can leave the window and stop being counted. The point that leaves is the oldest one, not the worst one.

```python
from dense_armor.utility.misc import Skyline

s = Skyline(window=2)
for p in [
    {"price": 500.0, "weight_kg": 1.0},
    {"price": 800.0, "weight_kg": 1.2},
    {"price": 600.0, "weight_kg": 1.5},
]:
    _ = s.update(p)
print(s.skyline())
```

```
[0, 1]
```

The window holds two points at a time. After the third update the window is (800 EUR, 1.2 kg) and (600 EUR, 1.5 kg). The second is cheaper but heavier, the first is lighter but more expensive: neither dominates the other, so both stay on the skyline.

## 4. Inputs that do not fit

A point that is missing a dimension, that is not a dict, or that carries `NaN` or `inf` is counted in `n_missing` and dropped; it does not enter the window and does not raise.

```python
from dense_armor.utility.misc import Skyline

s = Skyline(window=3)
_ = s.update({"price": 800.0, "weight_kg": 1.2})
_ = s.update({"price": float("nan"), "weight_kg": 1.0})
_ = s.update({"price": 100.0})
_ = s.update("not a dict")
print(s.n_seen_, s.n_missing, len(s.skyline()))
```

```
1 3 1
```

One point entered the window, three were dropped. The skyline has one element: the only valid point.

## 5. What this is used for

The class is used in the factorial recognition benchmark: it takes the sliding-DFT features of consecutive frames and reports the ones that are not beaten on every dimension. A skyline of size one means a single dominant feature; a skyline of size five means several trade-offs to consider. The recognition page shows the plot.

The class has no classifier and no regressor: it is a summary structure. Feed it a stream of points, read the skyline, decide what to do with it. That is the whole interface.

## Details

### References

- Börzsönyi, S., Kossmann, D., Stocker, K. (2001). *The Skyline Operator.* In ICDE 2001. Section 2, page 2 (dominance); Section 3.3, page 4 (block nested loops).
- Tao, Y., Papadias, D. (2006). *Maintaining Sliding Window Skylines on Data Streams.* In ICDE 2006. Section 3, page 3 (sliding window maintenance).


### Choices stated explicitly

- The recomputation is O(W^2) per update, from the block nested loops algorithm (Börzsönyi et al. 2001, Section 3.3, page 4). The paper also gives an incremental algorithm for windows (Tao and Papadias 2006), which is not implemented. The module docstring says so explicitly and gives the reason: W is small in the tests, and the naive algorithm is easy to verify against the direct definition.
- Better means smaller on every dimension. To use larger-is-better, negate the inputs. This is a convention, not a parameter.
- Points that do not fit (missing dimension, not a dict, NaN, inf) are counted in `n_missing` and dropped, never raised. This is the same convention as the rest of the library.
- The class exposes `budget_s = 1.5e-2` and `memory_class = O(window * dims)` as class attributes, so that a pipeline that runs it can check that the budget fits the loop.

### Measured cost

The benchmark `benchmarks/facto_quality.py` contains `check_skyline_equals_bruteforce`: it runs `Skyline(window=10)` on 50 random points and compares the result with a brute-force computation of the skyline. The check passes on every point. The measured time per update is not recorded on this page; the class attribute `budget_s = 1.5e-2` is the number the code declares for the check of the budget.

### Development diary

This page was written after the main work of the branch, on request of the owner, because the class existed with a full docstring and a test but without a page of its own: a reader could find it in the API list and not know what it was for. The page is added on the same branch, in a separate commit, so that the owner can move it to a different PR if wanted.
