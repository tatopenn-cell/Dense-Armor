# Speed, acceleration, jerk and power

## 4. Speed, acceleration and jerk from positions

The robot reports positions; velocity, acceleration and jerk come from differences between
samples divided by the real time between them, even when the samples are not evenly spaced.

```python
from dense_armor.utility.preprocessing.joints import JointDerivatives

jd = JointDerivatives(order=3)
for t in [0.0, 0.3, 1.1]:
    jd.learn_one({"q": [t ** 3]}, t=t)
out = jd.transform_one({"q": [2.0 ** 3]}, t=2.0)
print([round(out[k][0], 6) for k in ("qd", "qdd", "jerk")])
```

```
[12.0, 12.0, 6.0]
```

Positions follow $q = t^3$ at uneven times 0, 0.3, 1.1, 2.0; at $t = 2$ the exact values are
$3t^2 = 12$, $6t = 12$, $6$: the method is exact on a cubic. It uses the Newton divided
differences of the last four samples; the current sample is included, so the value belongs to
the present moment, not to the previous sample. Divided differences are the natural generalisation
of "difference divided by time" to several unevenly spaced points: they are built recursively so
that the resulting polynomial passes through all the samples, and differentiating it at the last
point gives the derivative there.

## 5. Power

```python
from dense_armor.utility.preprocessing.joints import JointPower

out = JointPower().transform_one({"tau": [1.0, 2.0], "qd": [3.0, 4.0]})
print(out["power"], out["power_total"])
```

```
[3.0, 8.0] 11.0
```

Mechanical power of a joint is torque times angular velocity,

$$P = \tau\,\dot q$$

so the first joint contributes $1 \times 3 = 3$, the second $2 \times 4 = 8$, and the total is
$3 + 8 = 11$.
