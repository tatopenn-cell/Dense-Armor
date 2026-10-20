# Scaling features and joints

## 1. Same scale for every feature

A standard scaler subtracts the running mean and divides by the running standard deviation, so
every feature lives around 0 with spread 1.

```python
from dense_armor.utility.preprocessing.scale import StandardScaler

sc = StandardScaler()
for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
    sc.learn_one({"x": v})
print(round(sc.transform_one({"x": 5.0})["x"], 4))
```

```
1.2649
```

The mean of 1, 2, 3, 4, 5 is 3, the standard deviation is $\sqrt{2.5} = 1.581$, so
$(5 - 3) / 1.581 = 1.2649$. In general the scaler computes

$$z = \frac{x - \bar x}{s}$$

where $\bar x$ and $s$ are the running mean and standard deviation seen so far. Both are updated
one sample at a time, so the scaler never needs to store the past values.

## 2. One statistic per joint

A robot arm sends its joints as one vector; each joint gets its own mean and spread.

```python
from dense_armor.utility.preprocessing.scale import StandardScaler

sc = StandardScaler()
for q in ([0.0, 10.0], [1.0, 20.0], [2.0, 30.0]):
    sc.learn_one({"q": q})
print([round(v, 3) for v in sc.transform_one({"q": [2.0, 30.0]})["q"]])
```

```
[1.0, 1.0]
```

Both joints are at the top of their own range, so both get the same standardised value, even
though the second joint moves ten times more. The scaler keeps one mean and one standard
deviation per joint, not one for the whole vector.

## 3. Robust to spikes

The robust scaler uses the rolling median and the median absolute deviation (MAD, times 1.4826
so it matches the standard deviation on Gaussian data); one spike barely moves it.

```python
from dense_armor.utility.preprocessing.scale import RobustScaler, StandardScaler

rs, ss = RobustScaler(window=20), StandardScaler()
for v in [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 100.0]:
    rs.learn_one({"x": v})
    ss.learn_one({"x": v})
print(round(rs.transform_one({"x": 3.0})["x"], 3), round(ss.transform_one({"x": 3.0})["x"], 3))
```

```
0.674 -0.303
```

The same value 3.0 after a spike of 100: the robust scaler sees it as 0.674 spreads above the
median (median 2, MAD-based spread 1.483); the standard scaler, pulled by the spike, sees it as below average. The
median is the middle value of the sorted window, so a single extreme sample does not move it;
the MAD is the median of the distances from the median, so it measures the typical spread
without ever squaring the spike. The factor 1.4826 turns the MAD into the same scale as a
standard deviation when the data is roughly Gaussian.
