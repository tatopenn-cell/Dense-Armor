# Drift: the algorithms

A change detector reads a series of values and says where its level shifts. CUSUMDriftDetector fixes the reference level from the first values and adds up the deviations that follow until the sum crosses a threshold.

Every class reads the whole dataset with `fit` and leaves its results in attributes with a trailing underscore; `transform` or a query method applies them. The data are `synthetic` where a block makes them up.

## 1. `CUSUMDriftDetector`

Two-sided CUSUM of Page with a fixed reference, run on the whole series.

~~~python
import numpy as np
from dense_armor.ml.drift import CUSUMDriftDetector
x = [0.0, 1.0, -1.0, 0.0, 1.0, -1.0] * 5 + [6.0] * 5
m = CUSUMDriftDetector(radius=3, ref_mult=2, k=0.5, h=5.0).fit(x)
print(np.flatnonzero(m.flagged_).tolist())
~~~

~~~
[31, 33]
~~~

## Details

### References

Page, E. S. (1954). Continuous inspection schemes, Biometrika 41(1-2):
        100-115: the CUSUM with a fixed reference decided once from an
        in-control baseline (not in the folder, bibliographic only).
    Hawkins, D. M., Olwell, D. H. (1998). Cumulative Sum Charts and Charting
        for Quality Improvement, Springer: the slack `k` as half the shift to
        detect (not in the folder, bibliographic only).
    The recursions below are the ones written in the docstring of
    `cusum_detector` of the library.

### Choices stated explicitly

The class of a sketch is the exact value the sketch approximates, and the class of a statistic is the textbook formula on the whole array: both are `our choice`, because the references name no other batch algorithm.

### Development diary

The classes are new and independent of the online classes of dense_armor.utility.

Classes without a form:
- `ADWIN`: the reference defines the test on a stream, with a window or a running statistic that moves at each new value: the algorithm of the reference is written for one value at a time
- `KSWIN`: the reference defines the test on a stream, with a window or a running statistic that moves at each new value: the algorithm of the reference is written for one value at a time
- `PageHinkley`: the reference defines the test on a stream, with a window or a running statistic that moves at each new value: the algorithm of the reference is written for one value at a time
- `BayesianOnlineChangepoint`: Algorithm 1 of Adams and MacKay (2007) is the recursion on the run length that receives one value at a time; the cited pages give the recursion and the run-length posterior of the stream
