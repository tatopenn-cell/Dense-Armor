"""Item 0 of T4: the float32 error of each class against float64.

Stream: N(1e3, 0.01), 100,000 samples, seed 0, cast to float32. The
reference is the float64 variance of the same float32 values, never the
variance of the float64 values they came from. Three update rules are
compared: the plain Welford of the repository today, a compensated
Welford (TwoSum on the mean and on the M2 term), and the shift plus
compensated sums of the T3 kernel applied online.

Threshold 1e-6: a class above it keeps float64 accumulators.
"""

import sys

import numpy as np

sys.path.insert(0, ".")

N = 100_000
SEED = 0
THRESH = 1e-6


def two_sum(a, b):
    s = np.float32(a + b)
    bb = np.float32(s - a)
    e = np.float32((a - np.float32(s - bb)) + (b - bb))
    return s, e


def welford_plain(x32):
    n = np.float32(0.0)
    mean = np.float32(0.0)
    m2 = np.float32(0.0)
    for v in x32:
        v = np.float32(v)
        n_new = np.float32(n + np.float32(1.0))
        delta = np.float32(v - mean)
        mean = np.float32(mean + np.float32(delta / n_new))
        m2 = np.float32(m2 + np.float32(delta * np.float32(v - mean)))
        n = n_new
    return np.float32(m2 / n)


def welford_comp(x32):
    n = np.float32(0.0)
    mean = np.float32(0.0)
    c_mean = np.float32(0.0)
    m2 = np.float32(0.0)
    c_m2 = np.float32(0.0)
    for v in x32:
        v = np.float32(v)
        n_new = np.float32(n + np.float32(1.0))
        delta = np.float32(v - mean)
        d_over_n = np.float32(delta / n_new)
        mean_new, e_mean = two_sum(mean, d_over_n)
        c_mean = np.float32(c_mean + e_mean)
        mean = mean_new
        term = np.float32(delta * np.float32(v - mean))
        m2_new, e_m2 = two_sum(m2, term)
        c_m2 = np.float32(c_m2 + e_m2)
        m2 = m2_new
        n = n_new
    return np.float32(np.float32(m2 + c_m2) / n)


def shift_comp(x32):
    K = np.float32(x32[0])
    n = np.float32(0.0)
    s1 = np.float32(0.0); c1 = np.float32(0.0)
    s2 = np.float32(0.0); c2 = np.float32(0.0)
    for v in x32:
        y = np.float32(np.float32(v) - K)
        s1, e1 = two_sum(s1, y); c1 = np.float32(c1 + e1)
        s2, e2 = two_sum(s2, np.float32(y * y)); c2 = np.float32(c2 + e2)
        n = np.float32(n + np.float32(1.0))
    s1c = np.float32(s1 + c1); s2c = np.float32(s2 + c2)
    my = np.float32(s1c / n)
    return np.float32(np.float32(s2c / n) - np.float32(my * my))


def rel(a, b):
    a = float(a); b = float(b)
    if b == 0.0:
        return 0.0 if a == 0.0 else float("inf")
    return abs(a - b) / abs(b)


def main():
    rng = np.random.default_rng(SEED)
    x32 = (1e3 + 0.01 * rng.standard_normal(N)).astype(np.float32)
    x64 = x32.astype(np.float64)
    ref = float(x64.var())

    print("stream N(1e3, 0.01), N=%d, seed=%d" % (N, SEED))
    print("reference (float64 of the float32 values): %.6e" % ref)
    print()
    print("%-16s %14s %12s  decision" % ("update rule", "var32", "rel err"))
    for name, fn in [("welford_plain", welford_plain),
                     ("welford_comp", welford_comp),
                     ("shift_comp", shift_comp)]:
        v = fn(x32)
        e = rel(v, ref)
        print("%-16s %14.6e %12.3e  %s" %
              (name, float(v), e, "KEEP" if e <= THRESH else "float64"))


if __name__ == "__main__":
    main()
