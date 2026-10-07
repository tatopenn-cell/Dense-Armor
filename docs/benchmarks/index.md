# Benchmarks

Every module in this library has unit tests. Those tests verify the math, not the
behaviour on real signals. Real signals are messy: noise is not Gaussian, drift is not
a clean step, the frequencies are not the ones the synthetic benchmark chose. This
section is where the library is run, unchanged, on recordings from real hardware and
the numbers are reported as they come out.

The rule for every page here: **every number is printed by a script in the repository**.
No number is estimated, none is copied from a paper, none is rounded to look better.
If a result is negative, it is reported as negative.

## Pages

- **[Real robot arm (CASPER, UR3e)](real_robot_benchmark.md)** — the anomaly stack on
  24.5 h of a real industrial arm. The negative result and the reason for it.
