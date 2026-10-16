# Speed

The per-sample cost of the main estimators and of the robot-side
components. Every number on this page comes from a single run of
`benchmarks/speed.py`; the script writes the JSON, the JSON is read
here. Full log and machine in `benchmarks/speed_log.md`.

## Machine

| field | value |
|---|---|
| cpu | Intel(R) Xeon(R) CPU @ 2.20GHz |
| cores | 2 |
| python | 3.13.16 |
| platform | Linux-6.6.122+-x86_64-with-glibc2.39 |
| numpy | 2.2.6 |
| jax | 0.11.1 |
| seeds | 20 |
| samples per seed | 2000 (first 200 discarded) |

## Estimators

| component | median µs | p99 µs | p99.9 µs | max µs |
|---|---|---|---|---|
| Hoeffding tree | 9.1 | 18.1 | 52.2 | 60.8 |
| Gaussian naive Bayes | 44.2 | 85.8 | 127.3 | 203.3 |
| Adaptive random forest (10 models) | 754.1 | 1374.0 | 1683.7 | 2032.3 |
| Half-Space Trees (25 trees) | 1285.2 | 2412.3 | 3234.9 | 4061.5 |
| ADWIN (`update`) | 9.5 | 16.2 | 30.3 | 59.7 |

## Robot side

| component | median µs | p99 µs | p99.9 µs | max µs | share of 1 kHz at the max |
|---|---|---|---|---|---|
| rate_limited_follower | 36.5 | 76.7 | 2116.9 | 9167.5 | 917% |
| cbf_safety_filter | 0.9 | 1.3 | 13.9 | 154.9 | 15.5% |
| kinematic_tracking_controller | 8.1 | 21.5 | 56.6 | 4184.0 | 418% |
| RigidBodyModel.mass_matrix | 66.3 | 2444.6 | 3913.4 | 4413.9 | 441% |
| RigidBodyModel.gravity_forces | 55.0 | 1320.1 | 3394.3 | 5192.6 | 519% |

Every component has a median inside the 1 kHz budget of a robot control
loop. The four that exceed it at the maximum are listed with the cause in
`benchmarks/speed_log.md`.

## How to reproduce

From the repository root:

    python benchmarks/speed.py

The script writes a JSON next to the working directory and prints the
per-component table. It reads only the library under `dense_armor/` and
the standard library; a reference implementation is imported only if the
environment variable `SPEED_REF` names the module to import.
