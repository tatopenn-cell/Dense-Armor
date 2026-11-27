# Results: CUSUM against two classic drift detectors

[← CUSUM](index.md)

20 seeded streams of 1,000 N(0, 1) samples per row, mean shift at sample 500. Each cell is
recall (fraction of streams where the shift was found) / delay in samples / false alarms per
stream:

| shift (σ) | CUSUM adaptive | CUSUM fixed | Page–Hinkley | ADWIN |
|---|---|---|---|---|
| 0.5 | 0.00 / — / 0.00 | 0.55 / 96.0 / 0.10 | 1.00 / 81.0 / 0.05 | 1.00 / 127.8 / 0.00 |
| 1.0 | 0.00 / — / 0.00 | 0.90 / 42.3 / 0.50 | 1.00 / 41.2 / 0.10 | 1.00 / 54.2 / 0.00 |
| 1.5 | 0.15 / 15.0 / 0.05 | 1.00 / 20.9 / 0.40 | 1.00 / 42.9 / 0.10 | 1.00 / 43.0 / 0.00 |
| 2.0 | 0.05 / 6.0 / 0.00 | 1.00 / 12.6 / 0.35 | 1.00 / 14.9 / 0.05 | 1.00 / 41.4 / 0.00 |
| 3.0 | 0.85 / 9.8 / 0.00 | 1.00 / 7.8 / 0.05 | 1.00 / 11.2 / 0.05 | 1.00 / 11.0 / 0.00 |

"fixed" finds a sustained shift fastest from 1.5σ upward, at the price of more false alarms
than Page–Hinkley and ADWIN. "adaptive" is built for something else: its sliding reference
catches up with a new level, so it reacts to the leading edge of a drift and misses most small
step changes. ADWIN never raised a false alarm here but is the slowest on small shifts.
