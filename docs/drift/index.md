# Drift detection

Drift is a slow, sustained change: each sample looks normal, but together they move. A drift detector accumulates that evidence and raises one alarm when the change is real.

- **[CUSUM and ARL theory](cusum.md)** — the cumulative-sum detector, its streaming version, and how long it takes to fire (or to false-alarm) before you run it.
