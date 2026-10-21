# Jetson energy: does the online pipeline use less energy than the batch one on my board?

The question is the title. A drone flies on a battery, and a joule the pipeline spends is a joule the motors do not have. On a Jetson the two models of [Static vs online](static_vs_online.md) run on the same stream; this page measures the energy of each from a power trace.

The measurement has three pieces: the energy of a trace (section 1), the benchmark that runs the two models with a power source and pairs the runs by seed (section 2), and one command that runs the benchmark on a board (section 3). The results table is the run of this session with a **synthetic** power source, a constant 5000 mW. A second table is empty on purpose, to be filled with the numbers of a board.

## 1. The energy of a power trace

A power trace is two arrays of the same length: the sample times in seconds and the power in milliwatt. The energy is the integral of the power over time by the trapezoid rule; a milliwatt-second is one thousandth of a joule. The **net** energy subtracts the idle power at every sample, and the total never goes below zero.

~~~python
import numpy as np
from edge.power import PowerTrace, energy_j, net_energy_j

t = np.array([0.0, 1.0, 2.0, 3.0])
mw = np.array([5000.0, 5000.0, 5000.0, 5000.0])
tr = PowerTrace(t_s=t, mw=mw)
e = energy_j(tr)
idle = 3000.0
en = net_energy_j(tr, idle_mw=idle)
print(f"energy_j: {e:.3f} J")
print(f"net_energy_j: {en:.3f} J")
~~~

~~~
energy_j: 15.000 J
net_energy_j: 6.000 J
~~~

A constant 5000 mW held for 3 s is 15 J. With an idle power of 3000 mW subtracted at every sample, the part above idle is 2000 mW held for 3 s, 6 J.

The sensors of a Jetson read the whole-board power from the INA3221 through sysfs nodes, one integer in milliwatt per read (2306.13107, page 2). The sensor underestimates the power by up to 50% in most cases, and the linear calibration of Table I, page 4 of the same paper, brings the error within 3%. The built-in sensors measure average values and miss the current peaks, and the sampling rate changes the precision (page 1). The calibration is one call:

~~~python
from edge.power import CALIBRATION_MW, calibrate_mw

for dev in sorted(CALIBRATION_MW):
    print(dev, f"{calibrate_mw(1000.0, dev):.2f} mW")
~~~

~~~
AGX Orin 4135.39 mW
Nano 1342.60 mW
TX2 2898.80 mW
Xavier NX 4230.41 mW
~~~

The four lines are the ones of Table I, page 4: AGX Orin `1.02x + 3115.39`, Xavier NX `1.10x + 3130.41`, TX2 `0.90x + 1998.80`, Nano `1.11x + 232.60`, with errors of 3%, 2%, 3%, 0.8%. The default of every function that could apply them is to leave the reading as it is: the owner decides whether the calibration holds on a different setup.

## 2. The benchmark: online against batch, paired by seed

For every seed the benchmark builds the same stream of [Static vs online](static_vs_online.md), splits it 50/50, fits the batch CART tree on the training half, and lets the online Hoeffding tree predict and learn on the test half. A power source is started and stopped around each model; the wall time, the CPU time and the net energy are written per run, and the runs of the two models are paired by seed. The idle power is measured for `idle_s` seconds before the runs and subtracted.

~~~bash
python -m edge.energy_bench --smoke
~~~

~~~
  seed  0: batch j/1000 = 0.144, online j/1000 = 0.105
  seed  1: batch j/1000 = 0.060, online j/1000 = 0.100
  seed  2: batch j/1000 = 0.056, online j/1000 = 0.102
json: benchmarks/drones/energy_bench_4ce6e14ced10.json

host: 4ce6e14ced10
board: not a Jetson
power: synthetic
idle_mw: 0.0 (not measured (idle_s = 0))
n_seeds: 3, n_samples: 600
~~~

The `--smoke` run uses 3 seeds, 600 samples per stream, and a constant 5000 mW, and finishes in about two seconds on x86. Without `--smoke` it uses 10 seeds and 2000 samples; with `--power-node PATH` it reads a sysfs node; with neither flag the energy fields are `null` and the wall and CPU time are still written. The JSON goes to `benchmarks/drones/energy_bench_<host>.json`, and the field `power` is `synthetic`, `sysfs` or `null`.

### The results

| Model | Wall time median (s) | CPU time median (s) | Net energy median (J) | J per 1000 samples, median | J per 1000 samples, IQR |
|---|---|---|---|---|---|
| BatchDecisionTreeClassifier (CART) | 0.0037 | 0.0037 | 0.0183 | 0.060 | 0.0439 |
| HoeffdingTreeClassifier | 0.0062 | 0.0062 | 0.0298 | 0.102 | 0.0025 |

**Synthetic**: constant 5000 mW, `dt_s = 0.1`, 3 seeds, 600 samples per stream, host `4ce6e14ced10`, `board: null`. The batch model used 0.060 J per 1000 samples and the online model 0.102 J per 1000 samples; the delta batch minus online is -0.040 J per 1000. Two seeds favour the batch model and one favours the online model. The paired Wilcoxon is **not computed**, fewer than six pairs, and A12 is 0.333 with a bootstrap 95% interval [0.0, 1.0]. This run does not say which model is cheaper on a board.

A board fills the second table. Run the command of section 3 once per power mode.

| Module | Mode | Model | Median ms/sample | Net energy median (J) | J per 1000 samples |
|---|---|---|---|---|---|
| to be filled with the numbers of a board | | | | | |

The table is empty on purpose: a number written here without a board is an invented number, and this page does not invent numbers. The script that fills it is `edge/board_selftest.sh`, which reads the model from `/proc/device-tree/model` and writes the JSON with the model in its file name.

## 3. One command on the board

`edge/board_selftest.sh` runs the smoke benchmark, reads the model, the OS and the Python version, and with the environment variable `POWER_NODE` set and readable it also runs the energy benchmark and prints the path of the JSON.

~~~bash
POWER_NODE=/sys/devices/platform/.../power1_input bash edge/board_selftest.sh
~~~

~~~
energy: /sys/devices/platform/.../power1_input
energy json: benchmarks/drones/energy_bench_<board>.json
~~~

Without `POWER_NODE` the same script prints one line that says the energy was not measured and gives the reason, and every other line of its output is unchanged.

## Details

### References

- Shalavi, N., Khoshsirat, A., Stellini, M., Zanella, A., Rossi, M. (2023). Accurate Calibration of Power Measurements from Internal Power Sensors on NVIDIA Jetson Devices. arXiv:2306.13107, Department of Information Engineering, University of Padova. Page 1: the sensors underestimate the power by up to 50% and the calibration brings the error within 3%; the built-in sensors measure average values and miss the current peaks; the sampling rate changes the precision. Page 2: the whole-board power is read from the INA3221 sensors through sysfs nodes. Page 3: the gap between internal and external readings is large and the envelope of the signal is preserved. Page 4, Table I: the four linear calibration lines in milliwatt, with the error of each.
- Shrestha, S. et al. (2024). Benchmarking Deep Learning Models on NVIDIA Jetson Nano for Real-Time Systems. arXiv:2406.17749. Page 2: models that demand fewer computational resources consume less power. The motivation of this page, not a measurement method.
- `edge/power.py`, `edge/energy_bench.py`, `edge/board_selftest.sh`, `pipelines/drones/static_vs_online.py`: the code this page runs.

### Choices stated explicitly

- The power source of the smoke run is a constant 5000 mW, `SyntheticPowerSource(mw=5000.0, dt_s=0.1)`. The wall time is the one measured between `start()` and `stop()`, and the energy is the constant power times that wall time. The label `synthetic` is in the code, in the JSON and in the table.
- `dt_s = 0.1` is my choice. The paper states that the sampling rate changes the precision (2306.13107, page 1) and gives no recommended interval, so the interval is a parameter without a default for the sysfs source, and the smoke run fixes it at 0.1 s.
- The idle power is measured for `idle_s` seconds before the runs and subtracted at every sample. In `--smoke` `idle_s` is zero and the idle is not measured; on a board the default is 30 s.
- The calibration of Table I is not applied by default. The page shows the four lines, and the owner decides whether the setup of the paper holds on a different board.
- The paired Wilcoxon is computed only when there are at least six pairs. With fewer, the JSON writes `null` and the reason in `wilcoxon_note`, instead of a number.
- `--smoke` and `--power-node` are mutually exclusive: a synthetic run labelled `synthetic` next to a real sensor would mix the two.
- The empty table is declared empty in the text next to it. A number in it without a board would be an invented number.

### Development diary

This delivery does not show a reduction of energy on a Jetson, because no board ran it. The only energy numbers on this page are labelled `synthetic` and come from a constant 5000 mW source on the x86 host of the session. A board with `POWER_NODE` set produces the same JSON with `"power": "sysfs"`, the model of the board in `board`, and the real trace; the table of section 2 is then filled with the numbers of that JSON, and the page states them as measured on that board.

The `energy_bench.py` script of this session had two bugs caught by the tests and fixed in the same branch: the idle measurement did a `start()` and `stop()` back to back and left one sample in the trace (fixed by sleeping for `idle_s`), and the print of the summary used `:10.3f` on `j_per_1000_median` which is `None` when there is no power source (fixed by printing the string `null`).
