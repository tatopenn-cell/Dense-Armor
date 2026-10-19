# Jetson power: how much does the power mode change the time per sample?

A Jetson module runs at different power levels. The same code, on the same board, takes a different time per sample in each mode. This page states the modes of each Orin module, the commands to read and set the mode, and how to measure `edge.edge_bench --smoke` in each one. The measurements table is empty: it is filled with the numbers of a board, not from a desktop.

## 1. The modes of each module

The table is the one of the NVIDIA developer guide, [Platform Power and Performance](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html). The number in parentheses is the id used by `nvpmodel -m`.

| Module | Modes (default marked) |
|---|---|
| Orin Nano 4 GB | 10W (default, 0), 7W_AI (1), 7W_CPU (2) |
| Orin Nano 8 GB | 15W (default, 0), 7W (1), MaxN (2) |
| Orin NX 8 GB | MaxN (0), 10W (1), 15W (default, 2), 20W (3) |
| Orin NX 16 GB | MaxN (0), 10W (1), 15W (default, 2), 25W (3) |
| AGX Orin 32 GB | MaxN (0), 15W (1), 30W (default, 2), 40W (3) |
| AGX Orin 64 GB | MaxN (0), 15W (1), 30W or 35W (default, 2), 50W or 60W (3) |

The memory and the TOPS of the modules are on [the Jetson modules page](https://developer.nvidia.com/embedded/jetson-modules).

## 2. Read and set the mode

```bash
sudo /usr/sbin/nvpmodel -q
sudo /usr/sbin/nvpmodel -m 0
```

The first command prints the current mode; the second sets the mode with the id in the table. The ids are per module: id 0 is the default on Orin Nano 4 GB and 8 GB, MaxN on Orin NX and AGX Orin. `sudo /usr/sbin/nvpmodel -q --verbose` prints the mode and the clocks.

## 3. Watch the board while the benchmark runs

```bash
sudo tegrastats --interval 1000
```

`tegrastats` prints one line per second with the power, the temperatures and the clocks of the module. Run it in a second terminal while the benchmark runs: the time per sample and the power of the mode are read together.

## 4. Measure the benchmark in each mode

```bash
python -m edge.edge_bench --smoke
```

`edge.edge_bench --smoke` runs every pipeline on a small synthetic stream and prints, for each one, the median, the interquartile range and the p99 of the time per sample. Run it once per mode:

```bash
for m in 0 1 2; do sudo /usr/sbin/nvpmodel -m $m; python -m edge.edge_bench --smoke; done
```

The loop sets each mode and runs the smoke benchmark. The id must be one the module has: on Orin Nano 4 GB the ids are 0, 1, 2; on AGX Orin 64 GB they are 0, 1, 2, 3.

## 5. The measurements

| Module | Mode | Pipeline | Median ms/sample | IQR ms/sample | p99 ms/sample |
|---|---|---|---|---|---|
| to be filled with the numbers of a board | | | | | |

The table is empty. A number written here without a board is a number invented, and this page does not invent numbers. The script that fills it is `edge/board_selftest.sh`, which runs the smoke benchmark, reads the model from `/proc/device-tree/model`, and writes the JSON with the model in its file name.

## Details

### References

- NVIDIA, *Jetson Orin Nano, Orin NX and AGX Orin Series: Platform Power and Performance*. URL in section 1. The table of the modes is copied from it.
- NVIDIA, *Jetson Modules*. URL in section 1. The memory and the TOPS of each module.
- `edge/edge_bench.py` in this repository: the benchmark the page runs.

### Choices stated explicitly

- The page states the commands and the table, and leaves the measurements empty on purpose: the task asks for the numbers of a board, and no board is available here.
- The loop in section 4 sets the modes in order and runs the smoke benchmark for each. It does not restore the previous mode at the end: the last mode stays. A script that restores it is `edge/board_selftest.sh` for the single run, not for the loop.
- The mode ids are per module, and the table is the source. A module whose id is not in the table is not covered.

### Development diary

The first version of the page had a table with three rows filled from a desktop run of `edge.edge_bench --smoke`. The numbers were the desktop's, not a Jetson's, and the page does not compare a desktop to a board. The table is now empty, and the script that fills it is `edge/board_selftest.sh`.
