# Start here: a drone and a board

You have a drone, a flight controller that speaks MAVLink, and a small computer (a Raspberry Pi or an NVIDIA Jetson). You want to run one of the pipelines of this section on the board, read the alarm over the network, and use it. This page is the path from a fresh operating system to a score on an MQTT topic.

What to buy is **not** stated here. The page assumes the drone, the flight controller, the board and the cable are already chosen and at hand. The steps are the same on a Raspberry Pi and on a Jetson, and the two differences (the base image, the Python install) are called out where they appear.

## The scene

A quadrotor on a desk, propellers off, battery disconnected. The flight controller is connected to the board over a serial cable (TELEM2 on a Pixhawk, UART on a Raspberry Pi or a Jetson). The board runs the operating system of the vendor (Raspberry Pi OS 64-bit, or JetPack 6 on a Jetson). The network is a Wi-Fi hotspot, and the laptop with the MQTT client is on the same network.

The goal is to see, on the laptop, one line per sample with the score of the pipeline, and a line on the alarm topic when the pipeline raises an alarm. No propellers, no flight: the stream is synthetic (the pipelines of this section run on `synthetic_flight`, `synthetic_mqtt`, and the simulated lidar). When the same pipeline runs on a flight controller, the fields of the stream are the ones the flight controller sends, and the rest of the path is the same.

**The commands on this page were run on an x86 host. The two boards were not available, so the same commands have not been run on a Raspberry Pi or a Jetson. The page says so where it matters.**

## 1. What the board runs

The pipelines of the drones section are Python functions. Each one takes a stream and returns a result with a per-sample score and a per-sample alarm:

| Pipeline | Question |
|---|---|
| `actuator_fault` | Which motor broke, and how many seconds after? |
| `sensor_consistency` | Is the GPS telling the truth? |
| `propeller_health` | Is a propeller damaged? |
| `battery_endurance` | How many minutes are left? |
| `safety_filter` | Is the next move safe? |
| `link_health` | Is the radio link about to fail? |

The edge benchmark runs all six on synthetic data and prints the time per sample, on the board itself. That is the number you want before choosing a pipeline for the flight.

## 2. Install

On a fresh Raspberry Pi OS 64-bit or a fresh JetPack 6, clone the repository and run the install script. The script creates a virtual environment, installs the drone extra, and prints the Python version and the module paths.

```bash
git clone https://github.com/tatopenn-cell/Dense-Armor.git
cd Dense-Armor
bash edge/install.sh
```

The output ends with three lines: the path of the Python interpreter inside the virtual environment, the version, and the file of `dense_armor` and `pipelines`. If the first line is a path under the repository, the install worked. `bash -n edge/install.sh` checks the script without running it.

**Not run on the board.** The commands above were checked with `bash -n` on x86; they were not run on a Raspberry Pi or a Jetson, because the boards were not available. The two Dockerfile variants (`edge/Dockerfile.arm64` for Raspberry Pi, `edge/Dockerfile.jetson` for Jetson) are also written and not built.

## 3. The benchmark, on the board

Before running a pipeline, measure the time per sample on this board. The script runs the six pipelines on synthetic data, with a fixed number of samples per run, and prints the median, the interquartile range and the 99th percentile of the time per sample. The output is also written to a JSON file named after the host, so the table from a Raspberry Pi and the table from a Jetson are comparable.

```bash
source .venv/bin/activate
python -m edge.edge_bench --smoke
```

The `--smoke` flag uses 200 samples per run and 5 runs, and finishes in under a minute. Without the flag it uses 2000 samples and 20 runs. The JSON is in `benchmarks/drones/edge_bench_<hostname>.json`.

**Run on x86, not on the board.** The output below is from an x86 host (Colab, 2 CPUs). The page does not claim that a Raspberry Pi or a Jetson produces the same numbers; the whole point of the file is to run it on the board and get the board's numbers.

```text
host: 5da763c0fe9f, 2 cpus, temp=None
pipeline                  median     iqr     p99    dRSS KB
------------------------------------------------------------
actuator_fault             0.047   0.018   0.068        0.0
sensor_consistency         0.011   0.001   0.012        0.0
propeller_health           0.024   0.000   0.035        0.0
battery_endurance          0.098   0.004   0.152        0.0
safety_filter              0.024   0.000   0.024        0.0
link_health                0.006   0.000   0.006        0.0
```

## 4. The pipeline on MQTT

The bridge runs a pipeline and publishes its score and its alarm on MQTT, on the topics `drone/<id>/score` and `drone/<id>/alarm`. The id is a short string chosen by the user; by default a random four-letter suffix, so two boards on the same broker do not mix. On a board connected to the laptop over Wi-Fi, the broker can be the laptop itself (a local Mosquitto), or a public one.

```bash
python edge/mqtt_bridge.py --smoke --id d1
```

The output is the broker, the id, the two topics, the number of score messages and the number of alarm messages. With `--smoke` the stream is 200 samples; without, 2000. The pipeline is `sensor_consistency`, whose residual is defined and whose alarm is the OR of a CUSUM on the inter-arrival.

## 5. Read the score on the laptop

On the laptop, on the same network, install a MQTT client and subscribe to the two topics. The commands below are the ones a user runs; they were run against `test.mosquitto.org`, with the id `d1`, and the score messages arrived on the laptop side (the same machine).

```bash
# on the laptop, in a terminal
mosquitto_sub -h test.mosquitto.org -t 'drone/d1/score' -t 'drone/d1/alarm'
```

Each line is one message: a number (the score) on the score topic, a `1` on the alarm topic when the pipeline raises an alarm. The alarm is not repeated: the bridge publishes one message per rising edge of the alarm, not one per sample. So a long series of numbers with no `1` means the pipeline is running and has not flagged anything.

On the board, the same message arrives from the bridge. The bridge is the process that runs on the board; the subscription is on the laptop. The board does not need a MQTT client for this; it only needs the broker to be reachable.

## 6. The container, if the operating system is not enough

For a board where the Python of the operating system is not the version the library needs, there are two container definitions in `edge/`. The Raspberry Pi one uses the multi-arch `python:3.11-slim` image, the Jetson one uses the NVIDIA L4T PyTorch image for JetPack 6. Both are written and not built.

```bash
# on a Raspberry Pi 4 or 5, 64-bit OS, with Docker installed
docker buildx build --platform linux/arm64 -f edge/Dockerfile.arm64 -t dense-armor-drone .
docker run --rm -it dense-armor-drone python -m edge.edge_bench --smoke
```

**Not built.** The Dockerfiles were not built: there is no arm64 host and no Jetson. The build command is the one a user runs on the board. If the board's Docker does not have buildx, `docker build -f edge/Dockerfile.arm64 -t dense-armor-drone .` is the fallback.

## 7. What was run, and what was not

| Command | Run on x86 | Run on the board |
|---|---|---|
| `bash -n edge/install.sh` | yes | no |
| `python -m edge.edge_bench --smoke` | yes, 0.5 s | no |
| `python edge/mqtt_bridge.py --smoke --id d1` | yes, 500/500 score, 2/2 alarm | no |
| `docker buildx build ... edge/Dockerfile.arm64` | not built (no arm64 host) | no |
| `docker build ... edge/Dockerfile.jetson` | not built (no Jetson) | no |
| install on a board | no | no |

The last three rows are not run, and the page says so. The numbers in the previous sections are from x86, not from a board. The point of the file is that the same commands, run on the board, produce the board's numbers; the file is the file, not the board.
