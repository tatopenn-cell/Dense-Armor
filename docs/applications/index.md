# Applications

An application is a worked pipeline: it takes a stream, runs tools of the library on it, and says what the pipeline finds and how well. The pages of this tab are independent. Open the one that looks like your problem.

![The map of the applications: finance, drones, IoT, biomedical and ARC-AGI, with the number of pages and scripts of each](../assets/applications/map.png)

*The five applications. The page counts come from the menu of the site and the script counts from the folder `pipelines/`; the figure is drawn by a script, so it follows both.*

## 1. The five applications

| Application | What it is | Where to go |
|---|---|---|
| **Finance** | Nine analytical pipelines, and the parts of the pre-trade system around them, for risk, execution and fraud detection on market data. | [Financial pipelines](../finance/index.md) |
| **Drones** | One question about a drone per page: each page takes a synthetic stream, runs one pipeline on it and says what it predicts and how well. | [Drones](../drones/index.md), [Start here](../drones/start_here.md) |
| **IoT** | A machine sends five numbers every few seconds. The page watches that stream in three steps: isolate a broken reading, detect a change of regime, and flag the samples that look like a mechanical failure. | [Industrial telemetry guard](../iot/industrial_telemetry_guard.md) |
| **Biomedical** | A wearable ECG records 360 numbers per second. The page blocks the jumps of a loose electrode and scores every other sample. It is a technical demonstration, not a medical device. | [ECG lead guard](../biomedical/ecg_lead_guard.md) |
| **ARC-AGI** | A benchmark of tasks that are easy for a person and hard for a program: a few pairs of small grids, a rule, and a test input to solve. | [ARC-AGI](../arc/index.md), [Start here](../arc/start_here.md), [Limits](../arc/limits.md) |

## 2. How the pages are built

Each page takes data (real or synthetic), says which tools of the library it uses, and prints the numbers that the code gives. The tools come from the other tabs: [Online learning](../online/index.md), [Machine learning](../ml/index.md), [Roles](../roles/index.md) and the [Ontology](../ontology/index.md). The scripts that make the numbers are in the folder `pipelines/` of the repository.

## Details

- The page counts of the figure come from `mkdocs.yml` and the script counts from the folders of `pipelines/` (finance, drones, IoT and biomedical). ARC-AGI has no folder there: its code is in the library.
- The script `docs/assets/maps/make_area_maps.py` stops if an application is added to the menu and not to the figure.
