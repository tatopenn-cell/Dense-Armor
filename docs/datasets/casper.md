# Casper: the reader of the CASPER recording

`Casper(path).stream(columns, start_h, stop_h)` reads the recording of an industrial robot arm from a local copy of the CASPER dataset. The library does not ship the data, so this page has no example that runs without it. The overview of the section is on [Datasets](index.md).

## What it reads

- `Casper(path).stream(columns, start_h, stop_h)` reads `right_arm.csv` from a local copy of the
  CASPER dataset (Kayan et al. 2023, section III-A: a UR3e arm logged at about 20 Hz, a 24-hour
  test with about half of the samples anomalous); hours are counted from the first timestamp.
  The code repository of the dataset is MIT (https://github.com/hkayann/CASPER-PerCom), the data
  are CC BY-SA 4.0
  (https://www.kaggle.com/datasets/hkayan/industrial-robotic-arm-anomaly-detection); the library
  reads the user's copy and redistributes nothing.

## API reference

::: dense_armor.core.signals.datasets.casper

---

## Details

- Source: Kayan, H., Rana, O., Burnap, P.,
  Perera, C. (2023), "CASPER: Context-aware anomaly detection system for industrial robotic
  arms".
