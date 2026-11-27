# Drift detectors: sequential statistical architectures for concept drift isolation

## The map of the drift detectors

Click a page to open it, or the item on the left (or the button at the top) to go back to the [map of the library](../../index.md#the-library-folder-by-folder).

<object type="image/svg+xml" data="../../assets/maps/drift_detectors.svg" style="width:100%"></object>

This section explains how a system can notice slow, gradual changes in its operational numbers before an outright failure occurs. By accumulating small deviations over time rather than looking at individual readings, these four tools can pinpoint when a shift has happened, each trading off reaction delay against the rate of false alarms.

The cost of executing orders can creep up without any single order looking wrong: the mix of orders changes, a venue gets slower. The model still works, but the numbers it reads are no longer the numbers it learned at the start. That is **drift**, and a system that never notices it slowly loses accuracy.

The problem is as old as statistical quality control. Shewhart's control chart (1931) watched a single sample against a fixed band and shouted as soon as one of them fell outside it. It worked for large, sudden faults, but it was blind to slow changes, because a small shift in the mean never makes a single sample look unusual. Page (1954) proposed the answer that is still the basis of every detector in this section: instead of judging each sample alone, **accumulate** the small deviations over time, so that a shift too small to notice in one reading still grows into a signal after enough of them.

A drift detector reads one number per sample and answers a single question: *has the behaviour changed?* It must say so quickly, but it must not cry wolf: a detector that fires on every small fluctuation is as useless as one that stays silent. The four detectors use the same interface: `det.update(x)` for one sample, and the flag `det.drift_detected` that becomes `True` on the sample where the detector thinks something changed. Each one trades delay for false alarms in a different way.

## The pages of the section

| Page | What it shows |
|---|---|
| [CUSUM](../cusum/index.md) | the cumulative-sum detector, `CUSUMDriftDetector`: the same detector, explained in its own section with the average run length |
| [PageHinkley](page_hinkley.md) | the same idea with a running mean and standard deviation: the units do not matter |
| [ADWIN](adwin.md) | a window that cuts itself when two parts of it have different averages, and where it differs from the paper |
| [KSWIN](kswin.md) | compares the shapes of the newest values and of older ones, one alarm per change |
| [The four side by side](compare.md) | the four detectors on a robot stream, with the delay and the false alarms of each |

## The same on a robot

A robot that has been working fine for hours can start to behave differently without failing outright: a joint gets warmer and its torque creeps up, a new payload changes the load, a screw loosens and the vibration grows. The controller still works, but the numbers it reads are no longer the numbers it learned at the start. A robot that never notices the drift will slowly lose accuracy. The [last page](compare.md) runs the four detectors on a real robot arm.

## Details

- Sources:
  - Shewhart, W. A. (1931). *Economic Control of Quality of Manufactured Product*. Van Nostrand. The control chart.
  - Page, E. S. (1954). Continuous inspection schemes. *Biometrika* 41(1-2), 100-115. The CUSUM.
  - Hinkley, D. V. (1971). Inference about the change-point from cumulative sum tests. *Biometrika* 58(3), 509-523. The Page-Hinkley form.
  - Lorden, G. (1971). Procedures for reacting to a change in distribution. *Annals of Mathematical Statistics* 42(6), 1897-1908. Asymptotic optimality of CUSUM.
  - Moustakides, G. V. (1986). Optimal stopping times for detecting changes in distributions. *Annals of Statistics* 14(4), 1379-1387. Exact optimality of CUSUM.
  - Bifet, A., Gavaldà, R. (2007). Learning from time-changing data with adaptive windowing. In *Proceedings of the 2007 SIAM International Conference on Data Mining*. ADWIN.
  - Lu, J., Liu, A., Dong, F., Gu, F., Gama, J., Zhang, G. (2018). Learning under concept drift: a review. arXiv:2004.05785. ADWIN and Page-Hinkley described in one place.
  - Raab, C., Heusinger, M., Schleif, F.-M. (2020). Reactive soft prototype computing for concept drift streams. *Neurocomputing* 416, 340-351, arXiv:2007.05432. KSWIN.
  - Xie, L., Zou, S., Xie, Y., Veeravalli, V. V. (2021). Sequential (quickest) change detection: classical results and new directions. arXiv:2104.04186. The general CUSUM/Page/Shewhart framework.
  - Kayan, H., et al. (2023). CASPER: a real-world dataset for anomaly detection in a robot arm. In *IEEE PerCom Workshops*. The robot benchmark.
