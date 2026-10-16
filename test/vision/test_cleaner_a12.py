"""Measure the cleaner against the raw series on 20 seeds (A12 = 1.000)."""

import numpy as np
from scipy.stats import wilcoxon

from dense_armor.utility.vision.landmarks import LandmarkCleaner


def test_cleaner_beats_raw_on_20_seeds():
    raw_errors, clean_errors = [], []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        n = 400
        t = np.arange(n) * 0.05
        true = 0.3 * np.sin(2 * np.pi * 0.2 * t)
        noisy = true + rng.normal(0, 0.15, n)
        spikes = rng.choice(n, 10, replace=False)
        noisy[spikes] += rng.normal(1.5, 0.5, 10)
        miss = rng.choice(n, 5, replace=False)
        c = LandmarkCleaner(window=7, n_sigmas=3.0, max_gap=5)
        clean = np.zeros(n)
        for i in range(n):
            if i in miss:
                out = c.transform_one(
                    {"points": np.array([[0.0, 0.0]]),
                     "confidence": np.array([0.0])})
            else:
                out = c.transform_one({"points": np.array([[noisy[i], 0.0]])})
            clean[i] = out["points"][0, 0]
            _ = c.learn_one({"points": out["points"]})
        raw_errors.append(float(np.mean(np.abs(noisy[10:] - true[10:]))))
        clean_errors.append(float(np.mean(np.abs(clean[10:] - true[10:]))))
    a = np.array(raw_errors); b = np.array(clean_errors)
    w = wilcoxon(a, b)
    gt = int((b < a).sum())
    d = a[:, None] - b[None, :]
    a12 = float((d > 0).mean() + 0.5 * (d == 0).mean())
    assert gt >= 15
    assert a12 > 0.7
    assert w.pvalue < 0.05
