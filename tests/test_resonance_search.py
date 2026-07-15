# -*- coding: utf-8 -*-
from dense_armor.utility.resonance_search import smoke_test, apply_fast_resonance
import numpy as np


def test_smoke_test_passa():
    assert smoke_test() is True


def test_apply_fast_resonance_shape():
    rng = np.random.default_rng(0)
    m = rng.standard_normal((5, 8)).astype(np.float32)
    q = rng.standard_normal(8).astype(np.float32)
    scores = apply_fast_resonance(m, q)
    assert scores.shape == (5,)
    assert not np.any(np.isnan(scores))
