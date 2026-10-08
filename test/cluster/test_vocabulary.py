import numpy as np

from dense_armor.checks import check_estimator
from dense_armor.utility.vision.vocabulary import BagOfVisualWords


def _vertical_stripes(size=64, freq=0.7):
    x = np.arange(size)
    row = np.sin(x * freq)
    return np.tile(row, (size, 1)).astype(np.float32)


def _horizontal_stripes(size=64, freq=0.7):
    y = np.arange(size)
    col = np.sin(y * freq)
    return np.tile(col[:, None], (1, size)).astype(np.float32)


def test_vocabulary_two_textures_differ():
    vert = _vertical_stripes()
    horiz = _horizontal_stripes()
    bow = BagOfVisualWords(n_words=4, patch_size=8, n_bins=4, seed=0)
    for _ in range(30):
        bow.learn_one({"frame": vert})
        bow.learn_one({"frame": horiz})
    hv = np.array(bow.transform_one({"frame": vert})["hist"])
    hh = np.array(bow.transform_one({"frame": horiz})["hist"])
    dist = float(np.abs(hv - hh).sum())
    print(f"hist vertical = {hv.round(3).tolist()}")
    print(f"hist horizontal = {hh.round(3).tolist()}")
    print(f"L1 distance = {dist:.4f}")
    assert dist > 0.3, f"histograms too similar: L1 = {dist:.4f}"


def test_vocabulary_hist_normalised():
    img = np.zeros((16, 16), dtype=np.float32)
    img[:, 8:] = 1.0
    bow = BagOfVisualWords(n_words=3, patch_size=8, n_bins=4)
    for _ in range(20):
        bow.learn_one({"frame": img})
    hist = bow.transform_one({"frame": img})["hist"]
    assert len(hist) == 3
    assert abs(sum(hist) - 1.0) < 1e-6


def test_check_estimator_vocabulary():
    check_estimator(BagOfVisualWords(n_words=3, patch_size=8, n_bins=4))
