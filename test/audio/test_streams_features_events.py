import doctest
import importlib
import sys

import numpy as np
import pytest

from dense_armor.utility.online.anomaly.filters import (
    ChauvenetScorer,
    HampelScorer,
    TukeyScorer,
)
from dense_armor.utility.online.bayes.multinomial import MultinomialNB
from dense_armor.utility.online.cluster.kmeans import OnlineKMeans
from dense_armor.utility.online.drift.adwin import ADWIN
from dense_armor.utility.online.drift.detector import CUSUMDriftDetector
from dense_armor.utility.online.drift.kswin import KSWIN
from dense_armor.utility.online.drift.page_hinkley import PageHinkley
from dense_armor.utility.online.learn.online_classifiers import (
    OnlineGaussianNB,
    OnlineSoftmaxRegression,
)
from dense_armor.utility.online.learn.online_dynamics import RecursiveLeastSquares
from dense_armor.utility.online.linear.bayes import BayesianLinearRegression
from dense_armor.utility.online.neighbors.knn import KNNClassifier, KNNRegressor
from dense_armor.utility.online.timeseries.ar import OnlineAR
from dense_armor.utility.online.timeseries.garch import GARCH11
from dense_armor.utility.online.tree.fimt import FIMTDDRegressor
from dense_armor.utility.online.tree.hoeffding import HoeffdingTreeClassifier
from dense_armor.utility.signals.audio import (
    Audio,
    AudioFeatures,
    AudioFileStream,
    AudioLiveStream,
    detect_events,
    run_estimators,
    white_noise,
    write_wav,
)
from dense_armor.utility.signals.preprocessing.scale import (
    EWScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
from dense_armor.utility.signals.stats.dependence import (
    Autocorrelation,
    RunningCovariance,
)
from dense_armor.utility.signals.stats.mixture import OnlineGaussianMixture
from dense_armor.utility.signals.stats.moments import RunningMoments

SR = 8000


def _signal(seconds=3.0, tone_at=2.0, seed=1):
    t = np.arange(int(SR * seconds)) / SR
    return 0.1 * white_noise(len(t), seed) + np.where(
        t >= tone_at, np.sin(2 * np.pi * 440 * t), 0.0
    )


def _blocks(x, n=512):
    return [Audio(x[i : i + n], SR, i / SR) for i in range(0, len(x) - n + 1, n)]


def test_file_stream_block_count_and_time(tmp_path):
    x = _signal(1.0)
    p = tmp_path / "a.wav"
    write_wav(p, Audio(x, SR), "float32")
    blocks = list(AudioFileStream(p, 1000, 500))
    assert len(blocks) == 1 + (len(x) - 1000) // 500
    t = [b.t0 for b in blocks]
    assert t == sorted(t) and t[1] == pytest.approx(500 / SR)
    assert all(b.n == 1000 and b.sr == SR for b in blocks)
    with pytest.raises(ValueError):
        AudioFileStream(p, 0)


def _fake_backend(chunks):
    class Dev:
        def __init__(self, cb):
            self.cb = cb
            self.closed = False

        def start(self):
            for c in chunks:
                self.cb(c.reshape(-1, 1), len(c), None, None)

        def stop(self):
            pass

        def close(self):
            self.closed = True

    return lambda cb, sr, block, device: Dev(cb)


def test_live_stream_yields_known_signal():
    x = _signal(0.5)
    chunks = [x[i : i + 400] for i in range(0, 2000, 400)]
    s = AudioLiveStream(SR, 400, queue_size=8, backend=_fake_backend(chunks)).start()
    got = [s.read(0.1) for _ in range(5)]
    assert all(g is not None for g in got)
    assert np.allclose(np.concatenate([g.x for g in got]), x[:2000])
    assert [g.t0 for g in got] == pytest.approx([i * 400 / SR for i in range(5)])
    assert s.read(0.01) is None and s.dropped == 0
    s.stop()


def test_live_stream_counts_dropped_blocks():
    chunks = [np.zeros(100) for _ in range(10)]
    s = AudioLiveStream(SR, 100, queue_size=2, backend=_fake_backend(chunks)).start()
    assert s.dropped == 8


def test_live_stream_without_library_raises_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    with pytest.raises(ImportError):
        AudioLiveStream(SR, 100)


def test_features_keys_sine_band_and_errors():
    fe = AudioFeatures(8)
    t = np.arange(512) / SR
    f = fe.transform(Audio(np.sin(2 * np.pi * 440 * t), SR))
    assert list(f) == [f"b{i}" for i in range(8)]
    assert max(f, key=f.get) == "b0"
    st = Audio(np.stack([np.sin(2 * np.pi * 440 * t)] * 2, axis=1), SR)
    assert fe.transform(st) == pytest.approx(f)
    with pytest.raises(ValueError):
        AudioFeatures(0)
    with pytest.raises(ValueError):
        AudioFeatures(8).transform(Audio(np.ones(8), 8))
    assert len(AudioFeatures(5).transform(Audio(np.ones(8), 8))) == 5


def _estimators():
    return {
        "HampelScorer": HampelScorer(),
        "ChauvenetScorer": ChauvenetScorer(),
        "TukeyScorer": TukeyScorer(),
        "ADWIN": ADWIN(),
        "CUSUMDriftDetector": CUSUMDriftDetector(),
        "KSWIN": KSWIN(),
        "PageHinkley": PageHinkley(),
        "OnlineKMeans": OnlineKMeans(k=3),
        "OnlineGaussianNB": OnlineGaussianNB(),
        "OnlineSoftmaxRegression": OnlineSoftmaxRegression(),
        "RecursiveLeastSquares": RecursiveLeastSquares(),
        "BayesianLinearRegression": BayesianLinearRegression(),
        "KNNClassifier": KNNClassifier(),
        "KNNRegressor": KNNRegressor(),
        "HoeffdingTreeClassifier": HoeffdingTreeClassifier(),
        "FIMTDDRegressor": FIMTDDRegressor(),
        "MultinomialNB": MultinomialNB(),
        "OnlineAR": OnlineAR(),
        "GARCH11": GARCH11(),
        "EWScaler": EWScaler(),
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler(),
        "RunningMoments": RunningMoments(),
        "RunningCovariance": RunningCovariance(),
        "OnlineGaussianMixture": OnlineGaussianMixture(),
        "Autocorrelation": Autocorrelation(),
    }


def test_run_estimators_feeds_every_family_without_error():
    blocks = _blocks(_signal())
    rep = run_estimators(blocks, _estimators(), label=lambda b: int(b.t0 >= 2.0))
    errors = {k: r["error"] for k, r in rep.items() if r["error"]}
    assert errors == {}
    assert all(r["n"] == len(blocks) for r in rep.values())
    modes = {r["mode"] for r in rep.values()}
    assert modes == {"drift", "score", "supervised", "transform"}


def test_run_estimators_reports_drift_alarm_after_tone_and_none_before():
    blocks = _blocks(_signal())
    ests = {"cusum": CUSUMDriftDetector(), "adwin": ADWIN()}
    rep = run_estimators(blocks, ests)
    assert rep["cusum"]["alarms"] > 0 and rep["cusum"]["mode"] == "drift"
    first = _blocks(_signal(1.9, tone_at=99.0))
    quiet = run_estimators(first, {"cusum": CUSUMDriftDetector()})
    assert quiet["cusum"]["alarms"] == 0


def test_run_estimators_needs_labels_and_reports_errors():
    blocks = _blocks(_signal(0.5))
    rep = run_estimators(blocks, {"nb": OnlineGaussianNB()})
    assert rep["nb"]["error"] == "needs labels"

    class Broken:
        def learn_one(self, x):
            raise RuntimeError("boom")

    rep = run_estimators(blocks, {"b": Broken()})
    assert rep["b"]["error"].startswith("RuntimeError") and rep["b"]["n"] == 0


def test_detect_events_finds_tone_and_stays_quiet_on_noise(tmp_path):
    x = _signal()
    p = tmp_path / "a.wav"
    write_wav(p, Audio(x, SR), "float32")
    hits = detect_events(AudioFileStream(p, 512))
    assert hits and 1.9 <= hits[0][0] <= 2.1
    noise = detect_events(_blocks(_signal(tone_at=99.0)))
    assert noise == []


def test_detect_events_live_equals_file(tmp_path):
    x = _signal()
    n = 512
    chunks = [x[i : i + n] for i in range(0, len(x) - n + 1, n)]
    p = tmp_path / "a.wav"
    write_wav(p, Audio(x, SR), "float32")
    live = AudioLiveStream(
        SR, n, queue_size=len(chunks), backend=_fake_backend(chunks)
    ).start()
    got = [live.read(0.1) for _ in chunks]
    live_hits = detect_events(got)
    file_hits = detect_events(AudioFileStream(p, n))
    assert [h[0] for h in live_hits] == [h[0] for h in file_hits]
    assert np.allclose([h[1] for h in live_hits], [h[1] for h in file_hits], rtol=1e-4)


@pytest.mark.parametrize("name", ["streams", "features", "events"])
def test_doctests(name):
    mod = importlib.import_module(f"dense_armor.utility.signals.audio.{name}")
    assert doctest.testmod(mod, extraglobs={"np": np, "Audio": Audio}).failed == 0
