"""The scaling that AudioInput gives to the primitives that learn better with it."""

import numpy as np
import pytest

from dense_armor.core.audio import Audio, AudioInput
from dense_armor.core.audio.adapter import SCALING, _Standardizer
from dense_armor.core.online.anomaly.ocsvm import OneClassSGD
from dense_armor.core.online.forest.amf import AggregatedMondrianForestRegressor
from dense_armor.core.online.forest.oxt import OnlineExtraTreesRegressor
from dense_armor.core.online.learn.online_classifiers import OnlineGaussianNB
from dense_armor.core.online.rules.amrules import AMRulesRegressor
from dense_armor.core.signals.preprocessing.text_input import TextInput
from dense_armor.core.signals.vision.video_input import VideoInput


class Scorer:
    def __init__(self):
        self.seen = []

    def score_one(self, x):
        self.seen.append(dict(x))
        return 0.0

    def learn_one(self, x):
        return self


def _block(f, sr=8000, n=512, seed=0):
    t = np.arange(n) / sr
    rng = np.random.default_rng(seed)
    return Audio(np.sin(2 * np.pi * f * t) + 0.05 * rng.standard_normal(n), sr)


def test_the_names_of_the_table_are_classes_of_the_library():
    classes = [
        AMRulesRegressor,
        OneClassSGD,
        AggregatedMondrianForestRegressor,
        OnlineExtraTreesRegressor,
    ]
    assert {c.__name__ for c in classes} == set(SCALING)
    assert set(SCALING.values()) == {"std", "logstd"}


def test_the_standardizer_gives_zero_at_first_and_then_the_z_score_of_what_it_has_seen():
    s = _Standardizer(log=False)
    assert s({"a": 5.0, "b": 1.0}) == {"a": 0.0, "b": 0.0}
    seen = [5.0, 7.0, 12.0, 4.0]
    outs = [s({"a": v, "b": 1.0}) for v in seen[1:]]
    for n, out in enumerate(outs, start=2):
        v = np.array(seen[:n])
        assert out["a"] == pytest.approx((v[-1] - v.mean()) / v.std(ddof=1))
        assert out["b"] == 0.0


def test_the_log_standardizer_takes_log1p_first():
    s = _Standardizer(log=True)
    s({"a": 3.0})
    out = s({"a": 7.0})
    v = np.log1p([3.0, 7.0])
    assert out["a"] == pytest.approx((v[-1] - v.mean()) / v.std(ddof=1))


@pytest.mark.parametrize(
    ("primitive", "expected"),
    [
        (OneClassSGD(), "std"),
        (AMRulesRegressor(), "std"),
        (AggregatedMondrianForestRegressor(), "logstd"),
        (OnlineExtraTreesRegressor(), "logstd"),
        (OnlineGaussianNB(), None),
        (Scorer(), None),
    ],
)
def test_auto_takes_the_scaling_of_the_class_and_none_for_the_others(
    primitive, expected
):
    assert AudioInput(primitive).scaling == expected


def test_none_gives_the_magnitudes_as_they_are_and_a_name_forces_the_scaling():
    assert AudioInput(OneClassSGD(), scaling=None).scaling is None
    assert AudioInput(Scorer(), scaling="std").scaling == "std"
    assert AudioInput(Scorer(), scaling="logstd").scaling == "logstd"


def test_a_scaling_that_does_not_exist_is_refused():
    with pytest.raises(ValueError, match="scaling must be one of"):
        AudioInput(Scorer(), scaling="mel")


def test_the_primitive_receives_the_scaled_bands_and_the_raw_ones_without_scaling():
    raw, scaled = Scorer(), Scorer()
    a_raw, a_std = AudioInput(raw), AudioInput(scaled, scaling="std")
    for f in (300.0, 1200.0, 2500.0):
        a_raw.feed(_block(f))
        a_std.feed(_block(f))
    assert raw.seen[2]["b0"] > 0.0 and scaled.seen[0]["b0"] == 0.0
    assert scaled.seen[2]["b0"] != raw.seen[2]["b0"]
    assert max(abs(v) for v in scaled.seen[2].values()) < 10.0


def test_text_and_video_do_not_take_the_scaling_of_the_audio():
    assert TextInput(OneClassSGD()).scaling is None
    assert VideoInput(AMRulesRegressor()).scaling is None
