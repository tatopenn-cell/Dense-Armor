"""The adapter that opens the primitives of the library to video."""

import numpy as np
import pytest

from dense_armor.core.audio.adapter import NeedsLabels
from dense_armor.core.online.drift.adwin import ADWIN
from dense_armor.core.online.learn.online_classifiers import OnlineGaussianNB
from dense_armor.core.signals.stats.sdft import SDFT
from dense_armor.core.signals.vision.features import FrameFeatures
from dense_armor.core.signals.vision.streams import ArrayStream, Frame
from dense_armor.core.signals.vision.video_input import VideoInput


def _frames(n=12):
    rng = np.random.default_rng(0)
    return [rng.random((12, 16)).astype(np.float32) for _ in range(n)]


class Scorer:
    def __init__(self):
        self.seen = []

    def score_one(self, x):
        self.seen.append(dict(x))
        return float(x["mean"])

    def learn_one(self, x):
        return self


class Learner:
    def __init__(self):
        self.n = 0

    def learn_one(self, x):
        self.n += 1


class Transformer:
    def learn_one(self, x):
        return self

    def transform_one(self, x):
        return {"mean": x["mean"]}


def test_a_primitive_with_score_one_gets_the_features_of_the_frame():
    p = Scorer()
    a = VideoInput(p)
    out = a.feed(_frames(1)[0])
    assert a.mode == "score" and a.n == 1 and out == a.last
    assert {"mean", "contrast", "centroid_x"} <= set(p.seen[0])


def test_a_frame_object_a_dict_and_an_array_give_the_same_features():
    img = _frames(1)[0]
    outs = []
    for item in (img, Frame(img, 0.0), {"frame": img, "t": 0.0}):
        p = Scorer()
        VideoInput(p).feed(item)
        outs.append(p.seen[0]["mean"])
    assert outs[0] == outs[1] == outs[2]


def test_a_classifier_predicts_and_learns_with_the_label():
    a = VideoInput(OnlineGaussianNB())
    for i, f in enumerate(_frames()):
        a.feed(f, i % 2)
    assert a.mode == "supervised" and a.n == 12


def test_a_classifier_without_a_label_says_so():
    a = VideoInput(OnlineGaussianNB())
    with pytest.raises(NeedsLabels):
        a.feed(_frames(1)[0])


def test_a_drift_detector_gets_the_mean_luminance():
    a = VideoInput(ADWIN())
    for f in _frames(20):
        a.feed(f)
    assert a.mode == "drift" and a.n == 20


def test_a_transformer_learns_and_transforms():
    a = VideoInput(Transformer())
    assert a.feed(_frames(1)[0])["mean"] > 0.0 and a.mode == "transform"


def test_a_primitive_that_only_learns_just_learns():
    p = Learner()
    a = VideoInput(p)
    a.feed(_frames(1)[0])
    assert a.mode == "learn" and p.n == 1


def test_a_primitive_of_samples_has_no_samples_in_a_frame():
    with pytest.raises(ValueError, match="video has none"):
        VideoInput(SDFT(4)).feed(_frames(1)[0])


def test_another_extractor_and_precomputed_features_can_be_given():
    ex = FrameFeatures(cells=(1, 1))
    p = Scorer()
    a = VideoInput(p, extractor=ex)
    a.feed(_frames(1)[0])
    assert a.extractor is ex and len(p.seen[0]) < 100
    a.feed(None, x={"mean": 0.5})
    assert p.seen[1] == {"mean": 0.5}


def test_a_stream_of_frames_runs_in_order():
    stream = ArrayStream(np.stack(_frames(8)), fps=10.0)
    a = VideoInput(Scorer())
    out = a.run(stream)
    assert len(out) == 8 and a.n == 8
