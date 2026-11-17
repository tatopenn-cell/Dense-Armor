"""Onset detection on tracks of clicks with known times."""

import numpy as np
import pytest

from dense_armor.core.audio.onset import onset_detect


def click_track(seed, sr, dur=6.0):
    r = np.random.default_rng(seed)
    x = 0.005 * r.standard_normal(int(dur * sr))
    times, t = [], r.uniform(0.4, 0.8)
    while t < dur - 0.5:
        times.append(t)
        i, m = int(t * sr), int(0.02 * sr)
        decay = np.exp(-np.arange(m) / (0.004 * sr))
        x[i : i + m] += r.choice([-1, 1]) * decay * r.standard_normal(m) * 0.8
        t += r.uniform(0.35, 0.9)
    return x, np.array(times)


def match(found, truth, tol=0.05):
    used, hits = set(), 0
    for t in truth:
        d = [
            (abs(f - t), j)
            for j, f in enumerate(found)
            if j not in used and abs(f - t) <= tol
        ]
        if d:
            used.add(min(d)[1])
            hits += 1
    return hits


def f1(found, truth, tol=0.05):
    hits = match(found, truth, tol)
    p = hits / len(found) if len(found) else 0.0
    r = hits / len(truth)
    return 2 * p * r / (p + r) if p + r else 0.0


@pytest.mark.parametrize(
    ("sr", "kw"), [(22050, {}), (8000, {"n_fft": 1024, "hop": 256})]
)
def test_the_clicks_are_found_in_number_in_time_and_with_a_high_f1(sr, kw):
    total_hits = total_found = total_truth = 0
    for seed in range(10):
        x, truth = click_track(seed, sr)
        found = onset_detect(x, sr, **kw)
        assert abs(len(found) - len(truth)) <= 1
        hits = match(found, truth, 0.05)
        assert hits >= len(truth) - 1
        total_hits += hits
        total_found += len(found)
        total_truth += len(truth)
    p, r = total_hits / total_found, total_hits / total_truth
    assert 2 * p * r / (p + r) > 0.9


def test_a_long_frame_finds_the_clicks_but_not_to_the_millisecond():
    for seed in range(3):
        x, truth = click_track(seed, 8000)
        found = onset_detect(x, 8000)
        assert len(found) == len(truth)
        assert np.abs(found - truth).max() < 0.08


def test_the_times_are_increasing_seconds_and_inside_the_signal():
    x, _ = click_track(0, 22050)
    t = onset_detect(x, 22050)
    assert (np.diff(t) >= 0.1).all()
    assert t.min() > 0.0 and t.max() < len(x) / 22050


def test_silence_and_a_signal_shorter_than_a_frame_have_no_onsets():
    assert onset_detect(np.zeros(8000), 8000).shape == (0,)
    assert onset_detect(np.zeros(100), 8000).shape == (0,)


def test_a_steady_tone_has_no_onsets_but_a_note_that_starts_has_one():
    t = np.arange(16000) / 8000.0
    tone = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    assert onset_detect(tone, 8000).shape == (0,)
    note = np.where(t > 1.0, tone, 0.0)
    found = onset_detect(note, 8000)
    assert len(found) == 1 and abs(found[0] - 1.0) < 0.2


def test_two_events_closer_than_a_frame_are_one_onset_unless_the_frame_is_short():
    x = np.zeros(24000)
    x[6000:6040] = np.hanning(40)
    x[7200:7240] = np.hanning(40)
    assert len(onset_detect(x, 8000)) == 1
    assert len(onset_detect(x, 8000, n_fft=512, hop=128)) == 2


def test_min_flux_zero_turns_the_test_off_and_a_negative_one_is_refused():
    t = np.arange(16000) / 8000.0
    tone = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    assert len(onset_detect(tone, 8000, min_flux=0.0)) > 0
    with pytest.raises(ValueError, match="min_flux"):
        onset_detect(tone, 8000, min_flux=-0.1)


def test_min_gap_removes_the_weaker_of_two_close_peaks():
    x = np.zeros(24000)
    x[6000:6040] = np.hanning(40)
    x[8400:8440] = 0.6 * np.hanning(40)
    assert len(onset_detect(x, 8000, min_gap=0.1)) == 2
    assert len(onset_detect(x, 8000, min_gap=1.0)) == 1


def test_a_higher_delta_keeps_only_the_strong_peaks():
    x = np.zeros(24000)
    x[6000:6040] = np.hanning(40)
    x[14000:14040] = 0.15 * np.hanning(40)
    assert len(onset_detect(x, 8000, delta=0.02)) == 2
    assert len(onset_detect(x, 8000, delta=0.5)) == 1


@pytest.mark.parametrize("kw", [{"min_gap": 0.0}, {"context": -1.0}])
def test_the_parameters_must_be_positive(kw):
    with pytest.raises(ValueError, match="must be positive"):
        onset_detect(np.zeros(4096), 8000, **kw)
    with pytest.raises(ValueError, match="must be positive"):
        onset_detect(np.zeros(4096), 0.0)


def test_onsets_agree_with_librosa():
    librosa = pytest.importorskip("librosa")
    agree = []
    for seed in range(10):
        x, _ = click_track(seed, 22050)
        ours = onset_detect(x, 22050)
        env = librosa.onset.onset_strength(y=x, sr=22050, hop_length=512, n_fft=2048)
        ref = librosa.onset.onset_detect(
            onset_envelope=env, sr=22050, hop_length=512, units="time"
        )
        agree.append(f1(ours, ref, 0.05))
    assert np.mean(agree) > 0.9
