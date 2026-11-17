"""End-to-end: the primitives on a committed wav give the committed reference arrays."""

from pathlib import Path

import numpy as np
import pytest
from gen_golden import GOLDEN, OUTPUTS, SR

from dense_armor.core.audio import read_wav


def test_the_wav_is_one_second_of_440_hz_at_8_khz_pcm16():
    wav = GOLDEN / "sine_440.wav"
    audio = read_wav(wav)
    assert audio.sr == SR and len(audio.x) == SR
    assert abs(np.abs(audio.x).max() - 0.5) < 1e-3
    spectrum = np.abs(np.fft.rfft(audio.x))
    assert int(spectrum.argmax()) == 440
    assert wav.read_bytes()[20:22] == b"\x01\x00"
    assert wav.read_bytes()[34:36] == b"\x10\x00"


@pytest.mark.parametrize("name", sorted(OUTPUTS))
def test_the_primitive_reproduces_the_golden_file(name):
    x = read_wav(GOLDEN / "sine_440.wav").x
    got = OUTPUTS[name](x)
    ref = np.load(GOLDEN / name)
    assert got.shape == ref.shape
    assert np.allclose(got, ref, rtol=1e-6, atol=1e-8)


def test_every_golden_file_is_listed_and_every_listed_file_exists():
    on_disk = {p.name for p in Path(GOLDEN).glob("*.npy")}
    assert on_disk == set(OUTPUTS)
