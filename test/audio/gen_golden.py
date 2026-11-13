"""Writes the golden files of ``test/audio/golden/``. Run it once; the files are committed.

``sine_440.wav`` is one second of a 440 Hz sine of amplitude 0.5 at 8 kHz, PCM 16 bits. Every
other file is the output of one primitive on the samples READ BACK from that file (so the
quantisation of the PCM is part of the reference). ``test_golden.py`` loads the wav, applies the
same primitives and compares with the ``.npy`` files. Changing a file means changing the
behaviour of a primitive on purpose, and the commit says so.

    python test/audio/gen_golden.py
"""

from pathlib import Path

import jax
import numpy as np

from dense_armor.core.audio import (
    Audio,
    Compressor,
    JaxBiquad,
    SchroederReverb,
    jax_lowpass,
    mel_spectrogram,
    mfcc,
    onset_detect,
    pitch_shift,
    read_wav,
    time_stretch,
    write_wav,
)

GOLDEN = Path(__file__).resolve().parent / "golden"
SR = 8000


def _jax_lowpass_output(x):
    """``JaxBiquad`` with the cookbook low-pass, in float64 so the file does not depend on float32 rounding."""
    old = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", True)
    try:
        b, a = jax_lowpass(1000.0, float(SR), 0.707)
        return np.asarray(JaxBiquad(b, a).forward(np.asarray(x, dtype=np.float64)))
    finally:
        jax.config.update("jax_enable_x64", old)


OUTPUTS = {
    "sine_440_mel.npy": lambda x: mel_spectrogram(x, SR),
    "sine_440_mfcc.npy": lambda x: mfcc(x, SR),
    "sine_440_onset.npy": lambda x: onset_detect(x, SR),
    "sine_440_compressed.npy": lambda x: Compressor(sr=SR).update_many(x),
    "sine_440_reverb.npy": lambda x: SchroederReverb(sr=SR).update_many(x),
    "sine_440_stretch_2x.npy": lambda x: time_stretch(x, SR, 2.0),
    "sine_440_shift_up_octave.npy": lambda x: pitch_shift(x, SR, 12.0),
    "sine_440_jaxlowpass.npy": _jax_lowpass_output,
}


def make_wav(path: Path) -> None:
    t = np.arange(SR) / SR
    write_wav(path, Audio(0.5 * np.sin(2 * np.pi * 440.0 * t), SR), "pcm16")


def main() -> None:
    GOLDEN.mkdir(exist_ok=True)
    wav = GOLDEN / "sine_440.wav"
    if not wav.exists():
        make_wav(wav)
    x = read_wav(wav).x
    for name, fn in OUTPUTS.items():
        np.save(GOLDEN / name, fn(x))
        print(name, np.load(GOLDEN / name).shape)


if __name__ == "__main__":
    main()
