"""Audio: sound as a value, WAV files, windows, STFT, generators, filters, NMF and streams."""

from dense_armor.utility.audio.biquad import (
    Biquad,
    bandpass,
    highpass,
    highshelf,
    lowpass,
    lowshelf,
    notch,
    peaking,
)
from dense_armor.utility.audio.blamp import (
    blamp_residual,
    hard_clip_blamp,
    triangle_blamp,
)
from dense_armor.utility.audio.clipping import hard_clip
from dense_armor.utility.audio.events import detect_events
from dense_armor.utility.audio.features import AudioFeatures, run_estimators
from dense_armor.utility.audio.fir import FIR, DelayLine, convolve
from dense_armor.utility.audio.fourier import (
    StreamingSTFT,
    cola_ok,
    istft,
    stft,
)
from dense_armor.utility.audio.generators import (
    ADSR,
    KarplusStrong,
    Oscillator,
    chirp,
    pink_noise,
    plucked_string,
    white_noise,
)
from dense_armor.utility.audio.nmf import nmf, separate
from dense_armor.utility.audio.pitch_units import (
    cents_to_freq,
    freq_to_cents,
    freq_to_note,
    note_to_freq,
)
from dense_armor.utility.audio.resample import Resampler, resample
from dense_armor.utility.audio.signal import Audio
from dense_armor.utility.audio.streams import AudioFileStream, AudioLiveStream
from dense_armor.utility.audio.wav import read_wav, write_wav
from dense_armor.utility.audio.windows import (
    blackman,
    get_window,
    hamming,
    hann,
    rectangular,
)

__all__ = [
    "ADSR",
    "DelayLine",
    "FIR",
    "Audio",
    "AudioFeatures",
    "AudioFileStream",
    "AudioLiveStream",
    "Biquad",
    "KarplusStrong",
    "Oscillator",
    "Resampler",
    "StreamingSTFT",
    "bandpass",
    "blamp_residual",
    "blackman",
    "cents_to_freq",
    "chirp",
    "cola_ok",
    "convolve",
    "detect_events",
    "freq_to_cents",
    "freq_to_note",
    "get_window",
    "hamming",
    "hann",
    "hard_clip",
    "hard_clip_blamp",
    "highpass",
    "highshelf",
    "istft",
    "lowpass",
    "lowshelf",
    "nmf",
    "note_to_freq",
    "notch",
    "peaking",
    "pink_noise",
    "plucked_string",
    "read_wav",
    "rectangular",
    "resample",
    "run_estimators",
    "separate",
    "stft",
    "triangle_blamp",
    "white_noise",
    "write_wav",
]
