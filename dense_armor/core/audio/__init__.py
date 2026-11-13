"""Audio: sound as a value, WAV files, windows, STFT, generators, filters, NMF and streams."""

from dense_armor.core.audio.adapter import AudioInput, run_estimators
from dense_armor.core.audio.allpole import AllPole, reflection_to_lpc
from dense_armor.core.audio.biquad import (
    Biquad,
    bandpass,
    highpass,
    highshelf,
    lowpass,
    lowshelf,
    notch,
    peaking,
)
from dense_armor.core.audio.blamp import (
    blamp_residual,
    hard_clip_blamp,
    triangle_blamp,
)
from dense_armor.core.audio.clipping import hard_clip
from dense_armor.core.audio.dynamics import (
    Compressor,
    Limiter,
    SoftClip,
    gain_curve,
)
from dense_armor.core.audio.envelope import envelope_follower, hilbert
from dense_armor.core.audio.events import detect_events
from dense_armor.core.audio.features import AudioFeatures
from dense_armor.core.audio.fir import FIR, DelayLine, convolve
from dense_armor.core.audio.fourier import (
    StreamingSTFT,
    cola_ok,
    istft,
    stft,
)
from dense_armor.core.audio.generators import (
    ADSR,
    KarplusStrong,
    Oscillator,
    pink_noise,
    plucked_string,
    white_noise,
)
from dense_armor.core.audio.nmf import nmf, separate
from dense_armor.core.audio.onset import onset_detect
from dense_armor.core.audio.perceptual import (
    hz_to_mel,
    mel_filterbank,
    mel_spectrogram,
    mel_to_hz,
    mfcc,
)
from dense_armor.core.audio.pitch import YIN, cmndf, yin_f0
from dense_armor.core.audio.pitch_units import (
    cents_to_freq,
    freq_to_cents,
    freq_to_note,
    note_to_freq,
)
from dense_armor.core.audio.resample import (
    Resampler,
    pitch_ratio_transition,
    rational_ratio,
    resample,
)
from dense_armor.core.audio.signal import Audio
from dense_armor.core.audio.spectral import (
    spectral_centroid,
    spectral_flatness,
    spectral_flux,
    spectral_rolloff,
)
from dense_armor.core.audio.streams import AudioFileStream, AudioLiveStream
from dense_armor.core.audio.wav import read_wav, write_wav
from dense_armor.core.audio.windows import (
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
    "YIN",
    "AllPole",
    "Audio",
    "AudioFeatures",
    "AudioFileStream",
    "AudioInput",
    "AudioLiveStream",
    "Biquad",
    "Compressor",
    "KarplusStrong",
    "Limiter",
    "Oscillator",
    "Resampler",
    "SoftClip",
    "StreamingSTFT",
    "bandpass",
    "blamp_residual",
    "blackman",
    "cents_to_freq",
    "cmndf",
    "cola_ok",
    "convolve",
    "detect_events",
    "envelope_follower",
    "freq_to_cents",
    "freq_to_note",
    "gain_curve",
    "get_window",
    "hamming",
    "hann",
    "hard_clip",
    "hard_clip_blamp",
    "highpass",
    "highshelf",
    "hilbert",
    "hz_to_mel",
    "istft",
    "lowpass",
    "lowshelf",
    "mel_filterbank",
    "mel_spectrogram",
    "mel_to_hz",
    "mfcc",
    "nmf",
    "note_to_freq",
    "notch",
    "onset_detect",
    "peaking",
    "pink_noise",
    "pitch_ratio_transition",
    "plucked_string",
    "rational_ratio",
    "read_wav",
    "rectangular",
    "reflection_to_lpc",
    "resample",
    "run_estimators",
    "separate",
    "spectral_centroid",
    "spectral_flatness",
    "spectral_flux",
    "spectral_rolloff",
    "stft",
    "triangle_blamp",
    "white_noise",
    "write_wav",
    "yin_f0",
]
