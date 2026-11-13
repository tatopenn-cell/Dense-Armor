"""Generate the figures of docs/audio/.

Run from the repository root:

    python docs/assets/audio/make_figures.py

Every curve is computed with the functions of ``dense_armor.core.audio``.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dense_armor.core.audio import (  # noqa: E402
    ADSR,
    YIN,
    AllPole,
    Audio,
    AudioFeatures,
    AudioInput,
    Compressor,
    Limiter,
    Oscillator,
    Phaser,
    SchroederReverb,
    SoftClip,
    bandpass,
    cmndf,
    detect_events,
    envelope_follower,
    gain_curve,
    get_window,
    hard_clip,
    hard_clip_blamp,
    highpass,
    highshelf,
    hilbert,
    hz_to_mel,
    jax_biquad,
    jax_lowpass,
    lowpass,
    lowshelf,
    mel_filterbank,
    mel_spectrogram,
    mfcc,
    nmf,
    notch,
    onset_detect,
    peaking,
    pink_noise,
    pitch_ratio_transition,
    pitch_shift,
    plucked_string,
    reflection_to_lpc,
    reverberation_time,
    spectral_flux,
    stft,
    time_stretch,
    triangle_blamp,
    white_noise,
)
from dense_armor.core.online.anomaly.filters import HampelScorer  # noqa: E402
from dense_armor.core.online.drift.detector import CUSUMDriftDetector  # noqa: E402

HERE = Path(__file__).resolve().parent
BG = "#fafafa"
BLUE, RED, GREEN, AMBER, VIOLET, GREY = (
    "#2563eb",
    "#dc2626",
    "#059669",
    "#d97706",
    "#7c3aed",
    "#6b7280",
)


def _style(ax):
    ax.set_facecolor("white")
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _save(fig, name):
    fig.savefig(HERE / name, dpi=110, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(name)


def _box(ax, x, y, w, h, text, color, fs=9.5):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            fc="white",
            ec=color,
            lw=1.8,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, p, q, color=GREY):
    ax.add_patch(
        FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, color=color, lw=1.6)
    )


def fig_adapter():
    """How AudioInput gives a block to a primitive."""
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _box(ax, 0.0, 0.37, 0.15, 0.26, "an Audio block\n\nsamples, sr, t0", GREY)
    _box(
        ax,
        0.21,
        0.37,
        0.17,
        0.26,
        "AudioInput\n\nlooks at the methods\nof the primitive",
        BLUE,
    )
    _arrow(ax, (0.17, 0.5), (0.19, 0.5))
    rows = [
        (
            0.70,
            "the dict of band\nmagnitudes b0 .. b7\n(AudioFeatures)",
            GREEN,
            "score_one, predict_one + learn_one,\ntransform_one:\nHampelScorer, OnlineGaussianNB,\nStandardScaler ...",
        ),
        (
            0.39,
            "one number:\nthe feature b0",
            AMBER,
            "update + drift_detected:\nCUSUMDriftDetector, ADWIN,\nPageHinkley",
        ),
        (
            0.08,
            "the samples\nof the block, mono",
            VIOLET,
            "only update_many:\nSDFT, Biquad, FIR, AllPole,\nYIN, Resampler",
        ),
    ]
    for y, label, color, prim in rows:
        _box(ax, 0.47, y, 0.2, 0.22, label, color, fs=9)
        _arrow(ax, (0.40, 0.5), (0.45, y + 0.11), color)
        _box(ax, 0.73, y, 0.27, 0.22, prim, color, fs=8.5)
        _arrow(ax, (0.69, y + 0.11), (0.71, y + 0.11), color)
    ax.set_title(
        "AudioInput opens sound to the primitives that already exist", fontsize=12
    )
    _save(fig, "adapter.png")


def fig_wave_spectrogram():
    a = plucked_string(440.0, 8000, 0.5)
    t = np.arange(a.n) / a.sr
    S = stft(a.x, 256, 64, "hann")
    db = 20 * np.log10(np.abs(S).T + 1e-9)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), facecolor=BG, sharex=True)
    _style(a1)
    a1.plot(t, a.x, color=BLUE, lw=0.8)
    a1.set_ylabel("amplitude")
    a1.set_title(
        "A plucked string, 440 Hz at 8000 samples per second: the sound dies away"
    )
    im = a2.imshow(
        db,
        origin="lower",
        aspect="auto",
        extent=[0, t[-1], 0, a.sr / 2],
        cmap="magma",
        vmin=db.max() - 70,
        vmax=db.max(),
    )
    a2.set_ylabel("frequency (Hz)")
    a2.set_xlabel("time (s)")
    a2.set_title("stft(x, 256, 64, 'hann'): the harmonics fade, the high ones first")
    fig.colorbar(im, ax=[a1, a2], label="dB", pad=0.02)
    _save(fig, "wave_spectrogram.png")


def fig_windows():
    n = 64
    names = ["rectangular", "hann", "hamming", "blackman"]
    cols = [GREY, BLUE, RED, GREEN]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=BG)
    for ax in (a1, a2):
        _style(ax)
    for name, c in zip(names, cols):
        w = get_window(name, n)
        a1.plot(np.arange(n), w, color=c, lw=1.8, label=name)
        W = np.abs(np.fft.rfft(w, 2048))
        W = 20 * np.log10(W / W.max() + 1e-12)
        a2.plot(np.arange(len(W)) * n / 2048, W, color=c, lw=1.2, label=name)
    a1.set_title("The four windows, 64 samples (periodic form)")
    a1.set_xlabel("sample")
    a1.legend(frameon=False)
    a2.set_xlim(0, 12)
    a2.set_ylim(-110, 3)
    a2.set_xlabel("frequency (in bins of the frame)")
    a2.set_ylabel("dB")
    a2.set_title("Their spectra: a narrow main lobe or low side lobes, not both")
    _save(fig, "windows.png")


def fig_biquads():
    sr, f0 = 8000.0, 1000.0
    designs = [
        ("lowpass(1000, sr)", lowpass(f0, sr)),
        ("highpass(1000, sr)", highpass(f0, sr)),
        ("bandpass(1000, sr, q=2)", bandpass(f0, sr, 2.0)),
        ("notch(1000, sr, q=2)", notch(f0, sr, 2.0)),
        ("peaking(1000, sr, +6 dB)", peaking(f0, sr, 6.0)),
        ("lowshelf(1000, sr, +6 dB)", lowshelf(f0, sr, 6.0)),
        ("highshelf(1000, sr, +6 dB)", highshelf(f0, sr, 6.0)),
    ]
    fr = np.geomspace(40, 3990, 600)
    zi = np.exp(-1j * 2 * np.pi * fr / sr)
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), facecolor=BG, sharey=True)
    for ax, (title, (b, a)) in zip(axes.ravel(), designs):
        _style(ax)
        h = np.polyval(b[::-1], zi) / np.polyval(a[::-1], zi)
        ax.semilogx(fr, 20 * np.log10(np.abs(h) + 1e-9), color=BLUE, lw=1.8)
        ax.axvline(f0, color=GREY, lw=0.9, ls="--")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-40, 12)
    axes.ravel()[-1].axis("off")
    for ax in axes[1, :3]:
        ax.set_xlabel("frequency (Hz)")
    for ax in axes[:, 0]:
        ax.set_ylabel("gain (dB)")
    fig.subplots_adjust(hspace=0.45)
    fig.suptitle(
        "The seven biquads of the Audio EQ Cookbook, sample rate 8000 Hz (dashed: 1000 Hz)"
    )
    _save(fig, "biquads.png")


def fig_allpole_sign():
    rng = np.random.default_rng(7)
    while True:
        k = rng.uniform(-0.99, 0.99, 5)
        a = reflection_to_lpc(k)
        if np.max(np.abs(np.roots(np.r_[1.0, -a]))) > 1.15:
            break
    imp = np.r_[1.0, np.zeros(59)]
    yu = AllPole(a).update_many(imp)
    hayes = AllPole(-a).update_many(imp)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=BG)
    for ax in (a1, a2):
        _style(ax)
        ax.set_xlabel("sample")
    a1.plot(yu, color=GREEN, lw=1.6, marker="o", ms=3)
    a1.set_title(
        "Sign of Yu, y = x - sum a y\nthe response dies away "
        f"(largest pole {np.max(np.abs(np.roots(np.r_[1.0, a]))):.2f})"
    )
    a2.semilogy(np.abs(hayes) + 1e-12, color=RED, lw=1.6, marker="o", ms=3)
    a2.set_ylabel("|y| (log scale)")
    a2.set_title(
        "Sign of Hayes Eq. 4, y = e + sum a y\nthe same coefficients blow up "
        f"(largest pole {np.max(np.abs(np.roots(np.r_[1.0, -a]))):.2f})"
    )
    _save(fig, "allpole_sign.png")


def fig_oscillators():
    sr, f = 8000, 700.0
    n = 36
    kinds = [
        ("sine", {}),
        ("sawtooth_additive", {}),
        ("square_additive", {}),
        ("sawtooth_polyblep", {}),
        ("square_polyblep", {}),
        ("pulse_polyblep", {"duty": 0.25}),
        ("triangle_blamp", {}),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), facecolor=BG, sharey=True)
    for ax, (kind, kw) in zip(axes.ravel(), kinds):
        _style(ax)
        o = Oscillator(kind, sr, **kw)
        y = o.update_many(np.full(n + 2, f))
        y = y[2:] if kind == "triangle_blamp" else y[:n]
        ax.plot(np.arange(len(y)), y, color=BLUE, lw=1.2, marker="o", ms=3)
        title = kind + (" (duty 0.25)" if kw else "")
        ax.set_title(title + (" *" if kind == "triangle_blamp" else ""), fontsize=10)
        ax.set_ylim(-1.45, 1.45)
    axes.ravel()[-1].axis("off")
    axes.ravel()[-1].text(
        0,
        0.5,
        "* triangle_blamp is two samples late;\n  here it is shifted back by 2",
        fontsize=9,
        color=GREY,
    )
    for ax in axes[1, :3]:
        ax.set_xlabel("sample")
    fig.subplots_adjust(hspace=0.45)
    fig.suptitle(
        "The seven kinds of Oscillator at 700 Hz, 8000 samples per second (dots are the samples)"
    )
    _save(fig, "oscillators.png")


def _db_spectrum(x):
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return 20 * np.log10(X / X.max() + 1e-12)


def _off_db(x, sr, f0):
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    m = np.zeros(len(X), bool)
    for h in range(1, int(sr / 2 / f0) + 1):
        c = int(round(h * f0))
        m[max(0, c - 3) : c + 4] = True
    return 10 * np.log10(X[~m].sum() / X.sum())


def fig_aliasing():
    sr, f0 = 44100, 3141.5926
    ph = (np.arange(sr) * f0 / sr) % 1
    tri = 1 - 4 * np.abs(ph - 0.5)
    s = 2 * np.sin(2 * np.pi * f0 * np.arange(sr) / sr)
    saw = 2 * ph - 1
    cases = [
        ("Triangle wave", tri, triangle_blamp(f0, sr, sr), "polyBLAMP"),
        (
            "Sine of amplitude 2, clipped at 1",
            hard_clip(s, 1.0),
            hard_clip_blamp(s, 1.0),
            "polyBLAMP",
        ),
        (
            "Sawtooth",
            saw,
            Oscillator("sawtooth_polyblep", sr).update_many(np.full(sr, f0)),
            "polyBLEP",
        ),
    ]
    fr = np.fft.rfftfreq(sr, 1 / sr) / 1000
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), facecolor=BG, sharey=True)
    for ax, (title, naive, fixed, how) in zip(axes, cases):
        _style(ax)
        ax.plot(
            fr,
            _db_spectrum(naive),
            color=RED,
            lw=0.7,
            alpha=0.85,
            label=f"naive: {_off_db(naive, sr, f0):.1f} dB off the harmonics",
        )
        ax.plot(
            fr,
            _db_spectrum(fixed),
            color=GREEN,
            lw=0.7,
            alpha=0.9,
            label=f"{how}: {_off_db(fixed, sr, f0):.1f} dB off the harmonics",
        )
        ax.set_title(title)
        ax.set_xlabel("frequency (kHz)")
        ax.set_ylim(-120, 5)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    axes[0].set_ylabel("dB")
    fig.suptitle(
        "Aliasing at 3141.5926 Hz and 44100 Hz: the energy that falls between the harmonics"
    )
    _save(fig, "aliasing.png")


def fig_adsr_pink():
    e = ADSR(0.1, 0.2, 0.5, 0.3, 1.0, 1000)
    t = np.linspace(0, 1.5, 600)
    v = [e.value(x) for x in t]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=BG)
    for ax in (a1, a2):
        _style(ax)
    a1.plot(t, v, color=BLUE, lw=2)
    for x, lab, ha in [
        (0.1, "attack ends", "right"),
        (0.3, "decay ends", "left"),
        (1.0, "note off", "right"),
        (1.3, "release ends", "left"),
    ]:
        a1.axvline(x, color=GREY, ls="--", lw=0.9)
        a1.text(x, 1.05, lab, fontsize=8, ha=ha, color=GREY)
    a1.set_ylim(-0.05, 1.15)
    a1.set_xlabel("time (s)")
    a1.set_title("ADSR(0.1, 0.2, 0.5, 0.3, 1.0, 1000)")
    x = pink_noise(1 << 16, 3)
    X = np.abs(np.fft.rfft(x)) ** 2
    k = np.arange(1, len(X))
    a2.loglog(k, X[1:], ".", ms=1, color=BLUE, alpha=0.25)
    slope, icpt = np.polyfit(np.log(k), np.log(X[1:]), 1)
    a2.loglog(
        k, np.exp(icpt) * k**slope, color=RED, lw=2, label=f"fit: slope {slope:.2f}"
    )
    a2.set_xlabel("frequency bin")
    a2.set_ylabel("power")
    a2.set_title("pink_noise(65536, 3): the power falls as 1 / frequency")
    a2.legend(frameon=False)
    _save(fig, "adsr_pink.png")


def fig_yin():
    sr = 8000
    x = np.sin(2 * np.pi * 440 * np.arange(1024) / sr)
    d = cmndf(x, int(np.ceil(sr / 100.0)))
    tau = 18
    den = d[tau - 1] - 2 * d[tau] + d[tau + 1]
    delta = 0.5 * (d[tau - 1] - d[tau + 1]) / den
    est = sr / (tau + delta)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=BG)
    for ax in (a1, a2):
        _style(ax)
    a1.plot(np.arange(len(d)), d, color=BLUE, lw=1.6, marker="o", ms=3)
    a1.axhline(0.1, color=RED, ls="--", lw=1, label="threshold 0.1")
    a1.axvline(
        tau + delta,
        color=GREEN,
        lw=1.2,
        label=f"period {tau + delta:.2f} samples = {est:.1f} Hz",
    )
    a1.set_xlabel("lag (samples)")
    a1.set_ylabel("d'(lag)")
    a1.set_title(
        "The normalised difference function of a 440 Hz sine\n(first dip below the threshold, then a parabola)"
    )
    a1.legend(frameon=False, fontsize=9)
    dur, blk, hop = 2.0, 1024, 512
    fr = np.linspace(150, 400, int(sr * dur))
    sig = Oscillator("sine", sr).update_many(fr)
    centres, ests = [], []
    ai = AudioInput(YIN(sr, 80.0, 0.1))
    for i in range(0, len(sig) - blk + 1, hop):
        ests.append(ai.feed(Audio(sig[i : i + blk], sr, i / sr)))
        centres.append((i + blk / 2) / sr)
    a2.plot(
        np.linspace(0, dur, len(fr)),
        fr,
        color=GREY,
        lw=1.5,
        label="frequency given to the oscillator",
    )
    a2.plot(
        centres, ests, "o", ms=4, color=GREEN, label="YIN, one block of 1024 samples"
    )
    a2.set_xlabel("time (s)")
    a2.set_ylabel("Hz")
    a2.set_title("YIN follows a glide from 150 to 400 Hz")
    a2.legend(frameon=False, fontsize=9)
    _save(fig, "yin.png")


def fig_glide():
    steps = 20
    lin = pitch_ratio_transition(1.0, 2.0, 1.0, 1.0, steps, "ratio-linear")
    cen = pitch_ratio_transition(1.0, 2.0, 1.0, 1.0, steps, "cents-linear")
    j = np.arange(steps + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=BG)
    for ax in (a1, a2):
        _style(ax)
        ax.set_xlabel("step")
    a1.plot(j, lin, color=AMBER, lw=1.8, marker="o", ms=3, label="ratio-linear")
    a1.plot(j, cen, color=BLUE, lw=1.8, marker="o", ms=3, label="cents-linear")
    a1.set_ylabel("step q of the Resampler")
    a1.set_title("Algorithm 4: the steps of an octave glide")
    a1.legend(frameon=False)
    a2.plot(
        j,
        1200 * np.log2(lin),
        color=AMBER,
        lw=1.8,
        marker="o",
        ms=3,
        label="ratio-linear",
    )
    a2.plot(
        j,
        1200 * np.log2(cen),
        color=BLUE,
        lw=1.8,
        marker="o",
        ms=3,
        label="cents-linear",
    )
    a2.set_ylabel("pitch (cents above the start)")
    a2.set_title("The same two paths as the ear hears them")
    a2.legend(frameon=False)
    _save(fig, "glide.png")


def fig_nmf():
    sr = 8000
    t = np.arange(sr * 2) / sr
    x = np.where(t < 1, np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 1000 * t))
    S = stft(x, 512, 256, "hann")
    V = np.abs(S).T
    W, H, _ = nmf(V, 2, 200, "euclidean")
    freqs = np.arange(V.shape[0]) * sr / 512
    times = (np.arange(V.shape[1]) * 256 + 256) / sr
    db = 20 * np.log10(V + 1e-6)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), facecolor=BG)
    axes[0].imshow(
        db,
        origin="lower",
        aspect="auto",
        extent=[0, 2, 0, sr / 2],
        cmap="magma",
        vmin=db.max() - 70,
    )
    axes[0].set_title("The mixture: 440 Hz, then 1000 Hz")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("frequency (Hz)")
    cols = [BLUE, RED]
    for j in range(2):
        _style(axes[1])
        axes[1].plot(freqs, W[:, j], color=cols[j], lw=1.6, label=f"part {j}")
        _style(axes[2])
        axes[2].plot(times, H[j], color=cols[j], lw=1.6, label=f"part {j}")
    axes[1].set_xlim(0, 2000)
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_title("W: what each part sounds like")
    axes[1].legend(frameon=False)
    axes[2].set_xlabel("time (s)")
    axes[2].set_title("H: when each part is on")
    axes[2].legend(frameon=False)
    _save(fig, "nmf.png")


def fig_events():
    t = np.arange(24000) / 8000
    x = 0.1 * white_noise(24000, 1) + np.where(
        t >= 2.0, np.sin(2 * np.pi * 440 * t), 0.0
    )
    blocks = [Audio(x[i : i + 512], 8000, i / 8000) for i in range(0, 24000 - 511, 512)]
    feat = AudioFeatures(8)
    b0 = [feat.transform(b)["b0"] for b in blocks]
    scores = AudioInput(HampelScorer()).run(blocks)
    det = CUSUMDriftDetector()
    alarms = []
    for b, s in zip(blocks, scores):
        det.update(s)
        if det.drift_detected:
            alarms.append(b.t0)
    assert alarms == [h[0] for h in detect_events(blocks)]
    t0 = [b.t0 for b in blocks]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), facecolor=BG, sharex=True)
    for ax in axes:
        _style(ax)
        ax.axvline(2.0, color=GREY, ls="--", lw=1)
        for a in alarms[:1]:
            ax.axvline(a, color=RED, lw=1.6)
    axes[0].plot(t, x, color=BLUE, lw=0.5)
    axes[0].set_ylabel("sound")
    axes[0].set_title("Noise for three seconds, with a 440 Hz tone from 2.0 s (dashed)")
    axes[1].plot(t0, b0, color=GREEN, lw=1.4, marker="o", ms=3)
    axes[1].set_ylabel("feature b0\n(lowest band)")
    sc = np.minimum(np.asarray(scores, float), 1e3)
    axes[2].semilogy(t0, np.maximum(sc, 1e-3), color=AMBER, lw=1.4, marker="o", ms=3)
    axes[2].set_ylabel("HampelScorer score")
    axes[2].set_xlabel("time (s)")
    axes[2].set_title(
        f"First alarm (red): {alarms[0]:.3f} s, the block that holds the start of the tone",
        fontsize=10,
    )
    _save(fig, "events.png")


def fig_mel():
    fb = mel_filterbank(20, 1024, 8000.0)
    f = np.linspace(0, 4000.0, fb.shape[1])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=BG)
    for ax in (a1, a2):
        _style(ax)
    cmap = plt.get_cmap("viridis")
    for i, row in enumerate(fb):
        a1.plot(f, row, color=cmap(i / 19), lw=1.4)
    a1.set_xlabel("frequency (Hz)")
    a1.set_ylabel("weight")
    a1.set_title("mel_filterbank(20, 1024, 8000): narrow low, wide high, same area")
    hz = np.linspace(0, 8000, 800)
    a2.plot(hz, hz_to_mel(hz), color=BLUE, lw=2)
    a2.axvline(1000, color=GREY, ls="--", lw=0.9)
    a2.text(1100, 3, "1 kHz: line, then logarithm", fontsize=9, color=GREY)
    a2.set_xlabel("frequency (Hz)")
    a2.set_ylabel("mel")
    a2.set_title("hz_to_mel: a straight line to 1 kHz, then a logarithm")
    _save(fig, "mel.png")


def fig_mel_mfcc():
    sr = 8000
    t = np.arange(int(1.5 * sr)) / sr
    x = np.sin(2 * np.pi * np.where(t < 0.75, 440.0, 1760.0) * t)
    mel = mel_spectrogram(x, float(sr), 40, 512, 128)
    cep = mfcc(x, float(sr), 13, 40, 512, 128)
    ts = (np.arange(len(mel)) * 128 + 256) / sr
    db = 10 * np.log10(np.maximum(mel, 1e-10)).T
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=BG)
    im = a1.imshow(
        db,
        origin="lower",
        aspect="auto",
        extent=[ts[0], ts[-1], 0, 40],
        cmap="magma",
        vmin=db.max() - 70,
        vmax=db.max(),
    )
    a1.set_xlabel("time (s)")
    a1.set_ylabel("mel filter")
    a1.set_title("mel_spectrogram: 440 Hz, then 1760 Hz")
    fig.colorbar(im, ax=a1, label="dB", pad=0.02)
    im2 = a2.imshow(
        cep.T,
        origin="lower",
        aspect="auto",
        extent=[ts[0], ts[-1], -0.5, 12.5],
        cmap="coolwarm",
    )
    a2.set_xlabel("time (s)")
    a2.set_ylabel("coefficient")
    a2.set_title("mfcc: 13 numbers per frame, they change when the tone changes")
    fig.colorbar(im2, ax=a2, pad=0.02)
    _save(fig, "mel_mfcc.png")


def fig_onset():
    sr = 8000
    rng = np.random.default_rng(5)
    x = 0.005 * rng.standard_normal(3 * sr)
    truth = np.array([0.4, 1.0, 1.35, 2.2, 2.65])
    for t0 in truth:
        i = int(t0 * sr)
        m = int(0.02 * sr)
        x[i : i + m] += (
            np.exp(-np.arange(m) / (0.004 * sr)) * rng.standard_normal(m) * 0.8
        )
    found = onset_detect(x, float(sr), 1024, 256)
    flux = spectral_flux(x, 1024, 256)
    tf = (np.arange(len(flux)) * 256 + 512) / sr
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.6), facecolor=BG, sharex=True)
    for ax in (a1, a2):
        _style(ax)
    a1.plot(np.arange(len(x)) / sr, x, color=BLUE, lw=0.6)
    for k, t0 in enumerate(truth):
        a1.axvline(t0, color=GREEN, ls="--", lw=1.1, label="click" if k == 0 else None)
    for k, t0 in enumerate(found):
        a1.axvline(t0, color=RED, lw=1.1, label="onset_detect" if k == 0 else None)
    a1.set_ylabel("amplitude")
    a1.legend(frameon=False, loc="upper right")
    a1.set_title(
        f"Five clicks: {len(found)} onsets, each within {np.abs(found - truth).max() * 1000:.0f} ms"
    )
    a2.plot(tf, flux / flux.max(), color=VIOLET, lw=1.4)
    a2.set_ylabel("spectral flux")
    a2.set_xlabel("time (s)")
    a2.set_title("spectral_flux: it rises when the spectrum gains energy")
    _save(fig, "onset.png")


def fig_envelope():
    sr = 8000
    e = ADSR(0.1, 0.2, 0.5, 0.3, 1.0, sr)
    t = np.arange(int(1.6 * sr)) / sr
    amp = np.array([e.value(v) for v in t])
    x = amp * np.sin(2 * np.pi * 440.0 * t)
    raw = np.abs(hilbert(x))
    env = envelope_follower(x, float(sr), 0.01, 0.1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=BG, sharey=True)
    for ax in (a1, a2):
        _style(ax)
    a1.plot(t, x, color=BLUE, lw=0.4, alpha=0.6, label="the sound")
    a1.plot(t, amp, color=GREEN, lw=1.8, label="ADSR that made it")
    a1.set_title("A 440 Hz note shaped by ADSR(0.1, 0.2, 0.5, 0.3, 1.0)")
    a1.legend(frameon=False, loc="upper right")
    a2.plot(t, raw, color=AMBER, lw=1.2, label="|hilbert(x)|")
    a2.plot(
        t, env, color=RED, lw=2, label="envelope_follower: attack 10 ms, release 100 ms"
    )
    a2.set_title("The envelope: raw, and smoothed with an attack and a release")
    a2.legend(frameon=False, loc="upper right")
    for ax in (a1, a2):
        ax.set_xlabel("time (s)")
    a1.set_ylabel("amplitude")
    _save(fig, "envelope.png")


def fig_compressor():
    x = np.linspace(-60, 0, 301)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=BG)
    for ax in (a1, a2):
        _style(ax)
    a1.plot(x, x, color=GREY, ls="--", lw=1, label="no compression")
    a1.plot(x, gain_curve(x, -20, 4, 0), color=BLUE, lw=2, label="ratio 4, knee 0")
    a1.plot(x, gain_curve(x, -20, 4, 12), color=RED, lw=2, label="ratio 4, knee 12 dB")
    a1.axvline(-20, color=GREY, ls=":", lw=1)
    a1.text(-19.5, -58, "threshold", fontsize=8, color=GREY)
    a1.set_xlabel("input level (dB)")
    a1.set_ylabel("output level (dB)")
    a1.set_title("gain_curve: above the threshold 4 dB in give 1 dB out")
    a1.legend(frameon=False, loc="upper left")
    sr = 8000.0
    t = np.arange(int(1.0 * sr)) / sr
    amp = np.where((t > 0.25) & (t < 0.65), 10 ** (-8 / 20), 10 ** (-35 / 20))
    sig = amp * np.sin(2 * np.pi * 440 * t)
    c = Compressor(-20.0, 4.0, 0.02, 0.1, 0.0, 0.0, sr)
    out = np.empty(len(sig))
    gr = np.empty(len(sig))
    for i, v in enumerate(sig):
        out[i] = c.update(v).out
        gr[i] = c.gain_reduction_db
    a2.plot(t, sig, color=BLUE, lw=0.5, alpha=0.6, label="input")
    a2.plot(t, out, color=RED, lw=0.6, label="output")
    a2.set_xlabel("time (s)")
    a2.set_ylabel("amplitude")
    b2 = a2.twinx()
    b2.plot(t, gr, color=GREEN, lw=2, label="gain reduction (dB)")
    b2.set_ylabel("gain reduction (dB)", color=GREEN)
    a2.set_title(
        "A burst with its peak 12 dB above the threshold: the gain falls, then recovers"
    )
    a2.legend(frameon=False, loc="upper right")
    _save(fig, "compressor.png")


def fig_limiter():
    sr = 8000.0
    t = np.arange(int(0.8 * sr)) / sr
    amp = np.where((t > 0.3) & (t < 0.45), 1.2, 0.25)
    x = amp * np.sin(2 * np.pi * 200 * t)
    lim = Limiter(-6.0, 0.005, 0.05, sr)
    y = lim.update_many(x)
    ceiling = 10 ** (-6 / 20)
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=BG)
    _style(ax)
    ax.plot(t, x, color=BLUE, lw=0.8, label="input")
    ax.plot(t, y, color=RED, lw=1.0, label="output, delayed by 5 ms")
    for s in (1, -1):
        ax.axhline(s * ceiling, color=GREY, ls="--", lw=1)
    ax.text(0.01, ceiling + 0.03, "ceiling -6 dB", fontsize=8, color=GREY)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude")
    peak_db = 20 * np.log10(np.abs(y).max())
    ax.set_title(
        f"Limiter(-6, 0.005): the burst is brought to the ceiling, output peak {peak_db:.2f} dB"
    )
    ax.legend(frameon=False, loc="upper right")
    _save(fig, "limiter.png")


def fig_softclip():
    sr = 44100.0
    f0 = 5000.0
    x = np.sin(2 * np.pi * f0 * np.arange(int(sr)) / sr)
    seg = slice(4410, 4410 + 8820)
    fig, ax = plt.subplots(figsize=(10, 4.4), facecolor=BG)
    _style(ax)
    for os, col in ((1, RED), (4, BLUE)):
        y = SoftClip(12.0, os, sr).update_many(x)[seg]
        spec = np.abs(np.fft.rfft(y)) ** 2
        db = 10 * np.log10(spec / spec.max() + 1e-14)
        f = np.fft.rfftfreq(len(y), 1 / sr) / 1000
        step = int(round(f0 * len(y) / sr))
        mask = np.ones(len(spec), bool)
        mask[::step] = False
        mask[0] = False
        alias = 10 * np.log10(spec[mask].sum() / spec[step])
        ax.plot(
            f,
            db,
            color=col,
            lw=0.8,
            label=f"oversample {os}: the rest is {alias:.0f} dB below the tone",
        )
    ax.set_ylim(-120, 5)
    ax.set_xlabel("frequency (kHz)")
    ax.set_ylabel("dB")
    ax.set_title("SoftClip(12): a 5 kHz sine, with and without oversampling")
    ax.legend(frameon=False, loc="upper right")
    _save(fig, "softclip.png")


def fig_reverb():
    sr = 8000.0
    r = SchroederReverb(0.5, 0.0, 1.0, 0.0, sr)
    h = r.update_many(np.r_[1.0, np.zeros(int(2.8 * sr) - 1)])
    t = np.arange(len(h)) / sr
    edc = np.cumsum((h**2)[::-1])[::-1]
    db = 10 * np.log10(np.maximum(edc / edc[0], 1e-12))
    measured = reverberation_time(h, sr)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), facecolor=BG, sharex=True)
    for ax in (a1, a2):
        _style(ax)
    a1.plot(t, h, color=BLUE, lw=0.5)
    a1.set_ylabel("amplitude")
    a1.set_title("The response of SchroederReverb(0.5, 0) to one impulse")
    a2.plot(t, db, color=VIOLET, lw=1.6, label="energy decay (dB)")
    a2.axhline(-60, color=GREY, ls=":", lw=1)
    a2.axvline(r.t60_s, color=GREEN, ls="--", lw=1.4, label=f"asked: {r.t60_s:.2f} s")
    a2.axvline(measured, color=RED, lw=1.2, label=f"measured: {measured:.2f} s")
    a2.set_ylim(-80, 3)
    a2.set_xlabel("time (s)")
    a2.set_ylabel("dB")
    a2.set_title("The time to fall by 60 dB is the one that was asked")
    a2.legend(frameon=False, loc="upper right")
    _save(fig, "reverb.png")


def fig_modulation():
    sr = 8000.0
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=BG)
    for ax in (a1, a2):
        _style(ax)
    t = np.linspace(0, 4, 800)
    a1.plot(t, 15.0 + 5.0 * np.sin(2 * np.pi * 0.5 * t), color=BLUE, lw=2)
    a1.set_xlabel("time (s)")
    a1.set_ylabel("delay (ms)")
    a1.set_title("Chorus(0.5, 5): the delay swings between 10 and 20 ms")
    p = Phaser(0.05, 1.0, 4, sr, 0.5)
    n_late = int(sr / (4 * 0.05))
    for n, col in ((0, BLUE), (n_late, RED)):
        q = p.clone()
        q._n = n
        h = q.update_many(np.r_[1.0, np.zeros(4095)])
        f = np.fft.rfftfreq(len(h), 1 / sr)
        a2.plot(
            f,
            np.abs(np.fft.rfft(h)),
            color=col,
            lw=1.4,
            label=f"corner {q.corner_frequency(n):.0f} Hz",
        )
    a2.set_xlim(0, 4000)
    a2.set_xlabel("frequency (Hz)")
    a2.set_ylabel("gain")
    a2.set_title("Phaser(0.05, 1, 4): two notches, and they move with the LFO")
    a2.legend(frameon=False)
    _save(fig, "modulation.png")


def fig_timescale():
    sr = 8000
    t = np.arange(sr) / sr
    x = np.sin(2 * np.pi * np.where(t < 0.5, 440.0, 660.0) * t)
    cases = [
        ("the sound: 440 Hz, then 660 Hz", x),
        ("time_stretch(x, sr, 2): twice as long, same pitch", time_stretch(x, sr, 2.0)),
        ("pitch_shift(x, sr, 12): same length, an octave up", pitch_shift(x, sr, 12.0)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor=BG)
    for ax, (title, y) in zip(axes, cases):
        spec = stft(y, 256, 64, "hann")
        db = 20 * np.log10(np.abs(spec).T + 1e-9)
        ax.imshow(
            db,
            origin="lower",
            aspect="auto",
            extent=[0, len(y) / sr, 0, sr / 2],
            cmap="magma",
            vmin=db.max() - 70,
            vmax=db.max(),
        )
        ax.set_ylim(0, 2000)
        ax.set_xlabel("time (s)")
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("frequency (Hz)")
    _save(fig, "timescale.png")


def fig_differentiable():
    import jax
    import jax.numpy as jnp

    sr = 8000.0
    x = jnp.asarray(np.random.default_rng(0).standard_normal(2000), dtype=jnp.float32)
    target = jax_biquad(*jax_lowpass(1000.0, sr), x)

    def loss(log_f0):
        return jnp.mean(
            (jax_biquad(*jax_lowpass(jnp.exp(log_f0), sr), x) - target) ** 2
        )

    step = jax.jit(jax.value_and_grad(loss))
    u = jnp.log(2000.0)
    history = []
    for _ in range(60):
        value, grad = step(u)
        history.append((float(jnp.exp(u)), float(value)))
        u = u - 2.0 * grad
    w = np.linspace(0.001, np.pi, 600)
    z = np.exp(-1j * w)

    def response(f0):
        b, a = lowpass(f0, sr)
        return 20 * np.log10(
            np.abs((b[0] + b[1] * z + b[2] * z**2) / (1 + a[1] * z + a[2] * z**2))
        )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=BG)
    ax = axes[0]
    _style(ax)
    for n, color in ((0, AMBER), (1, VIOLET), (2, GREEN)):
        ax.plot(
            w * sr / (2 * np.pi),
            response(history[n][0]),
            color=color,
            label=f"step {n}: {history[n][0]:.0f} Hz",
        )
    ax.plot(
        w * sr / (2 * np.pi),
        response(1000.0),
        "--",
        color=GREY,
        label="target: 1000 Hz",
    )
    ax.set_ylim(-40, 5)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("magnitude (dB)")
    ax.set_title("the cutoff moves to the target", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    _style(ax)
    ax.semilogy(range(len(history)), [h[1] + 1e-16 for h in history], color=BLUE)
    ax.set_xlabel("gradient step")
    ax.set_ylabel("mean squared error")
    ax.set_title("the loss, through jax_lowpass and jax_biquad", fontsize=10)
    _save(fig, "differentiable.png")


if __name__ == "__main__":
    for f in (
        fig_adapter,
        fig_wave_spectrogram,
        fig_windows,
        fig_biquads,
        fig_allpole_sign,
        fig_oscillators,
        fig_aliasing,
        fig_adsr_pink,
        fig_yin,
        fig_glide,
        fig_nmf,
        fig_events,
        fig_mel,
        fig_mel_mfcc,
        fig_onset,
        fig_envelope,
        fig_compressor,
        fig_limiter,
        fig_softclip,
        fig_reverb,
        fig_modulation,
        fig_timescale,
        fig_differentiable,
    ):
        f()
