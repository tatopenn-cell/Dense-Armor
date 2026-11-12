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
    Oscillator,
    bandpass,
    cmndf,
    detect_events,
    get_window,
    hard_clip,
    hard_clip_blamp,
    highpass,
    highshelf,
    lowpass,
    lowshelf,
    nmf,
    notch,
    peaking,
    pink_noise,
    pitch_ratio_transition,
    plucked_string,
    reflection_to_lpc,
    stft,
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
    ):
        f()
