"""
core/online/protect/arbiter.py
==============================
Routes each point to the right specialist instead of forcing a single one
on the whole signal -- born from the real comparison in
test/test_pressure_valve_vs_orca_experiment.py: hybrid_shield (phi_ab)
clearly wins on isolated impulses (A, B, C, G) but loses on a sustained
regime change (F), pressure_valve does the exact opposite.
Neither beats the other everywhere -- the real difference between the two
cases is the DURATION of the deviation, not its amplitude.

APPROACH: each point is compared with a WIDE reference window
(radius*ref_mult, the same scale used by the JSD "spring" of
pressure_valve) -- deliberately not with the narrow local window, which in
the benchmark was seen to collapse (median/MAD/IQR exactly 0) when several
nearby outliers end up in the same small window.
This window is CAUSAL (_window_causal: only points BEFORE i, never after)
-- a symmetric window, used right across a level transition, would mix the
old and the new level in its own scale, diluting the deviation that should
trigger the threshold (this was the case in the first version of this
module, fixed after a real case that was not detected -- see the history in
CHANGELOG.md [1.1.10]).
Once the "deviating" points are marked, the LENGTH of the consecutive run
of deviating points decides: short (<= spike_run_max) = isolated impulse,
long = genuine regime change.

DEGENERATE CASE (exactly flat baseline, robust scale = 0): this is not an
estimator problem -- see the KNOWN LIMIT in the docstring of
pressure_valve, a Huber M-estimator converges to scale 0 anyway for the
same structural reason. Here we never divide by a zero scale:
if the dispersion of the reference window is below epsilon, ANY non-zero
departure from the median is treated as deviating, without needing a
threshold in sigma (on a truly constant baseline, "different from
constant" is already the answer, nothing more is needed).

HISTORY (limit found and then fixed in the same session): the first
version used a SYMMETRIC reference window, and scenario F (structural
break, permanent level jump) was not detected as 'regime' --
deviazione[i] stayed 0 over the whole sequence around the transition,
because the symmetric window, used right across the jump, contained half
the old level and half the new one, inflating its own scale (MAD) instead
of staying small. A first test had mistaken this for a success (RMSE 0.0
with route_and_correct standalone), but it was an artifact: the raw data
of that scenario already coincide EXACTLY with the target (the
"corruption" is the jump itself, not added noise), so "pass everything
raw" gives RMSE 0 regardless of how each point is labelled -- the real
test was the integrated Orca pipeline (core/online/protect/orca.py,
_applica_arbitro), where the missed detection translated into SAFE but
not improving behaviour (9.4007 identical with or without the arbiter).
With the CAUSAL window (above), detection becomes real and verifiable:
on scenario F, indices 60-74 (exact, verified -- the real transition is
at 60) are labelled 'regime', and through the real Orca pipeline the RMSE
drops from 9.4007 (default) to 6.0449 (use_arbiter=True) -- a real
improvement, no longer a tie. See test/test_arbiter_orca_integration.py
for the full verification on the 7 scenarios with this fix.
"""

from collections import deque

import numpy as np

from dense_armor._dtype import as_default_array, get_default_dtype


def _window(x: np.ndarray, i: int, radius: int) -> np.ndarray:
    n = len(x)
    lo, hi = max(0, i - radius), min(n, i + radius + 1)
    return x[lo:hi]


def _window_causal(x: np.ndarray, i: int, span: int) -> np.ndarray:
    """Only points BEFORE i (never i itself, never points after) -- used for
    the detection reference (not for the correction value, which stays
    on the symmetric window: once a point is already classified
    'spike', also using the future context to estimate the clean value
    is correct, the same non-causal principle already declared for
    pressure_valve). The reason for this causal window is detection only:
    see the KNOWN LIMIT (now resolved) below -- a symmetric window across
    a transition mixes the old and the new level in its own scale,
    diluting the deviation that should trigger the threshold."""
    lo = max(0, i - span)
    return x[lo:i]


def _robust_center_scale(w: np.ndarray) -> tuple[float, float]:
    med = float(np.median(w))
    mad = float(np.median(np.abs(w - med)))
    return med, 1.4826 * mad


def _detector_kind(detector) -> str | None:
    """'point' for a detector with ``score_one`` and ``learn_one`` (a score for every point), 'event' for one with
    ``update`` and ``drift_detected`` (an alarm when the level changes), None for anything else."""
    if detector is None:
        return None
    if hasattr(detector, "score_one") and hasattr(detector, "learn_one"):
        return "point"
    if hasattr(detector, "update") and hasattr(detector, "drift_detected"):
        return "event"
    return None


def _detector_threshold(detector, threshold, n_sigmas: float, eps: float) -> float:
    """The score above which a point deviates: the given one, else ``detector.threshold``, else ``n_sigmas``."""
    thr = threshold if threshold is not None else getattr(detector, "threshold", None)
    try:
        thr = float(thr)
    except (TypeError, ValueError):
        thr = float(n_sigmas)
    return thr if np.isfinite(thr) and thr > eps else float(n_sigmas)


def classify_segments(
    x: np.ndarray, radius: int = 10, ref_mult: int = 3,
    n_sigmas: float = 3.0, spike_run_max: int = 2, eps: float = 1e-9,
    detector=None, threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ``detector`` (default None, the window of this module) can be any detector of the library:
      - a point detector (it has ``score_one`` and ``learn_one``, like HampelScorer or LODA): its score
        replaces the 'is this point deviating?' test of the window; a point deviates when the score is
        above ``threshold`` (default: ``detector.threshold`` if it has one, else ``n_sigmas``). The run
        length rule that decides spike or regime stays the one of this module;
      - an event detector (it has ``update`` and ``drift_detected``, like ADWIN or PageHinkley): it replaces
        nothing; a run of deviating points during which it signals a change is labelled 'regime'.
    The detector is cloned (if it has ``clone``), so the object given is not changed. Any other object
    is not used.

    Returns (etichette, deviazione, incertezza):
      etichette   -- array of strings 'clean'/'spike'/'regime', one per point.
                     A non-finite point is 'clean' and is left out of the
                     reference windows of the following points, as in
                     streamDD.
      deviazione  -- |x[i] - reference| / scale (or only |x[i]-reference|
                     if the scale is degenerate), same unit for every point.
      incertezza  -- 0..1, how close the point is to the decision boundary
                     (deviating/clean threshold, or spike/regime threshold on
                     the length of the run) -- not how "correct" it is,
                     but how precarious the classification itself is.
    """
    x = as_default_array(x).ravel()
    n = x.size
    deviante = np.zeros(n, dtype=bool)
    deviazione = np.zeros(n, dtype=get_default_dtype())
    scarto_normalizzato = np.zeros(n, dtype=get_default_dtype())  # |dev|/threshold, for the uncertainty
    med_rif = np.zeros(n, dtype=get_default_dtype())  # median of the reference window, per point

    kind = _detector_kind(detector)
    det = detector.clone() if kind and hasattr(detector, "clone") else detector
    thr = _detector_threshold(detector, threshold, n_sigmas, eps) if kind == "point" else None
    events: list[int] = []
    finite_seen: deque = deque(maxlen=radius * ref_mult)
    for i in range(n):
        if not np.isfinite(x[i]):
            continue
        w_rif = np.array(finite_seen, dtype=x.dtype)
        finite_seen.append(x[i])
        score = None
        if kind == "point":
            score = float(det.score_one({"x": float(x[i])}))
            det.learn_one({"x": float(x[i])})
        elif kind == "event":
            det.update(float(x[i]))
            if det.drift_detected:
                events.append(i)
        if w_rif.size < 4:
            continue
        med, scala = _robust_center_scale(w_rif)
        med_rif[i] = med
        scarto = abs(x[i] - med)
        if score is not None:
            if np.isfinite(score):
                deviazione[i] = score
                deviante[i] = score > thr
                scarto_normalizzato[i] = score / thr
        elif scala < eps:
            # Degenerate baseline: no division, "different from constant"
            # is already the answer. scarto_normalizzato saturates at a clearly
            # high/low value (never near the boundary by construction),
            # so the uncertainty on these points stays low -- the
            # degenerate case is the ONLY one where the decision is truly unambiguous.
            deviazione[i] = scarto
            deviante[i] = scarto > eps
            scarto_normalizzato[i] = 0.0 if deviante[i] else 10.0
        else:
            deviazione[i] = scarto / scala
            deviante[i] = deviazione[i] > n_sigmas
            scarto_normalizzato[i] = deviazione[i] / n_sigmas

    etichette = np.full(n, "clean", dtype=object)
    incertezza = np.zeros(n, dtype=get_default_dtype())

    # Uncertainty on the deviating/clean threshold: near scarto_normalizzato==1
    # (i.e. exactly at the n_sigmas boundary) -> maximum ambiguity.
    incertezza_soglia = np.exp(-4.0 * (scarto_normalizzato - 1.0) ** 2)

    i = 0
    while i < n:
        if not deviante[i]:
            incertezza[i] = incertezza_soglia[i]
            i += 1
            continue
        j = i
        while j < n and deviante[j]:
            j += 1
        run_len = j - i
        if any(i <= e < j for e in events):
            label = "regime"
        elif run_len <= spike_run_max:
            label = "spike"
        else:
            # A long run is a real regime change only if its points
            # SETTLE on a new coherent level -- it is not enough that
            # they are "different from the old one", they must also be similar
            # TO EACH OTHER. Verified: 3 alternating impulses (+500,-500,+500,
            # scenario A) are a run of length 3 (> spike_run_max
            # default 2), but their INTERNAL dispersion is huge
            # (std~471) -- without this check they were mistaken for a
            # regime and passed raw, RMSE 86.6 instead of ~0. A run
            # that really settles (scenario F) has small internal dispersion
            # compared with the size of the jump itself.
            run_vals = x[i:j]
            run_spread = float(np.std(run_vals))
            run_jump = float(abs(np.median(run_vals) - med_rif[i]))
            coerente = run_spread < 0.5 * max(run_jump, eps)

            # Internal coherence is not enough: a temporary collapse that then
            # RETURNS to the old level (scenario C, 3 zeros then return to
            # 120) is internally as coherent as a true regime (internal
            # spread zero in both cases) -- the real difference is seen
            # ONLY by looking at what happens right after the run:
            # a true regime (scenario F) stays near the NEW level,
            # a collapse returns towards the OLD one. Without this
            # check scenario C stayed at RMSE 20.8 instead of ~0
            # (mistaken for a regime, passed raw instead of rejected).
            post_lo, post_hi = j, min(n, j + radius)
            post_vals = x[post_lo:post_hi]
            post_vals = post_vals[np.isfinite(post_vals)]
            if post_vals.size:
                post_med = float(np.median(post_vals))
                run_med = float(np.median(run_vals))
                persiste = abs(post_med - run_med) < abs(post_med - med_rif[i])
            else:
                persiste = True  # end of series: no "after" to check

            label = "regime" if (coerente and persiste) else "spike"
        etichette[i:j] = label
        # Uncertainty on the spike/regime threshold: near run_len==spike_run_max
        # -> maximum ambiguity on the length; combined (mean) with the one
        # on the deviating/clean threshold of the single points of the run.
        incertezza_run = float(np.exp(-1.0 * (run_len - spike_run_max) ** 2))
        incertezza[i:j] = 0.5 * incertezza_soglia[i:j] + 0.5 * incertezza_run
        i = j

    return etichette, deviazione, incertezza


def route_and_correct(
    x: np.ndarray, radius: int = 10, ref_mult: int = 3,
    n_sigmas: float = 3.0, spike_run_max: int = 2,
    detector=None, threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies the classification and corrects each point according to its label:
      'clean'/'regime' -> passes the raw value (a genuine regime change
                          must not be rejected, same philosophy as the
                          constant gate of Orca);
      'spike'          -> replaced with the median of the wide reference
                          window (robust, not contaminated by the single
                          short impulse).

    Returns (pulito, etichette, incertezza)."""
    x = as_default_array(x).ravel()
    n = x.size
    etichette, _, incertezza = classify_segments(
        x, radius, ref_mult, n_sigmas, spike_run_max, detector=detector, threshold=threshold
    )

    pulito = np.copy(x)
    for i in range(n):
        if etichette[i] == "spike":
            w_rif = _window(x, i, radius * ref_mult)
            w_rif = w_rif[np.isfinite(w_rif)]
            if w_rif.size:
                pulito[i] = float(np.median(w_rif))

    return pulito, etichette, incertezza
