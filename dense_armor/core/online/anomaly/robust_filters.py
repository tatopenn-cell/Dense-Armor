"""
robust_filters.py -- four "low-cost" anomaly detectors, no heavy
machinery (no dynamic model, no state, no automatic
differentiation): classical statistics, known and used for decades
in telemetry/astronomy, each computed on a centered local window
(like healing_filter.py in this same package -- non-causal, designed
for offline/batch cleaning of an already recorded series, not for the
real-time loop of utility/hybrid_engine.py).

Every function has the same signature (x, radius=..., **threshold) -> (pulito, anomalie):
    pulito   -- 1D array, same shape as x: raw value where not anomalous,
                local median where anomalous.
    anomalie -- list of indices (int) flagged as anomalous.

The four methods, in order of historical introduction:
- Chauvenet (1863): rejection threshold based on the expected probability that a
  sample of the local window (size N) deviates so much by pure chance --
  if the expected number of such events (N * P) is < 0.5, the point is a
  statistically "too improbable for the available sample" outlier.
- Tukey's fences / IQR (1970s): no assumption on the distribution,
  only quartiles -- beyond Q1-1.5*IQR or Q3+1.5*IQR it is an outlier.
- Hampel filter: median + scaled MAD (1.4826*MAD, consistency constant
  for a Gaussian) as the threshold -- the robust generalization of "3 sigma"
  that never uses a mean/std sensitive to the outliers it is looking for.
- Iterative sigma-clipping: standard in astronomy (image stacking,
  photometry) -- recomputes mean/std of the window excluding the points beyond
  N sigma, repeats until stable or max_iters, then judges the central point
  against the "cleaned" statistics.
"""
import math

import numpy as np

from dense_armor._dtype import as_default_array, get_default_dtype


def _window(x: np.ndarray, i: int, radius: int) -> np.ndarray:
    n = len(x)
    lo, hi = max(0, i - radius), min(n, i + radius + 1)
    return x[lo:hi]


def _jensen_shannon(a: np.ndarray, b: np.ndarray, n_bins: int = None, pseudo: float = 1.0) -> float:
    """Jensen-Shannon divergence (in bits, log base 2) between the empirical
    distributions of `a` and `b` -- same bins for both (histogram on the union
    of the two ranges). 0.0 = identical distributions, 1.0 = completely
    disjoint supports. Pure numpy arithmetic (no dependency on
    scipy), same "light" philosophy as the other detectors in this module.

    Two precautions for small windows (verified: without them, the JSD
    came out HIGHER on pure stationary Gaussian noise than near
    a real level transition -- the signal in the wrong direction,
    a pure artifact of few samples per bin):
    - adaptive n_bins (default None -> max(3, min(10, (len(a)+len(b))//6)),
      about 6 samples per bin instead of a fixed number independent of the
      window size;
    - true Laplace smoothing (pseudo=1.0, one pseudo-count per bin,
      not a near-zero epsilon just to avoid log(0)) -- with few
      samples two sparse histograms already look "different" by pure
      sampling noise; the smoothing brings them closer until a real
      difference in the distribution emerges anyway above
      it. Verified on stationary noise vs a true step 1.0->5.0
      (radius=10): mean stationary JSD 0.018-0.025, mean JSD near
      the transition 0.04-0.08 (2-3x higher, correct direction)."""
    lo = min(float(a.min()), float(b.min()))
    hi = max(float(a.max()), float(b.max()))
    if hi - lo < 1e-12:
        return 0.0
    if n_bins is None:
        n_bins = max(3, min(10, (a.size + b.size) // 6))
    edges = np.linspace(lo, hi, n_bins + 1, dtype=get_default_dtype())
    p, _ = np.histogram(a, bins=edges)
    q, _ = np.histogram(b, bins=edges)
    p = p.astype(get_default_dtype()) + pseudo
    q = q.astype(get_default_dtype()) + pseudo
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log2(p / m)))
    kl_qm = float(np.sum(q * np.log2(q / m)))
    return 0.5 * kl_pm + 0.5 * kl_qm


def chauvenet_criterion(x: np.ndarray, radius: int = 10) -> tuple[np.ndarray, list[int]]:
    """Chauvenet's criterion (1863) on a local window of radius `radius`.

    For point i, with the window as the sample (mean mu, std sigma, size
    N): d = |x[i] - mu| / sigma; P = erfc(d / sqrt(2)) (two-tailed probability
    of such a deviation under a Gaussian hypothesis). If N*P < 0.5 -- the expected
    number of deviations equal to this one, repeating the experiment, is less
    than half a sample -- the point is rejected as an outlier.

    Note: it uses mean/std (not robust), as in the original formulation --
    on a window with MORE than one outlier, mean/std are already distorted by the
    other outliers before even judging point i (known and
    documented limit of the original criterion, not specific to this implementation;
    for windows with multiple outliers prefer hampel_filter or tukey_fences,
    which use robust statistics).

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> x = rng.normal(0, 1, 60)
        >>> x[30] = 50.0
        >>> clean, idx = chauvenet_criterion(x, radius=5)
        >>> 30 in idx
        True
        >>> abs(float(clean[30])) < 5.0
        True
"""
    x = as_default_array(x)
    n = len(x)
    out = np.copy(x)
    anomalie = []
    for i in range(n):
        w = _window(x, i, radius)
        if w.size < 3:
            continue
        mu, sigma = float(np.mean(w)), float(np.std(w))
        if sigma < 1e-12:
            continue
        d = abs(x[i] - mu) / sigma
        p = math.erfc(d / math.sqrt(2))
        if w.size * p < 0.5:
            anomalie.append(i)
            out[i] = float(np.median(w))
    return out, anomalie


def tukey_fences(x: np.ndarray, radius: int = 10, k: float = 1.5) -> tuple[np.ndarray, list[int]]:
    """Tukey's fences / IQR on a local window of radius `radius`.

    Q1/Q3 = 25th/75th percentile of the window, IQR = Q3-Q1. Point i is an
    outlier if outside [Q1 - k*IQR, Q3 + k*IQR] (k=1.5 is the standard, k=3.0
    the "extreme outliers" variant). No assumption on the distribution --
    the only one of the four that does not presuppose normality.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> x = rng.normal(0, 1, 60)
        >>> x[30] = 50.0
        >>> clean, idx = tukey_fences(x, radius=5)
        >>> 30 in idx
        True
        >>> abs(float(clean[30])) < 5.0
        True
"""
    x = as_default_array(x)
    n = len(x)
    out = np.copy(x)
    anomalie = []
    for i in range(n):
        w = _window(x, i, radius)
        if w.size < 4:
            continue
        q1, q3 = np.percentile(w, [25, 75])
        iqr = q3 - q1
        lo_fence, hi_fence = q1 - k * iqr, q3 + k * iqr
        if x[i] < lo_fence or x[i] > hi_fence:
            anomalie.append(i)
            out[i] = float(np.median(w))
    return out, anomalie


def hampel_filter(x: np.ndarray, radius: int = 10, n_sigmas: float = 3.0) -> tuple[np.ndarray, list[int]]:
    """Hampel filter on a local window of radius `radius`.

    median and MAD (Median Absolute Deviation) of the window, scaled by
    1.4826 (consistency constant: makes scaled_MAD an unbiased
    estimator of the standard deviation IF the data were Gaussian,
    but without ever computing a true mean/std -- every single statistic
    used here is already robust, unlike Chauvenet). Outlier if
    |x[i] - median| > n_sigmas * scaled_MAD. When the scaled MAD is below
    1e-12 (a flat window) any point farther than 1e-9 from the median is an
    outlier.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> x = rng.normal(0, 1, 60)
        >>> x[30] = 50.0
        >>> clean, idx = hampel_filter(x, radius=5)
        >>> 30 in idx
        True
        >>> abs(float(clean[30])) < 5.0
        True
"""
    x = as_default_array(x)
    n = len(x)
    out = np.copy(x)
    anomalie = []
    for i in range(n):
        w = _window(x, i, radius)
        if w.size < 3:
            continue
        med = float(np.median(w))
        mad = float(np.median(np.abs(w - med)))
        scaled_mad = 1.4826 * mad
        if scaled_mad < 1e-12:
            if abs(x[i] - med) > 1e-9:
                anomalie.append(i)
                out[i] = med
            continue
        if abs(x[i] - med) > n_sigmas * scaled_mad:
            anomalie.append(i)
            out[i] = med
    return out, anomalie


def sigma_clip(x: np.ndarray, radius: int = 10, n_sigmas: float = 3.0, max_iters: int = 5) -> tuple[np.ndarray, list[int]]:
    """Iterative sigma-clipping (standard in astronomy: stacking, photometry)
    on a local window of radius `radius`.

    On the window (excluding point i itself, so a true outlier at the
    center cannot protect itself by inflating its own std): computes
    mean/std, discards the points beyond n_sigmas, recomputes on the remaining ones,
    repeats until stable or max_iters. Point i is an outlier if outside
    n_sigmas of the FINAL mean/std (cleaned of the other outliers of the
    window, unlike chauvenet_criterion which uses raw mean/std).

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> x = rng.normal(0, 1, 60)
        >>> x[30] = 50.0
        >>> clean, idx = sigma_clip(x, radius=5)
        >>> 30 in idx
        True
        >>> abs(float(clean[30])) < 5.0
        True
"""
    x = as_default_array(x)
    n = len(x)
    out = np.copy(x)
    anomalie = []
    for i in range(n):
        lo, hi = max(0, i - radius), min(n, i + radius + 1)
        neighbor_idx = [j for j in range(lo, hi) if j != i]
        if len(neighbor_idx) < 3:
            continue
        w = x[neighbor_idx].astype(get_default_dtype())
        mask = np.ones(w.shape, dtype=bool)
        for _ in range(max_iters):
            subset = w[mask]
            if subset.size < 2:
                break
            mu, sigma = float(np.mean(subset)), float(np.std(subset))
            if sigma < 1e-12:
                break
            new_mask = np.abs(w - mu) <= n_sigmas * sigma
            if np.array_equal(new_mask, mask):
                break
            mask = new_mask
        subset = w[mask]
        if subset.size < 1:
            continue
        mu, sigma = float(np.mean(subset)), float(np.std(subset))
        if sigma < 1e-12:
            if abs(x[i] - mu) > 1e-9:
                anomalie.append(i)
                out[i] = float(np.median(subset))
            continue
        if abs(x[i] - mu) > n_sigmas * sigma:
            anomalie.append(i)
            out[i] = float(np.median(subset))
    return out, anomalie


def pressure_valve(
    x: np.ndarray, radius: int = 10, soglia_pressione: float = 8.0, n_sigmas: float = 3.0,
    ref_mult: int = 3, k_molla: float = 3.0,
) -> tuple[np.ndarray, list[int], np.ndarray, np.ndarray]:
    """Automatic orchestrator of the four detectors above: not a vote
    (how many methods flag the point), but a CONSTRAINED MINIMUM-VARIANCE
    combination -- the classic BLUE estimator (Best Linear Unbiased
    Estimator) of statistics, derived with a Lagrange multiplier.

    Each method produces a local (center, scale) pair: Chauvenet uses
    mean/std (not robust, the original formulation -- the weakness is
    intended, see below), sigma-clipping uses mean/std iteratively cleaned
    of the outliers of the window, Hampel uses median/scaled MAD
    (1.4826*MAD), Tukey uses median/scaled IQR (IQR/1.349, the
    same constant that makes IQR an estimator of sigma for a Gaussian).
    Four independent estimates of the same "true" local center, each with
    its own uncertainty (scale).

    Combining them not with a simple mean but by looking for the weights w_k that
    minimize the VARIANCE of the weighted combination Sum(w_k * centro_k),
    constrained to Sum(w_k) = 1 (unbiased combination) -- a
    constrained optimization problem, solved with a Lagrange multiplier
    lambda:
        L(w, lambda) = Sum(w_k^2 * scala_k^2) - lambda * (Sum(w_k) - 1)
        dL/dw_k = 0  =>  w_k = lambda / (2 * scala_k^2)
        from the constraint  =>  w_k = (1/scala_k^2) / Sum_j(1/scala_j^2)
    The result (weight inversely proportional to the SQUARE of its own
    uncertainty) is the minimum-variance estimator among all unbiased
    linear combinations of independent estimates (Gauss-Markov) -- it is not a
    hand-picked weight, it is what the constraint imposes. A method whose
    scale inflates (e.g. Chauvenet when the window already contains an
    outlier, its std is not robust) is automatically weighted less,
    without needing to discard it explicitly.

    Caveat: the optimality of these weights holds for INDEPENDENT unbiased
    estimates whose scales are their standard errors. Here the four
    estimates are computed on the same window (they are strongly correlated)
    and each scala_k is the dispersion of the data, not the standard error of
    the center. The combination is therefore a calibrated heuristic, not a
    BLUE: scala_combinata is below every scala_k by construction, and
    soglia_pressione=8.0 is an empirical value.

    centro_combinato = Sum(w_k * centro_k)
    scala_combinata  = 1 / sqrt(Sum(1/scala_k^2))   (the same derivation
                        also gives the variance of the combination itself --
                        ALWAYS narrower than any single scala_k,
                        it is the property that makes combining several
                        independent estimates useful)
    pressione        = |x[i] - centro_combinato| / scala_combinata

    A single final threshold (soglia_pressione) decides -- binary trigger,
    like the rest of this ecosystem (utility/hybrid_engine.py,
    dense_evolution/healing.py): never a third intermediate state, never a
    blend between raw and clean. It is no longer a value "in sigma" in the
    classical sense (scala_combinata is narrower than every scala_k, so the
    same real deviation produces a higher pressure than with a
    single method) -- soglia_pressione=8.0 is the default calibrated
    directly (see test/test_robust_filters.py): on pure Gaussian noise
    (N=300, fixed seed) the percentiles of the pressure are
    50%=1.4, 95%=4.6, 99%=6.0, 99.9%=7.6 -- 8.0 leaves 1/300 false
    positives while still keeping a true outlier (pressure~88) and a
    pair of close outliers (pressure~40) well above the threshold with a
    wide margin. `n_sigmas` remains the parameter of the internal
    sigma-clipping sub-step (how many sigma beyond which a neighbor is excluded
    in the iteration, not the final threshold).

    THE SPRING (JSD): the threshold itself is not fixed. At each point the local
    window (radius `radius`, the same used by the 4 methods) is compared with
    a wider reference window (radius `radius * ref_mult`,
    default ref_mult=3 -- the same constant as healing_filter.py in this
    package) via the Jensen-Shannon divergence (_jensen_shannon, 0..1).
    If the two distributions are similar (stationary noise, no real
    transition in progress), JSD~0 and the threshold stays soglia_pressione. If
    they are very different (a true regime transition is in progress: the
    local window, closer to the new level, no longer resembles the
    reference window, still dominated by the old one), JSD rises and the
    threshold widens proportionally:
        soglia_effettiva[i] = soglia_pressione * (1 + k_molla * JSD[i])
    "the spring yields" -- the shield becomes less nervous exactly at the points where
    a genuine change is more plausible, without ever ceasing to
    judge in binary (anomaly above soglia_effettiva, nothing else).

    _jensen_shannon already uses adaptive n_bins and Laplace smoothing (see its
    docstring) precisely for this use: on small windows, a "naive" JSD
    (fixed bins, near-zero epsilon) came out HIGHER on pure stationary
    noise than near a true transition -- the spring would have
    activated everywhere, not "only where needed". With the corrected version,
    verified on low-amplitude stationary noise vs a true step
    1.0->5.0 (radius=10, k_molla=3.0): mean soglia_effettiva ~8.3 on
    stationary noise, ~8.7-8.8 near the transition (higher,
    right direction) -- still a noisy signal by nature (it never
    perfectly separates the two tails), but the threshold can only WIDEN
    (JSD >= 0), never narrow below soglia_pressione: even in the
    worst case (spring that activates "unnecessarily" on noise), the result is
    a slightly more permissive shield, never more nervous.

    No configuration required from the caller beyond the defaults: designed
    for a context where whoever reads the result must not choose weights or
    thresholds per method, only read `anomalie`/`pressione`. Verified
    (test/test_robust_filters.py): few false positives on pure Gaussian
    noise, detects a true outlier with pressure well above the threshold,
    does not touch a genuine sustained step.

    KNOWN LIMIT, NOW RESOLVED -- mutual masking between close
    outliers on a nearly flat baseline (test/test_pressure_valve_vs_
    orca_experiment.py, scenario "alternating impulses" +/-500 on a constant
    120 background, zero real noise): the local window used for each
    estimate already EXCLUDES point i itself (fix applied below), but if
    other outliers of the same burst remain in the window (here:
    radius 10, 3 peaks within 2 steps of each other), median/IQR and
    median/MAD still stay EXACTLY 0 -- it is not a bug in their
    implementation, it is mathematically correct: with 18/20 points of the
    window identical, MAD and IQR (breakdown ~50%/~25%) see nothing
    anomalous. A Huber M-estimator (IRLS, see Sung et al. 2019,
    arXiv:1912.04982, used there for 2-qubit spectroscopy with experimental
    outliers) was tried as an alternative and ALSO converges to
    scale 0, for the same underlying reason (its final scale is itself a
    MAD of the residuals). Having read the paper directly (not assumed
    from the formula): their robust technique works because the scale
    sigma^2=var(O_tau) is estimated from REPEATED MEASUREMENTS on the SAME point
    (shot noise, M projective repetitions), a data axis orthogonal
    to the time window -- not available here (a single
    scalar series, no repetition per point), so their solution is
    not portable as it is.

    A fallback scale from the wider reference window (radius*ref_mult)
    was also tried: VERIFIED NOT TO WORK, and for a structural
    reason, not an implementation error -- widening the window
    always reduces the FRACTION of outliers over the total (3 points out of 21
    local = 14%, 3 out of 61 reference = 5%), so it can never make
    IQR/MAD (estimators with a fixed fractional breakdown) exceed the breakdown
    threshold that makes them collapse to zero: verified numerically,
    both scales stayed 0.0 even on the reference window.

    THE REAL CORRECTION: when IQR or MAD truly collapse to zero
    (genuinely homogeneous local window) but point i still departs
    from the local median, that is NOT "no information" to
    discard -- it is the strongest proof possible (real deviation against
    null local dispersion). The bug was discarding this evidence together with
    the degenerate estimator. Now: null-local-scale + real deviation =>
    pressure forced to infinity (certain anomaly), bypassing the
    weighted combination for that point. Verified: on the real scenario
    "alternating impulses" (+/-500 on a 120 background without noise), before the fix
    pressure at the 3 peaks = 3.16/5.19/3.16 (below threshold 8.0, zero
    anomalies, RMSE=86.6); after the fix, all 3 peaks detected,
    RMSE=0.0. No regression: 223/223 tests of the suite pass,
    including the false-positive rate on pure Gaussian noise (unchanged)
    and all the other 6 scenarios of the three-way comparison.

    Returns (pulito, anomalie, pressione, soglia_effettiva): `pressione` is
    the continuous combined deviation for each point, `soglia_effettiva` the
    threshold actually applied at that point (>= soglia_pressione, it shows
    where/how much the spring yielded).

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> x = rng.normal(0, 1, 60)
        >>> x[30] = 50.0
        >>> clean, idx, p, thr = pressure_valve(x, radius=5)
        >>> 30 in idx
        True
        >>> abs(float(clean[30])) < 5.0
        True
"""
    x = as_default_array(x)
    n = len(x)
    pressione = np.zeros(n, dtype=get_default_dtype())
    soglia_effettiva = np.full(n, soglia_pressione)

    for i in range(n):
        w_self = _window(x, i, radius)
        if w_self.size < 4:
            continue
        # Point i EXCLUDED from its own window (as only the sigma-clipping
        # sub-step below already did): including it allowed an extreme
        # outlier to inflate its own reference scale
        # (masking) -- verified on isolated impulses
        # (+/-500 on a constant 120 background): with the self-inclusive window, maximum
        # pressure 4.6 (below any threshold, zero anomalies detected);
        # with the exclusive window, the outlier no longer contaminates the scale
        # it is judged with.
        w = np.delete(w_self, min(i - max(0, i - radius), w_self.size - 1))
        w_riferimento = _window(x, i, radius * ref_mult)

        centri, scale = [], []

        mu, sigma = float(np.mean(w)), float(np.std(w))
        if sigma > 1e-12:
            centri.append(mu)
            scale.append(sigma)

        med = float(np.median(w))
        q1, q3 = np.percentile(w, [25, 75])
        iqr_scale = (q3 - q1) / 1.349
        if iqr_scale > 1e-12:
            centri.append(med)
            scale.append(iqr_scale)

        mad = float(np.median(np.abs(w - med)))
        scaled_mad = 1.4826 * mad
        if scaled_mad > 1e-12:
            centri.append(med)
            scale.append(scaled_mad)

        # Zero-scale-is-certain-evidence: if IQR or MAD truly collapse to
        # zero (genuinely homogeneous local window, >=25%/>=50% of the
        # points identical) BUT point i still departs from the center, it is not
        # "no information" to discard -- it is the strongest proof
        # possible (real deviation against null local dispersion).
        # The bug of the original "KNOWN LIMIT" was exactly discarding this
        # evidence together with the degenerate estimator. Not attempted: a
        # fallback scale from the wider reference window (tried and
        # verified NOT to work -- see below: widening the window
        # always reduces the fraction of outliers, it can never revive
        # a statistic with a fixed fractional breakdown point). Verified on the
        # real "alternating impulses" scenario (test/test_pressure_valve_vs_
        # orca_experiment.py, +/-500 on a 120 background without noise): before
        # this fix, pressure at the 3 peaks = 3.16/5.19/3.16 (below threshold
        # 8.0, zero anomalies, RMSE=86.6); with this fix, all 3
        # peaks detected, RMSE=0.0.
        scala_locale_nulla = (iqr_scale <= 1e-12) or (scaled_mad <= 1e-12)
        if scala_locale_nulla and abs(x[i] - med) > 1e-9:
            pressione[i] = float("inf")
            continue

        neighbor_idx = [j for j in range(max(0, i - radius), min(n, i + radius + 1)) if j != i]
        if len(neighbor_idx) >= 3:
            w_neighbors = x[neighbor_idx].astype(get_default_dtype())
            mask = np.ones(w_neighbors.shape, dtype=bool)
            for _ in range(5):
                subset = w_neighbors[mask]
                if subset.size < 2:
                    break
                mu_s, sigma_s = float(np.mean(subset)), float(np.std(subset))
                if sigma_s < 1e-12:
                    break
                new_mask = np.abs(w_neighbors - mu_s) <= n_sigmas * sigma_s
                if np.array_equal(new_mask, mask):
                    break
                mask = new_mask
            subset = w_neighbors[mask]
            if subset.size >= 1:
                mu_s, sigma_s = float(np.mean(subset)), float(np.std(subset))
                if sigma_s > 1e-12:
                    centri.append(mu_s)
                    scale.append(sigma_s)

        if not centri:
            continue

        centri_arr = np.array(centri)
        scale_arr = np.array(scale)
        pesi = (1.0 / scale_arr ** 2)
        pesi /= pesi.sum()  # w_k = (1/scala_k^2) / Sum_j(1/scala_j^2), from the Lagrange multiplier

        centro_combinato = float(np.sum(pesi * centri_arr))
        scala_combinata = float(1.0 / np.sqrt(np.sum(1.0 / scale_arr ** 2)))

        if scala_combinata > 1e-12:
            pressione[i] = abs(x[i] - centro_combinato) / scala_combinata

        if w_riferimento.size >= 4 and w.size >= 4:
            jsd = _jensen_shannon(w, w_riferimento)
            soglia_effettiva[i] = soglia_pressione * (1.0 + k_molla * jsd)

    out = np.copy(x).astype(get_default_dtype())
    anomalie = []
    for i in range(n):
        if pressione[i] > soglia_effettiva[i]:
            anomalie.append(i)
            out[i] = float(np.median(_window(x, i, radius)))
    return out, anomalie, pressione, soglia_effettiva
