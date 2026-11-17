import numpy as np

from dense_armor._dtype import get_default_dtype


def healing_filter(x: np.ndarray, radius: int = 2, sustain_threshold: float = 0.7, wide_mult: int = 3) -> np.ndarray:
    """
    Signal-recovery filter inspired by healing.py (Dense-Evolution) --
    evolution brought to generic AI signals. Unlike ABCollatz and
    the Stage 1 damping (both judge a point by looking only at its
    own instantaneous residual), this classifies each point by looking at
    HOW MANY NEIGHBORS share the same deviation from the local baseline:
      - isolated deviation (no neighbor shares it) -> noise/spike ->
        replaced with the baseline (median of a wider window)
      - sustained deviation (the majority of neighbors in the
        narrow window share sign and amplitude) -> true change of the signal
        -> let through unchanged

    Parameters tuned by grid search (40 seeds, jump+spike scenario,
    see test/test_healing_filter.py): radius=2, sustain_threshold=0.7,
    wide_mult=3 -> 40/40 wins over moving median r2, mean/std
    ratio ~2.56 (not statistical noise).

    KNOWN LIMITS (measured, not hypothesized):
    - on a smooth signal + pure Gaussian noise (no true jump), it loses
      against a simple moving median at low/medium noise (0/30, 0/30);
      it wins only at high/very high noise. It is not the primary use case:
      this filter is for where there are true transitions to preserve, not
      for pure denoising of stationary signals.
    - denser spike variant (15% instead of 5%): wins 30/30 but with high
      variance (std=0.177 > mean=0.124) -- the advantage is consistent
      in sign but not uniform in amplitude, to be investigated if high
      spike densities will be used in production.
    - the window is centered, so the filter is not causal: it uses the
      points after the one it judges. A non-finite point is replaced by the
      baseline, and the baseline is the median of the finite points of the
      wide window.
    - O(n * wide) per call (Python loop, not vectorized/JIT) --
      before production it must be ported to JAX with jax.lax.scan or vmap over
      windows, like the rest of the codebase.
    - TEMPORARY collapse (a few consecutive points at an extreme value,
      then return to the original level) mistaken for a true change
      and let through: measured on a real scenario with 3
      consecutive points collapsed to zero on a constant background (then returned to the
      original level), RMSE 20.8 instead of ~0 -- see
      test/testKalman.py scenario C and
      test/test_arbiter_orca_integration.py in Dense-Armor for the same
      scenario. Cause: this function decides point by point (no
      grouping into runs), so it has no way to check what
      happens AFTER a sustained deviation to tell a
      temporary collapse from a regime that really settles -- the same
      check added to `core/online/protect/arbiter.py` (which does group into
      runs) solves this case there, but would require a change of
      design here, not yet done so as not to risk the calibration on
      140 seeds already verified above.
    """
    n = len(x)
    out = np.zeros(n, dtype=get_default_dtype())
    wide = radius * wide_mult
    for i in range(n):
        lo_wide, hi_wide = max(0, i - wide), min(n, i + wide + 1)
        wide_vals = x[lo_wide:hi_wide]
        wide_vals = wide_vals[np.isfinite(wide_vals)]
        baseline = np.median(wide_vals) if wide_vals.size else np.nan
        lo, hi = max(0, i - radius), min(n, i + radius + 1)
        window = x[lo:hi]
        dev_i = x[i] - baseline
        devs = window - baseline
        if abs(dev_i) < 1e-9:
            out[i] = x[i]
            continue
        same_sign_share = np.mean(
            (np.sign(devs) == np.sign(dev_i)) & (np.abs(devs) > sustain_threshold * abs(dev_i))
        )
        out[i] = x[i] if same_sign_share > 0.5 else baseline
    return out
