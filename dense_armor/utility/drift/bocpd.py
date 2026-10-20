"""Bayesian online changepoint detection (Adams and MacKay 2007).

A changepoint is a time at which the distribution of the stream
changes. The algorithm tracks, for every possible length of the
current run, the probability that the run has that length, and updates
the distribution one sample at a time. When the probability mass moves
to a run length of zero, a changepoint has just happened.

Reference: Adams, R. P. and MacKay, D. J. C. (2007). *Bayesian Online
Changepoint Detection.* arXiv:0710.3742. The hazard function is
equation 4, page 2; the conjugate-exponential predictive is section
2.3, pages 3-4; Algorithm 1 is on pages 3-4; the truncation of the run
length tail is section 2.4, page 4.

The implementation follows the paper in three points:

- the hazard is constant, ``H(tau) = 1 / lam`` (equation 4, page 2),
  which the paper calls "a discrete exponential prior over the interval
  between changepoints" (section 3, page 4);
- the model is Normal with a Normal-Inverse-Gamma prior, summarised by
  ``(mu, kappa, alpha, beta)`` and updated with the standard
  conjugate-exponential recursion (section 2.3, equations 8-11, pages
  3-4);
- the run length distribution is truncated at ``max_run_length``. The
  paper says "a trivial modification of the algorithm is to discard the
  run length probability estimates in the tail of the distribution which
  have a total mass less than some threshold, say 10^{-4}" (section
  2.4, page 4). This implementation truncates by length, not by mass,
  and the choice of the length is this implementation's: it bounds the
  memory at ``O(max_run_length)`` per step.

The predictive probability ``P(x_t | run)`` is the Student-t density
that the Normal-Inverse-Gamma prior gives (the closed form of equation
11, page 3, which the paper writes as an integral). The closed form is
not written explicitly in the paper: it is derived in the docstring of
:meth:`_predictive` and pinned by a test against the recursion without
truncation.

Example:
    >>> import numpy as np
    >>> from dense_armor.utility.drift.bocpd import BayesianOnlineChangepoint
    >>> det = BayesianOnlineChangepoint(lam=100.0)
    >>> rng = np.random.default_rng(0)
    >>> stream = list(rng.normal(0.0, 0.1, 200)) + list(rng.normal(1.0, 0.1, 200))
    >>> fired = []
    >>> for i, x in enumerate(stream):
    ...     _ = det.update(x)
    ...     if det.drift_detected:
    ...         fired.append(i)
    >>> len(fired) >= 1
    True
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from dense_armor.roles import DriftDetector


class BayesianOnlineChangepoint(DriftDetector):
    """Online changepoint detection with a Normal-Inverse-Gamma model.

    The stream is modelled as Normal with unknown mean and variance,
    with a Normal-Inverse-Gamma prior (Adams and MacKay 2007, section
    2.3, page 3). For every possible run length ``r`` the posterior
    parameters ``(mu_r, kappa_r, alpha_r, beta_r)`` are updated with the
    conjugate recursion, and the run length distribution is propagated
    with the growth and changepoint probabilities of Algorithm 1 (pages
    3-4). A changepoint fires when the probability mass on the zero run
    length exceeds ``detection_threshold`` on a step where it did not on
    the previous step.

    Args:
        lam: mean of the discrete exponential prior on the interval
            between changepoints, ``H(tau) = 1 / lam`` (equation 4,
            page 2). Default 250.0, the value the paper uses on the
            well-log data (section 3.1, page 4). It is this
            implementation's default; the paper does not fix one.
        mu0: prior mean of the Normal. Default 0.0.
        kappa0: prior number of observations for the mean. Default 0.01.
        alpha0: prior shape of the Inverse-Gamma on the variance.
            Default 0.01.
        beta0: prior scale of the Inverse-Gamma on the variance.
            Default 0.01.
        max_run_length: hard cap on the number of run lengths kept, a
            safety bound on memory. The primary truncation is by mass
            (see ``mass_threshold``). Default 10000, chosen above the
            expected run length ``lam`` so the cap does not cut the
            mode. It is this implementation's choice.
        mass_threshold: the run lengths in the tail whose cumulative
            mass is below this value are discarded, as section 2.4,
            page 4 of the paper suggests (``10^-4`` is the paper's
            example). Default 1e-4.
        mass_window: number of shortest run lengths whose cumulative
            posterior mass is compared to ``detection_threshold``.
            The changepoint signal is the mass that the posterior
            places on short runs, not the exact mass on run length
            zero: with a constant hazard, ``P(r_t = 0 | x_{1:t})``
            is about ``1 / lam`` on every step, before and after a
            changepoint, because the growth term moves the mass to
            run length one (see Algorithm 1, page 3). Default 10.
            It is this implementation's choice: the paper reports
            the run length posterior, not a threshold on it.
        detection_threshold: cumulative mass on the first
            ``mass_window`` run lengths above which
            ``drift_detected`` becomes True for one step. Default
            0.5. It is this implementation's choice.

    Raises:
        ValueError: on a non-positive ``lam``, ``max_run_length``,
            ``kappa0``, ``alpha0`` or ``beta0``, or a
            ``detection_threshold`` outside ``(0, 1)``.
    """

    budget_s = 7.13e-03
    memory_class = "O(max_run_length)"

    def __init__(
        self,
        lam: float = 250.0,
        mu0: float = 0.0,
        kappa0: float = 0.01,
        alpha0: float = 0.01,
        beta0: float = 0.01,
        max_run_length: int = 10000,
        mass_threshold: float = 1e-4,
        mass_window: int = 10,
        detection_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        if lam <= 0:
            raise ValueError(f"lam must be > 0, got {lam}")
        if max_run_length < 1:
            raise ValueError(f"max_run_length must be >= 1, got {max_run_length}")
        if not (0.0 <= mass_threshold < 1.0):
            raise ValueError(
                f"mass_threshold must be in [0, 1), got {mass_threshold}")
        if mass_window < 1:
            raise ValueError(f"mass_window must be >= 1, got {mass_window}")
        if kappa0 <= 0 or alpha0 <= 0 or beta0 <= 0:
            raise ValueError("kappa0, alpha0 and beta0 must be > 0")
        if not (0.0 < detection_threshold < 1.0):
            raise ValueError(
                f"detection_threshold must be in (0, 1), got {detection_threshold}")
        self.lam = float(lam)
        self.mu0 = float(mu0)
        self.kappa0 = float(kappa0)
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)
        self.max_run_length = int(max_run_length)
        self.mass_threshold = float(mass_threshold)
        self.mass_window = int(mass_window)
        self.detection_threshold = float(detection_threshold)
        self._reset()

    def _reset(self) -> None:
        super()._reset()
        self.n_alarms_: int = 0
        self.n_seen_: int = 0
        self.run_length_: np.ndarray = np.zeros(1, dtype=float)
        self.run_length_[0] = 1.0
        self.mu_: np.ndarray = np.full(1, self.mu0, dtype=float)
        self.kappa_: np.ndarray = np.full(1, self.kappa0, dtype=float)
        self.alpha_: np.ndarray = np.full(1, self.alpha0, dtype=float)
        self.beta_: np.ndarray = np.full(1, self.beta0, dtype=float)
        self.map_run_length_: int = 0
        self.changepoint_mass_: float = 1.0
        self.short_run_mass_: float = 1.0
        self._was_alarm: bool = True

    def _unit_test_skips(self) -> set:
        return set()

    def _predictive(
        self, x: float,
        mu: np.ndarray, kappa: np.ndarray,
        alpha: np.ndarray, beta: np.ndarray,
    ) -> np.ndarray:
        """P(x_t | run length r) for every r, the Student-t of the NIG.

        With a Normal-Inverse-Gamma prior, the marginal of a new sample
        is a Student-t (the closed form of equation 11, page 3, which
        the paper writes as an integral):

            P(x | r) = t_{2 alpha_r}(x | mu_r, beta_r (kappa_r + 1) /
                                               (alpha_r kappa_r))

        where ``t_nu(x | m, s^2)`` is the Student-t density with ``nu``
        degrees of freedom, location ``m`` and scale ``s``.
        """
        nu = 2.0 * alpha
        scale2 = beta * (kappa + 1.0) / (alpha * kappa)
        scale = np.sqrt(scale2)
        return stats.t.pdf(x, df=nu, loc=mu, scale=scale)

    def _update_nig(
        self, x: float,
        mu: np.ndarray, kappa: np.ndarray,
        alpha: np.ndarray, beta: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Conjugate update of the NIG posterior with one sample."""
        kappa_new = kappa + 1.0
        mu_new = (kappa * mu + x) / kappa_new
        alpha_new = alpha + 0.5
        beta_new = beta + 0.5 * kappa * (x - mu) ** 2 / kappa_new
        return mu_new, kappa_new, alpha_new, beta_new

    def update(self, x: float, t: float | None = None) -> BayesianOnlineChangepoint:
        """Update the run length distribution with one sample."""
        self._drift_detected = False
        if not np.isfinite(x):
            self.n_missing_ = getattr(self, "n_missing_", 0) + 1
            return self
        x = float(x)
        self.n_seen_ += 1

        pi = self._predictive(x, self.mu_, self.kappa_, self.alpha_, self.beta_)
        pi = np.maximum(pi, 1e-300)
        hazard = 1.0 / self.lam

        growth = self.run_length_ * pi * (1.0 - hazard)
        cp_mass = float(np.sum(self.run_length_ * pi * hazard))

        new_r = np.empty(len(growth) + 1, dtype=float)
        new_r[0] = cp_mass
        new_r[1:] = growth

        mu_n, kappa_n, alpha_n, beta_n = self._update_nig(
            x, self.mu_, self.kappa_, self.alpha_, self.beta_)

        new_mu = np.empty(len(growth) + 1, dtype=float)
        new_kappa = np.empty(len(growth) + 1, dtype=float)
        new_alpha = np.empty(len(growth) + 1, dtype=float)
        new_beta = np.empty(len(growth) + 1, dtype=float)
        new_mu[0] = self.mu0
        new_kappa[0] = self.kappa0
        new_alpha[0] = self.alpha0
        new_beta[0] = self.beta0
        new_mu[1:] = mu_n
        new_kappa[1:] = kappa_n
        new_alpha[1:] = alpha_n
        new_beta[1:] = beta_n

        cut = len(new_r)
        tail = 0.0
        while cut > 1 and tail + new_r[cut - 1] < self.mass_threshold:
            tail += new_r[cut - 1]
            cut -= 1
        cut = min(cut, self.max_run_length + 1)
        if cut < len(new_r):
            new_r = new_r[:cut]
            new_mu = new_mu[:cut]
            new_kappa = new_kappa[:cut]
            new_alpha = new_alpha[:cut]
            new_beta = new_beta[:cut]

        total = float(new_r.sum())
        if total > 0:
            new_r /= total

        self.run_length_ = new_r
        self.mu_ = new_mu
        self.kappa_ = new_kappa
        self.alpha_ = new_alpha
        self.beta_ = new_beta
        self.map_run_length_ = int(np.argmax(self.run_length_))
        self.changepoint_mass_ = float(self.run_length_[0])

        k = min(self.mass_window, len(self.run_length_))
        short_mass = float(np.sum(self.run_length_[:k]))
        self.short_run_mass_ = short_mass
        is_alarm = short_mass >= self.detection_threshold
        if is_alarm and not self._was_alarm:
            self._drift_detected = True
            self.n_alarms_ += 1
        self._was_alarm = is_alarm
        return self
