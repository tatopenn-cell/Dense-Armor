"""Portfolio: how do I weigh an online portfolio when the covariance is noisy?

Two weightings are computed from the shrunk covariance of the returns of
four tickers: the minimum-variance weights and the inverse-volatility
weights. The minimum-variance weights solve

    min_w  w' S w   subject to  1' w = 1

and the Lagrange multiplier gives

    w = S^{-1} 1 / (1' S^{-1} 1).

The derivation is in the docstring of :meth:`min_variance_weights` and
is tested against a direct numerical minimisation.

The covariance is the one of :class:`OnlineShrinkageCovariance`
(:mod:`dense_armor.utility.stats.shrinkage`, Bodnar, Gupta and Parolya
2014, arXiv:1308.2608, equation 4.2 page 12), whose intensities are
computed from the running moments. The inverse-volatility weights
``w_i proportional to 1 / sigma_i`` are this implementation's choice:
they are not a paper formula, and the docstring says so.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dense_armor.utility.stats.shrinkage import OnlineShrinkageCovariance

_YAHOO_HEADER_ROWS = 3
TICKERS = ("AAPL", "GSPC", "MSFT", "TSLA")


def load_returns(path) -> np.ndarray:
    """Daily log returns from a Yahoo CSV (three header rows)."""
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if len(rows) <= _YAHOO_HEADER_ROWS:
        raise ValueError(f"{path}: fewer than {_YAHOO_HEADER_ROWS + 1} rows")
    header = rows[0]
    if "Close" not in header:
        raise ValueError(f"{path}: no Close column")
    j = header.index("Close")
    closes = []
    for row in rows[_YAHOO_HEADER_ROWS:]:
        if len(row) <= j:
            continue
        try:
            c = float(row[j])
        except ValueError:
            continue
        closes.append(c)
    arr = np.asarray(closes, dtype=float)
    if np.any(arr <= 0):
        raise ValueError(f"{path}: a close is not positive")
    return np.log(arr[1:] / arr[:-1])


def load_panel(fixture_dir) -> np.ndarray:
    """A ``n`` by ``p`` matrix of returns, one column per ticker.

    The four tickers of :data:`TICKERS` are read from the fixtures, and
    the panel is cut to the shortest series so the rows align.
    """
    d = Path(fixture_dir)
    cols = [load_returns(d / f"{t}.csv") for t in TICKERS]
    n = min(len(c) for c in cols)
    return np.column_stack([c[-n:] for c in cols])


def min_variance_weights(cov: np.ndarray) -> np.ndarray:
    """Minimum-variance weights from a covariance matrix.

    The problem is

        min_w  w' S w   subject to  1' w = 1

    The Lagrangian is ``L = w' S w - lambda (1' w - 1)``, and
    ``dL/dw = 2 S w - lambda 1 = 0`` gives ``w = (lambda/2) S^{-1} 1``.
    The constraint fixes ``lambda/2 = 1 / (1' S^{-1} 1)``, so

        w = S^{-1} 1 / (1' S^{-1} 1).

    Args:
        cov: a symmetric positive definite ``p`` by ``p`` matrix.

    Returns:
        The weights, summing to one.

    Raises:
        ValueError: if the matrix is not positive definite or the
            denominator is zero.
    """
    p = cov.shape[0]
    ones = np.ones(p)
    try:
        s_inv_ones = np.linalg.solve(cov, ones)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"covariance is singular: {exc}") from exc
    denom = float(np.dot(ones, s_inv_ones))
    return s_inv_ones / denom


def inverse_volatility_weights(cov: np.ndarray) -> np.ndarray:
    """Weights proportional to the inverse of the volatilities.

    ``w_i proportional to 1 / sqrt(S_ii)``, normalised to sum to one.
    The choice is this implementation's: the paper gives the
    minimum-variance solution, not this one.

    Raises:
        ValueError: if a variance is not positive.
    """
    var = np.diag(cov)
    if np.any(var <= 0):
        raise ValueError("a variance is not positive")
    inv = 1.0 / np.sqrt(var)
    return inv / inv.sum()


@dataclass
class PortfolioResult:
    """One run of the portfolio.

    Attributes:
        n_samples: number of returns fed to the covariance.
        weights_min_var: minimum-variance weights.
        weights_inv_vol: inverse-volatility weights.
        realised_vol_min_var: realised volatility of the minimum-variance
            portfolio on the sample.
        realised_vol_inv_vol: realised volatility of the
            inverse-volatility portfolio on the sample.
        realised_vol_equal: realised volatility of the equal-weight
            portfolio on the sample.
        turnover_min_var: sum of ``|w_t - w_{t-1}|`` over the rolling
            windows, for the minimum-variance weights.
        alpha: shrinkage intensity of the shrunk covariance.
        beta: shrinkage intensity on the target.
    """

    n_samples: int
    weights_min_var: np.ndarray
    weights_inv_vol: np.ndarray
    realised_vol_min_var: float
    realised_vol_inv_vol: float
    realised_vol_equal: float
    turnover_min_var: float
    alpha: float
    beta: float


class PortfolioPipeline:
    """Run the two weightings on a panel of returns.

    Args:
        window: number of returns in the rolling window used for the
            turnover. Default 100.
        refit_every: refit the weights every ``refit_every`` windows.
            Default 20.
    """

    def __init__(self, window: int = 100, refit_every: int = 20) -> None:
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        if refit_every < 1:
            raise ValueError(f"refit_every must be >= 1, got {refit_every}")
        self.window = int(window)
        self.refit_every = int(refit_every)

    def run(self, panel: np.ndarray) -> PortfolioResult:
        """Run the pipeline on a ``n`` by ``p`` panel of returns."""
        x = np.asarray(panel, dtype=float)
        n, p = x.shape
        if n < 2:
            raise ValueError(f"at least 2 rows, got {n}")
        online = OnlineShrinkageCovariance(p)
        for row in x:
            online.learn_one({f"v{i}": float(row[i]) for i in range(p)})
        shrunk, alpha, beta = online.shrinkage()
        w_mv = min_variance_weights(shrunk)
        w_iv = inverse_volatility_weights(shrunk)
        w_eq = np.full(p, 1.0 / p)
        port_mv = x @ w_mv
        port_iv = x @ w_iv
        port_eq = x @ w_eq
        vol_mv = float(np.std(port_mv))
        vol_iv = float(np.std(port_iv))
        vol_eq = float(np.std(port_eq))
        turnover = 0.0
        prev = None
        for start in range(0, n - self.window, self.refit_every):
            sub = x[start:start + self.window]
            c = np.cov(sub, rowvar=False, ddof=1)
            try:
                w = min_variance_weights(c)
            except ValueError:
                continue
            if prev is not None:
                turnover += float(np.abs(w - prev).sum())
            prev = w
        return PortfolioResult(
            n_samples=n,
            weights_min_var=w_mv,
            weights_inv_vol=w_iv,
            realised_vol_min_var=vol_mv,
            realised_vol_inv_vol=vol_iv,
            realised_vol_equal=vol_eq,
            turnover_min_var=turnover,
            alpha=alpha,
            beta=beta,
        )


def run_on_fixtures(fixture_dir) -> PortfolioResult:
    """Run the pipeline on the four fixtures in ``fixture_dir``."""
    panel = load_panel(fixture_dir)
    return PortfolioPipeline().run(panel)
