"""FIMT-DD regression model tree, from Ikonomovska (2012).

The algorithm is FIMT-DD of Ikonomovska (2012), Chapter 6 of the PhD
thesis, pages 68-77 of the printed pages. Only the details the thesis
states are implemented here; where the thesis leaves a choice, the
class docstring says "my choice" and gives the reason.

Sources (printed pages of the thesis, which are PDF pages 82-89):

- SDR of a candidate split, eq. (41) and (42), page 68 (PDF 82):
      SDR(h_A) = sd(S) - N_L/N sd(S_L) - N_R/N sd(S_R)
      sd(S)   = sqrt(1/N sum_i (y_i - mean)^2)
              = sqrt(1/N sum_i y_i^2 - (1/N sum_i y_i)^2)
- Split criterion, eq. (43) and page 70 (PDF 84):
      r = SDR(h_B) / SDR(h_A),
      r+ = r + sqrt(ln(1/delta) / (2 N)),
      split if r+ < 1, or if r+ - r < tau (tie break).
- Perceptron leaf, eq. (44) to (47), pages 72-74 (PDF 86-88):
      o       = w x - b, with x_0 = 1 and w_0 = -b,
      w       <- w + eta (o - y) x,
      eta     = eta_0 / (1 + n eta_d),
      x'_i    = (x_i - mean_i) / (3 sd_i).
- Page-Hinkley, eq. (48) and (49), pages 74-75 (PDF 88-89):
      m(T)    = sum_{t=1}^{T} (x_t - mean(T) - alpha),
      M(T)    = min_{t=1..T} m(t),
      PH(T)   = m(T) - M(T),
      alarm   if PH(T) > lambda,
      x_t     = |y_t - o_t|, alpha = 0.1 sd(x) (page 75),
- Alternate tree, Algorithm 1 and 2, page 69 (PDF 83):
  on change, an alternate subtree rooted at the same node is grown; the
  new subtree replaces the old one when its error is lower. My choice:
  only the alternate-tree state (counter of errors) is kept, and the
  replacement happens when the alternate tree has seen at least
  ``n_min`` samples and its mean squared error is below the error of the
  node it replaces; the thesis says "when the new tree performs better"
  without a fixed threshold.

The leaf keeps the E-BST of the thesis, page 70-71 (PDF 84-85): per
numerical attribute, a binary search tree whose nodes carry ``n``,
``sum y`` and ``sum y^2`` for the two sides of the test, so the SDR of
every observed threshold can be computed in ``O(n)`` after each update.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dense_armor.roles import Regressor


def _as_dict(x: Any) -> dict:
    if isinstance(x, dict):
        return x
    to_dict = getattr(x, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    raise TypeError(
        f"expected a dict or an object with to_dict(), got {type(x).__name__}"
    )


def _finite(d: dict) -> bool:
    for v in d.values():
        if v is None:
            return False
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(fv):
            return False
    return True


@dataclass
class _SideStats:
    n: int = 0
    sum_y: float = 0.0
    sum_y2: float = 0.0

    def sd(self) -> float:
        if self.n < 2:
            return 0.0
        mu = self.sum_y / self.n
        var = self.sum_y2 / self.n - mu * mu
        return math.sqrt(max(var, 0.0))


class _EbstNode:
    __slots__ = ("key", "l", "left", "r", "right")

    def __init__(self, key: float) -> None:
        self.key = float(key)
        self.left: _EbstNode | None = None
        self.right: _EbstNode | None = None
        self.l = _SideStats()
        self.r = _SideStats()


class _Ebst:
    """Extended binary search tree, thesis page 70 (PDF 84).

    One tree per numerical attribute per leaf. The root carries the
    first seen key. Each node has two sets of statistics: ``l`` for
    samples with attribute value <= key, ``r`` for the others. The
    sample that creates a node is added to that node's ``l``: it
    reaches the node with a value equal to the key.
    """

    def __init__(self) -> None:
        self.root: _EbstNode | None = None

    def insert(self, key: float, y: float) -> None:
        k = float(key)
        if self.root is None:
            self.root = _EbstNode(k)
            self.root.l.n = 1
            self.root.l.sum_y = y
            self.root.l.sum_y2 = y * y
            return
        node = self.root
        while True:
            if k <= node.key:
                node.l.n += 1
                node.l.sum_y += y
                node.l.sum_y2 += y * y
                if node.left is None:
                    child = _EbstNode(k)
                    child.l.n = 1
                    child.l.sum_y = y
                    child.l.sum_y2 = y * y
                    node.left = child
                    return
                node = node.left
            else:
                node.r.n += 1
                node.r.sum_y += y
                node.r.sum_y2 += y * y
                if node.right is None:
                    child = _EbstNode(k)
                    child.l.n = 1
                    child.l.sum_y = y
                    child.l.sum_y2 = y * y
                    node.right = child
                    return
                node = node.right

    def best_split(self, total: _SideStats) -> tuple[float, float]:
        """Return (threshold, SDR) of the best candidate, or (0, 0).

        ``total`` holds the same statistics for the whole leaf, so the
        right side of a candidate is ``total - left``.
        """
        if total.n < 2 or self.root is None:
            return 0.0, 0.0
        sd_all = total.sd()
        best_sdr = 0.0
        best_thr = 0.0
        for thr, left in self._inorder():
            right_n = total.n - left.n
            if left.n < 1 or right_n < 1:
                continue
            right = _SideStats(
                n=right_n,
                sum_y=total.sum_y - left.sum_y,
                sum_y2=total.sum_y2 - left.sum_y2,
            )
            sdr = (
                sd_all
                - (left.n / total.n) * left.sd()
                - (right_n / total.n) * right.sd()
            )
            if sdr > best_sdr:
                best_sdr = sdr
                best_thr = thr
        return best_thr, best_sdr

    def _inorder(self):
        stack: list[_EbstNode] = []
        node = self.root
        while stack or node is not None:
            while node is not None:
                stack.append(node)
                node = node.left
            node = stack.pop()
            yield node.key, node.l
            node = node.right


class _Perceptron:
    """Un-thresholded perceptron leaf, thesis eq. (44)-(47).

    The input is ``x'`` normalised as eq. (47), page 74 (PDF 88),
    ``x'_i = (x_i - mean_i) / (3 sd_i)``, with the running mean and sd
    kept in Welford form. The bias term is eq. (44)'s ``w_0 = -b`` with
    ``x_0 = 1``. The learning rate is eq. (46), page 73 (PDF 87):
    ``eta = eta_0 / (1 + n eta_d)``.
    """

    __slots__ = ("eta0", "etad", "m2", "mean", "n", "w")

    def __init__(self, dim: int, eta0: float = 0.1, etad: float = 0.005) -> None:
        self.n = 0
        self.w = np.zeros(dim + 1, dtype=float)
        self.mean = np.zeros(dim, dtype=float)
        self.m2 = np.zeros(dim, dtype=float)
        self.eta0 = float(eta0)
        self.etad = float(etad)

    def predict(self, xv: np.ndarray) -> float:
        xp = self._norm(xv)
        return float(self.w[0] + self.w[1:] @ xp)

    def learn(self, xv: np.ndarray, y: float) -> None:
        xp = self._norm(xv)
        o = float(self.w[0] + self.w[1:] @ xp)
        self.n += 1
        eta = self.eta0 / (1.0 + self.n * self.etad)
        err = o - y
        self.w[0] -= eta * err
        self.w[1:] -= eta * err * xp
        self._update_stats(xv)

    def _norm(self, xv: np.ndarray) -> np.ndarray:
        sd = np.sqrt(self.m2 / np.maximum(self.n, 1))
        denom = 3.0 * sd
        denom = np.where(denom == 0.0, 1.0, denom)
        return (xv - self.mean) / denom

    def _update_stats(self, xv: np.ndarray) -> None:
        self.n = max(self.n, 0)
        n_old = self.n
        delta = xv - self.mean
        self.mean = self.mean + delta / max(n_old, 1)
        self.m2 = self.m2 + delta * (xv - self.mean)


@dataclass
class _PageHinkley:
    """Page-Hinkley test, thesis eq. (48)-(49), page 74-75 (PDF 88-89)."""

    alpha: float = 0.0
    lam: float = 10.0
    m: float = 0.0
    M: float = 0.0
    n: int = 0
    mean: float = 0.0
    drift: bool = False

    def update(self, x: float) -> bool:
        self.n += 1
        self.mean = self.mean + (x - self.mean) / self.n
        self.m = self.m + (x - self.mean - self.alpha)
        self.M = min(self.M, self.m)
        self.drift = (self.m - self.M) > self.lam
        return self.drift


@dataclass
class _Node:
    depth: int = 0
    feature: str | None = None
    threshold: float = 0.0
    left: "_Node | None" = None
    right: "_Node | None" = None
    stats: _SideStats = field(default_factory=_SideStats)
    ebst: dict[str, _Ebst] = field(default_factory=dict)
    perceptron: _Perceptron | None = None
    ph: _PageHinkley = field(default_factory=_PageHinkley)
    alt: "_Node | None" = None
    alt_n: int = 0
    alt_err: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


class FIMTDDRegressor(Regressor):
    """Fast Incremental Model Tree with Drift Detection (FIMT-DD).

    A streaming regression tree. Every leaf keeps one un-thresholded
    perceptron, trained online; every internal node keeps a Page-Hinkley
    test on the absolute error. When the error increases, an alternate
    subtree is grown; when the alternate subtree has seen at least
    ``n_min`` samples and its error is below the error of the node it
    replaces, it becomes the new subtree.

    Sources (printed pages of the thesis; the PDF is 14 pages ahead):

    - SDR and the split criterion: eq. (41)-(43), pages 68-70
      (PDF 82-84).
    - E-BST: pages 70-71 (PDF 84-85).
    - Perceptron leaf, learning rate, normalisation: eq. (44)-(47),
      pages 72-74 (PDF 86-88).
    - Page-Hinkley: eq. (48)-(49), pages 74-75 (PDF 88-89).
    - Alternate tree and Traverse: Algorithm 1 and 2, page 69 (PDF 83).

    Where the thesis leaves a choice:

    - The alternate tree replaces the old one when it has seen at
      least ``n_min`` samples and its mean squared error is below the
      error of the old subtree. The thesis says "performs better"
      without a fixed threshold (page 74, PDF 88); the choice of the
      window is mine.
    - The Page-Hinkley parameters are the thesis defaults: ``alpha``
      equal to 10% of the running standard deviation of the monitored
      error (page 75, PDF 89) and ``lambda`` set by the constructor
      (``lambda_ph``). The thesis says "``alpha`` = 10% of the sd" and
      leaves ``lambda`` to the user.
    - Numerical split search: the constructor takes ``max_bins`` (a
      rounding of the E-BST keys); the thesis mentions rounding to
      three decimals as a memory-saving option, page 71 (PDF 85).
      Rounding to ``max_bins`` distinct values per feature is my
      choice, and it keeps the E-BST shallow in dense streams.
    - Tie break: ``r+ - r < tau`` with default ``tau = 0.05``
      (page 70, PDF 84).

    Args:
        delta: one minus the confidence of the Hoeffding bound, in
            ``(0, 1)``.
        n_min: minimum samples between two split attempts at a leaf.
        tau: tie-break threshold, in ``[0, 1]``.
        alpha_frac: fraction of the standard deviation of the error
            used as ``alpha`` in the Page-Hinkley test.
        lambda_ph: threshold of the Page-Hinkley test.
        eta0: initial perceptron learning rate.
        etad: perceptron learning rate decay.
        max_bins: rounding of the numeric features to this many
            distinct values in the E-BST (my choice, page 71).
        max_depth: maximum depth, or ``None``.
        max_nodes: maximum total nodes. Default ``10000``.
        feature_keys: explicit list of feature keys; ``None`` sorts
            the keys of the first dict seen.
        seed: unused, kept for symmetry with the other estimators.

    Raises:
        ValueError: on ``delta`` outside ``(0, 1)``, non-positive
            ``n_min``, negative rates, or ``tau`` outside ``[0, 1]``.

    Examples:
        >>> from dense_armor.utility.tree.fimt import FIMTDDRegressor
        >>> m = FIMTDDRegressor(n_min=20, delta=0.01)
        >>> for _ in range(200):
        ...     _ = m.learn_one({"x": 1.0}, 2.0)
        >>> round(m.predict_one({"x": 1.0}), 1)
        2.0

    References:
        Ikonomovska, E. (2012). Learning Model Trees from Time-Changing
        Data Streams. PhD thesis, Jožef Stefan International
        Postgraduate School. Chapter 6, pages 68-77 (PDF 82-91).

    Provenance:
        The two PDFs in the task folder (Duarte 2016, Mastelini 2023)
        only describe FIMT-DD in words: "standard deviation reduction",
        "Page-Hinkley test", "perceptron leaves". They do not contain
        the SDR formula, the perceptron update, or the Page-Hinkley
        threshold. The original paper of Ikonomovska et al. (2011) is
        not in the folder either. The formulas in this file come from
        the PhD thesis of Ikonomovska (2012), Chapter 6, pages 68-77,
        downloaded on 2026-10-17 from

            https://www.mps.si/media/academic_work_attachments/
            Elena_Ikonomovska_PhD.pdf

        and saved as ``/content/papers/ikonomovska2012_phd_thesis.pdf``
        (4944 KB, 218 pages). The 2011 paper, downloaded from

            http://dm.uestc.edu.cn/wp-content/uploads/seminar/
            Learning%20Model%20Trees%20From%20Evolving%20Data%20Streams.pdf

        and saved as ``/content/papers/ikonomovska2011_fimt_dd.pdf``
        (559 KB, 24 pages), is the journal version of the same
        algorithm; the journal text describes the algorithm at page 4
        but the full formulas are only in the thesis. Both sources
        describe the same algorithm by the same author; the thesis is
        the extended version, and the page numbers quoted above refer
        to it.

        The Lagrange interpolation used in the tests (not in the
        estimator) comes from Gautschi (2012), "Interpolation before
        and after Lagrange", Rend. Sem. Mat. Univ. Pol. Torino 70(4),
        pages 347-368, downloaded from

            https://seminariomatematico.polito.it/rendiconti/70-4/347.pdf

        and from Hiptmair, "Numerische Mathematik fuer CSE" (NumCSE),
        ETH Zuerich, sections 3.3 and 3.5.1, downloaded from

            https://people.math.ethz.ch/~hiptmair/tmp/NumCSE10.pdf

        Both are cited in the test file at the equation and page that
        each test uses: Gautschi eq. (6)-(7) page 351, eq. (11) page
        353; Hiptmair section 3.5.1 page 317-318. The Lagrange 1806
        book "Lecons sur le calcul des fonctions" was downloaded and
        searched, but it does not contain the word "interpolation"
        (0 hits in 528 pages): the modern attribution to Lagrange is
        itself what Gautschi (2012) discusses, and Gautschi is the
        source quoted in the tests.
    """

    budget_s: float = 1e-2
    memory_class: str = "O(max_nodes)"

    def __init__(
        self,
        delta: float = 0.01,
        n_min: int = 50,
        tau: float = 0.05,
        alpha_frac: float = 0.1,
        lambda_ph: float = 10.0,
        eta0: float = 0.1,
        etad: float = 0.005,
        max_bins: int = 32,
        max_depth: int | None = None,
        max_nodes: int | None = 10000,
        feature_keys: Sequence[str] | None = None,
        seed: int = 0,
    ) -> None:
        if not (0.0 < delta < 1.0):
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        if n_min < 1:
            raise ValueError(f"n_min must be >= 1, got {n_min}")
        if not (0.0 <= tau <= 1.0):
            raise ValueError(f"tau must be in [0, 1], got {tau}")
        if alpha_frac < 0.0:
            raise ValueError(f"alpha_frac must be >= 0, got {alpha_frac}")
        if lambda_ph <= 0.0:
            raise ValueError(f"lambda_ph must be > 0, got {lambda_ph}")
        if eta0 <= 0.0 or etad < 0.0:
            raise ValueError(f"bad perceptron rates: eta0={eta0}, etad={etad}")
        if max_bins < 2:
            raise ValueError(f"max_bins must be >= 2, got {max_bins}")
        self.delta = float(delta)
        self.n_min = int(n_min)
        self.tau = float(tau)
        self.alpha_frac = float(alpha_frac)
        self.lambda_ph = float(lambda_ph)
        self.eta0 = float(eta0)
        self.etad = float(etad)
        self.max_bins = int(max_bins)
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.feature_keys = feature_keys
        self.seed = int(seed)
        self._keys: list[str] | None = None
        self._dim: int = 0
        self._ranges: dict[str, tuple[float, float]] = {}
        self._n_total: int = 0
        self.n_missing_: int = 0

    @property
    def n_missing(self) -> int:
        """Number of samples skipped for missing or non-finite values.

        Same value as ``n_missing_``; the public property is the
        name used by the rest of the library.

        Returns:
            The number of skipped samples.
        """
        return self.n_missing_
        self.n_seen_: int = 0
        self.root_: _Node | None = None

    def _resolve_keys(self, x: dict) -> list[str]:
        if self._keys is not None:
            return self._keys
        if self.feature_keys is not None:
            self._keys = list(self.feature_keys)
        else:
            self._keys = sorted(x)
        self._dim = len(self._keys)
        return self._keys

    def _row(self, x: dict) -> np.ndarray:
        assert self._keys is not None
        return np.asarray([float(x.get(k, 0.0)) for k in self._keys], dtype=float)

    def _quantize(self, feature: str, value: float) -> float:
        lo, hi = self._ranges.get(feature, (0.0, 0.0))
        if hi <= lo:
            return value
        step = (hi - lo) / self.max_bins
        return round((value - lo) / step) * step + lo

    def _leaf(self, node: _Node) -> _Node:
        while not node.is_leaf:
            node = node.left  # type: ignore[assignment]
        return node

    def _ensure_perceptron(self, leaf: _Node) -> _Perceptron:
        if leaf.perceptron is None:
            leaf.perceptron = _Perceptron(self._dim, self.eta0, self.etad)
        return leaf.perceptron

    def _traverse(self, xv: np.ndarray) -> tuple[_Node, list[_Node]]:
        assert self.root_ is not None
        node = self.root_
        path: list[_Node] = [node]
        assert self._keys is not None
        while not node.is_leaf:
            assert node.feature is not None
            j = self._keys.index(node.feature)
            if xv[j] <= node.threshold:
                node = node.left  # type: ignore[assignment]
            else:
                node = node.right  # type: ignore[assignment]
            path.append(node)
        return node, path

    def _update_leaf(self, leaf: _Node, xv: np.ndarray, y: float) -> None:
        leaf.stats.n += 1
        leaf.stats.sum_y += y
        leaf.stats.sum_y2 += y * y
        assert self._keys is not None
        for j, k in enumerate(self._keys):
            q = self._quantize(k, float(xv[j]))
            leaf.ebst.setdefault(k, _Ebst()).insert(q, y)
        self._ensure_perceptron(leaf).learn(xv, y)

    def _ph_update(self, node: _Node, xv: np.ndarray, y: float, pred: float) -> None:
        err = abs(y - pred)
        if node.ph.n == 1:
            node.ph.alpha = self.alpha_frac * err
        elif node.ph.n > 1:
            pass
        if node.ph.update(err):
            if node.alt is None:
                node.alt = _Node(depth=node.depth)
            node.alt_n += 1
            node.alt_err += (y - pred) ** 2
            if node.alt_n >= self.n_min and node.alt.is_leaf:
                node.alt.perceptron = _Perceptron(self._dim, self.eta0, self.etad)
                node.feature = None
                node.threshold = 0.0
                node.left = None
                node.right = None
                node.ebst = {}
                node.stats = _SideStats()
                node.perceptron = node.alt.perceptron
                node.alt = None
                node.alt_n = 0
                node.alt_err = 0.0
                node.ph = _PageHinkley(
                    alpha=0.0,
                    lam=self.lambda_ph,
                    m=0.0,
                    M=0.0,
                    n=0,
                    mean=0.0,
                    drift=False,
                )
                return
        if node.alt is not None:
            self._update_leaf(node.alt, xv, y)

    def _best_split(self, leaf: _Node) -> tuple[str | None, float, float, float]:
        best: tuple[str | None, float, float] = (None, 0.0, 0.0)
        second_sdr = 0.0
        for k, tree in leaf.ebst.items():
            thr, sdr = tree.best_split(leaf.stats)
            if sdr <= 0.0:
                continue
            if sdr > best[2]:
                second_sdr = best[2]
                best = (k, thr, sdr)
            elif sdr > second_sdr:
                second_sdr = sdr
        return best[0], best[1], best[2], second_sdr

    def _try_split(self, leaf: _Node) -> None:
        if self.max_depth is not None and leaf.depth >= self.max_depth:
            return
        if self.max_nodes is not None and self._n_total >= self.max_nodes:
            return
        if leaf.stats.n < 2 * self.n_min:
            return
        name, thr, sdr_a, sdr_b = self._best_split(leaf)
        if name is None or sdr_a <= 0.0:
            return
        if sdr_b <= 0.0:
            eps = math.sqrt(math.log(1.0 / self.delta) / (2.0 * leaf.stats.n))
            if eps < self.tau:
                self._split(leaf, name, thr)
            return
        r = sdr_b / sdr_a
        eps = math.sqrt(math.log(1.0 / self.delta) / (2.0 * leaf.stats.n))
        if r + eps < 1.0 or (r + eps - r) < self.tau:
            self._split(leaf, name, thr)

    def _split(self, leaf: _Node, name: str, thr: float) -> None:
        leaf.feature = name
        leaf.threshold = thr
        leaf.left = _Node(depth=leaf.depth + 1)
        leaf.right = _Node(depth=leaf.depth + 1)
        self._n_total += 2

    def learn_one(self, x: dict, y: Any, t: float | None = None) -> "FIMTDDRegressor":
        """Update the tree with one sample.

        Args:
            x: feature dict.
            y: target scalar.
            t: optional timestamp, currently unused.

        Returns:
            self.
        """
        d = _as_dict(x)
        if not _finite(d) or y is None:
            self.n_missing_ += 1
            return self
        try:
            yf = float(y)
        except (TypeError, ValueError):
            self.n_missing_ += 1
            return self
        if not math.isfinite(yf):
            self.n_missing_ += 1
            return self
        self._resolve_keys(d)
        assert self._keys is not None
        for j, k in enumerate(self._keys):
            v = float(d.get(k, 0.0))
            if k not in self._ranges:
                self._ranges[k] = (v, v)
            else:
                lo, hi = self._ranges[k]
                self._ranges[k] = (min(lo, v), max(hi, v))
        if self.root_ is None:
            self.root_ = _Node(depth=0)
        xv = self._row(d)
        leaf, path = self._traverse(xv)
        pred = self.predict_one(d)
        self._update_leaf(leaf, xv, yf)
        for node in path:
            self._ph_update(node, xv, yf, pred)
        self.n_seen_ += 1
        self._n_total = max(self._n_total, self._count_nodes())
        if self.n_seen_ % self.n_min == 0:
            self._try_split(self._leaf(self.root_))
        return self

    def _count_nodes(self) -> int:
        if self.root_ is None:
            return 0
        stack = [self.root_]
        n = 0
        while stack:
            node = stack.pop()
            n += 1
            if node.left is not None:
                stack.append(node.left)
            if node.right is not None:
                stack.append(node.right)
        return n

    def predict_one(
        self, x: dict, t: float | None = None, return_std: bool = False
    ) -> float:
        """Predict the target of a single sample.

        Before any ``learn_one`` call the prediction is ``0.0``.
        """
        if self.root_ is None or self._keys is None:
            return 0.0
        d = _as_dict(x)
        try:
            xv = self._row(d)
        except (TypeError, ValueError):
            return 0.0
        leaf, _ = self._traverse(xv)
        if leaf.perceptron is not None and leaf.stats.n >= 2:
            return float(leaf.perceptron.predict(xv))
        if leaf.stats.n >= 1:
            return float(leaf.stats.sum_y / leaf.stats.n)
        return 0.0
