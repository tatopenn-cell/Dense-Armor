"""Recommenders built on the SGD of Paterek (2007) and Koren (2009)."""

from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np

from dense_armor.roles import Root


class _RecoBase(Root):
    """Common code for Baseline, BiasedMF and FunkMF.

    These classes do not subclass ``Classifier`` or ``Regressor`` because
    the task fixes the signature ``learn_one(user, item, rating)``. They
    inherit ``Root`` so ``clone`` and ``get_params`` work. They are not
    checked by ``check_estimator``, which assumes ``learn_one(x, y)``
    with a single dict; the two signatures cannot both hold, and the
    task's signature wins.

    Args:
        lr: SGD learning rate, strictly positive.
        l2: L2 regularization strength, non-negative.
        seed: seed of the initialiser.
        init_scale: standard deviation of the latent-factor initialiser.
    """

    def __init__(
        self,
        lr: float = 0.01,
        l2: float = 0.05,
        seed: int = 0,
        init_scale: float = 0.01,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if l2 < 0.0:
            raise ValueError(f"l2 must be >= 0, got {l2}")
        self.lr = float(lr)
        self.l2 = float(l2)
        self.seed = int(seed)
        self.init_scale = float(init_scale)
        self.mu_: float = 0.0
        self.n_seen_: int = 0
        self.n_missing_: int = 0
        self._rng = np.random.default_rng(self.seed)

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    def _check_input(self, user: Any, item: Any, rating: Any) -> bool:
        if user is None or item is None:
            self.n_missing_ += 1
            return False
        try:
            r = float(rating)
        except (TypeError, ValueError):
            self.n_missing_ += 1
            return False
        if not np.isfinite(r):
            self.n_missing_ += 1
            return False
        return True

    def _new_vector(self, k: int) -> np.ndarray:
        return self._rng.normal(0.0, self.init_scale, size=k)


class Baseline(_RecoBase):
    """Global mean plus per-user and per-item biases.

    The prediction is::

        y_hat = mu + b_u + b_i

    where `mu` is the running mean of the ratings seen so far, `b_u` the
    user bias and `b_i` the item bias. A rating `r` updates the three
    quantities one step of SGD with L2 on the biases::

        mu   <- running mean of the ratings seen so far, with r included
        e = r - (mu + b_u + b_i)
        b_u  <- b_u  + lr * (e - l2 * b_u)
        b_i  <- b_i  + lr * (e - l2 * b_i)

    This is the baseline that Koren (2009, Section "Adding biases", page 4)
    subtracts before fitting the latent factors, with the SGD update of
    "Stochastic gradient descent" on the same page.

    Args:
        lr: learning rate.
        l2: L2 strength on the biases.
        seed: seed of the initialiser, unused here (kept for symmetry).
        init_scale: unused, kept for symmetry.

    Examples:
        >>> from dense_armor.utility.reco import Baseline
        >>> r = Baseline(lr=0.05, l2=0.0, seed=0)
        >>> for _ in range(100):
        ...     _ = r.learn_one("alice", "movie_a", 5.0)
        ...     _ = r.learn_one("bob", "movie_a", 3.0)
        >>> round(r.predict_one("alice", "movie_a"), 1)
        5.0

    References:
        Koren, Y., Bell, R., Volinsky, C. (2009). Matrix Factorization
        Techniques for Recommender Systems. IEEE Computer 42(8).
        Section "Adding biases", page 4, and the SGD update, page 4.
        Paterek, A. (2007). Improving Regularized SVD. In Netflix Prize
        workshop, KDD Cup and Workshop 2007. Section 3.3, eq. (2), page 2.
    """

    budget_s: float = 2e-4
    memory_class: str = "O(n_users + n_items)"

    def __init__(
        self,
        lr: float = 0.01,
        l2: float = 0.0,
        seed: int = 0,
        init_scale: float = 0.01,
    ) -> None:
        super().__init__(lr=lr, l2=l2, seed=seed, init_scale=init_scale)
        self.b_u_: dict = {}
        self.b_i_: dict = {}

    def learn_one(self, user: Hashable, item: Hashable, rating: Any) -> "Baseline":
        """One SGD step on a single (user, item, rating)."""
        if not self._check_input(user, item, rating):
            return self
        r = float(rating)
        self.n_seen_ += 1
        self.mu_ += (r - self.mu_) / self.n_seen_
        b_u = self.b_u_.get(user, 0.0)
        b_i = self.b_i_.get(item, 0.0)
        e = r - (self.mu_ + b_u + b_i)
        self.b_u_[user] = b_u + self.lr * (e - self.l2 * b_u)
        self.b_i_[item] = b_i + self.lr * (e - self.l2 * b_i)
        return self

    def predict_one(self, user: Hashable, item: Hashable) -> float:
        """Predicted rating, ``mu_`` for an unknown user or item."""
        b_u = self.b_u_.get(user, 0.0)
        b_i = self.b_i_.get(item, 0.0)
        return float(self.mu_ + b_u + b_i)


class BiasedMF(_RecoBase):
    """Global mean, biases and latent factors, with SGD.

    The prediction is::

        y_hat = mu + b_u + b_i + <p_u, q_i>

    with `p_u, q_i in R^k`. A rating `r` updates `mu`, the two biases
    and the two latent vectors one step of SGD with L2 on every
    parameter::

        mu   <- running mean of the ratings seen so far, with r included
        e    = r - y_hat, with the new mu
        b_u  <- b_u  + lr * (e - l2 * b_u)
        b_i  <- b_i  + lr * (e - l2 * b_i)
        p_u  <- p_u  + lr * (e * q_i - l2 * p_u)
        q_i  <- q_i  + lr * (e * p_u - l2 * q_i)

    The last two updates use the old `p_u` and `q_i` (the `q_i` update
    must not see the new `p_u`), which is the standard SGD ordering
    on the two vectors of a single rating.

    This is the "Improved regularized SVD" of Paterek (2007, eq. (2),
    page 2) written with the notation and the baseline of Koren (2009,
    Section "Adding biases", page 4, and the SGD on the same page).

    Args:
        k: number of latent factors, strictly positive.
        lr: learning rate.
        l2: L2 strength on every parameter.
        seed: seed of the initialiser.
        init_scale: standard deviation of the latent vectors.

    Raises:
        ValueError: if ``k < 1``, ``lr <= 0`` or ``l2 < 0``.

    Examples:
        >>> from dense_armor.utility.reco import BiasedMF
        >>> m = BiasedMF(k=4, lr=0.05, l2=0.0, seed=0)
        >>> for _ in range(100):
        ...     _ = m.learn_one("alice", "movie_a", 5.0)
        ...     _ = m.learn_one("bob", "movie_a", 3.0)
        >>> round(m.predict_one("alice", "movie_a"), 1)
        5.0

    References:
        Paterek, A. (2007). Improving Regularized SVD. eq. (2), page 2.
        Koren, Y., Bell, R., Volinsky, C. (2009). Matrix Factorization
        Techniques for Recommender Systems. Section "Adding biases",
        page 4, and the SGD update, page 4.
    """

    budget_s: float = 1e-2
    memory_class: str = "O((n_users + n_items) * k)"

    def __init__(
        self,
        k: int = 8,
        lr: float = 0.01,
        l2: float = 0.05,
        seed: int = 0,
        init_scale: float = 0.01,
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        super().__init__(lr=lr, l2=l2, seed=seed, init_scale=init_scale)
        self.k = int(k)
        self.b_u_: dict = {}
        self.b_i_: dict = {}
        self.p_: dict = {}
        self.q_: dict = {}

    def _ensure_user(self, user: Hashable) -> None:
        if user not in self.p_:
            self.p_[user] = self._new_vector(self.k)
            self.b_u_[user] = 0.0

    def _ensure_item(self, item: Hashable) -> None:
        if item not in self.q_:
            self.q_[item] = self._new_vector(self.k)
            self.b_i_[item] = 0.0

    def learn_one(self, user: Hashable, item: Hashable, rating: Any) -> "BiasedMF":
        """One SGD step on a single (user, item, rating)."""
        if not self._check_input(user, item, rating):
            return self
        r = float(rating)
        self._ensure_user(user)
        self._ensure_item(item)
        self.n_seen_ += 1
        self.mu_ += (r - self.mu_) / self.n_seen_
        p_u = self.p_[user]
        q_i = self.q_[item]
        b_u = self.b_u_[user]
        b_i = self.b_i_[item]
        e = r - (self.mu_ + b_u + b_i + float(p_u @ q_i))
        self.b_u_[user] = b_u + self.lr * (e - self.l2 * b_u)
        self.b_i_[item] = b_i + self.lr * (e - self.l2 * b_i)
        p_u_new = p_u + self.lr * (e * q_i - self.l2 * p_u)
        q_i_new = q_i + self.lr * (e * p_u - self.l2 * q_i)
        self.p_[user] = p_u_new
        self.q_[item] = q_i_new
        return self

    def predict_one(self, user: Hashable, item: Hashable) -> float:
        """Predicted rating, ``mu_`` for an unknown user or item."""
        b_u = self.b_u_.get(user, 0.0)
        b_i = self.b_i_.get(item, 0.0)
        if user not in self.p_ or item not in self.q_:
            return float(self.mu_ + b_u + b_i)
        return float(self.mu_ + b_u + b_i + float(self.p_[user] @ self.q_[item]))

    def rank(
        self, user: Hashable, items: Sequence[Hashable], top_k: int | None = None
    ) -> list:
        """Items sorted by predicted rating, the best first.

        Args:
            user: the user.
            items: the items to rank.
            top_k: keep only the first ``top_k``; ``None`` keeps all.

        Returns:
            The items sorted by decreasing predicted rating.
        """
        scored = [(it, self.predict_one(user, it)) for it in items]
        scored.sort(key=lambda t: -t[1])
        return [it for it, _ in scored[: (top_k or len(scored))]]


class FunkMF(_RecoBase):
    """Plain regularized MF with SGD, no user or item biases.

    The prediction is::

        y_hat = mu + <p_u, q_i>

    where `mu` is the running mean of the ratings. A rating `r` updates
    `mu`, `p_u`, `q_i` with SGD and L2 on the two latent vectors.

    This is eq. (1) of Paterek (2007), page 2, without the baseline
    that Paterek subtracts before the fit. The update rule is the one
    popularised by Simon Funk's blog post (no citation number, as the
    task asks); the mean is included so the prediction is on the same
    scale as the ratings.

    Args:
        k: number of latent factors, strictly positive.
        lr: learning rate.
        l2: L2 strength on the latent vectors.
        seed: seed of the initialiser.
        init_scale: standard deviation of the latent vectors.

    Raises:
        ValueError: if ``k < 1``, ``lr <= 0`` or ``l2 < 0``.

    Examples:
        >>> from dense_armor.utility.reco import FunkMF
        >>> m = FunkMF(k=4, lr=0.05, l2=0.0, seed=0)
        >>> for _ in range(100):
        ...     _ = m.learn_one("alice", "movie_a", 5.0)
        ...     _ = m.learn_one("bob", "movie_a", 3.0)
        >>> 0.0 < m.predict_one("alice", "movie_a") < 6.0
        True

    References:
        Paterek, A. (2007). Improving Regularized SVD. eq. (1), page 2.
    """

    budget_s: float = 2e-3
    memory_class: str = "O((n_users + n_items) * k)"

    def __init__(
        self,
        k: int = 8,
        lr: float = 0.01,
        l2: float = 0.05,
        seed: int = 0,
        init_scale: float = 0.01,
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        super().__init__(lr=lr, l2=l2, seed=seed, init_scale=init_scale)
        self.k = int(k)
        self.p_: dict = {}
        self.q_: dict = {}

    def _ensure_user(self, user: Hashable) -> None:
        if user not in self.p_:
            self.p_[user] = self._new_vector(self.k)

    def _ensure_item(self, item: Hashable) -> None:
        if item not in self.q_:
            self.q_[item] = self._new_vector(self.k)

    def learn_one(self, user: Hashable, item: Hashable, rating: Any) -> "FunkMF":
        """One SGD step on a single (user, item, rating)."""
        if not self._check_input(user, item, rating):
            return self
        r = float(rating)
        self._ensure_user(user)
        self._ensure_item(item)
        self.n_seen_ += 1
        self.mu_ += (r - self.mu_) / self.n_seen_
        p_u = self.p_[user]
        q_i = self.q_[item]
        e = r - (self.mu_ + float(p_u @ q_i))
        p_u_new = p_u + self.lr * (e * q_i - self.l2 * p_u)
        q_i_new = q_i + self.lr * (e * p_u - self.l2 * q_i)
        self.p_[user] = p_u_new
        self.q_[item] = q_i_new
        return self

    def predict_one(self, user: Hashable, item: Hashable) -> float:
        """Predicted rating, ``mu_`` for an unknown user or item."""
        if user not in self.p_ or item not in self.q_:
            return float(self.mu_)
        return float(self.mu_ + float(self.p_[user] @ self.q_[item]))

    def rank(
        self, user: Hashable, items: Sequence[Hashable], top_k: int | None = None
    ) -> list:
        """Items sorted by predicted rating, the best first."""
        scored = [(it, self.predict_one(user, it)) for it in items]
        scored.sort(key=lambda t: -t[1])
        return [it for it, _ in scored[: (top_k or len(scored))]]
