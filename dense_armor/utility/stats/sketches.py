"""Streaming sketches for frequencies, distinct count, membership and
heavy hitters.

Four structures, one sample at a time, constant memory:

- **CountMinSketch** — Cormode & Muthukrishnan (2005). Frequency
  estimation with ``eps · N`` one-sided error and probability
  ``1 - delta``, using ``w = ceil(e / eps)`` counters per row and
  ``d = ceil(ln(1 / delta))`` rows.
- **HyperLogLog** — Flajolet et al. (2007). Distinct count with about
  ``1.04 / sqrt(m)`` relative standard error, using ``m = 2^p`` registers.
- **BloomFilter** — Bloom (1970). Membership test with no false
  negatives and a false positive rate of at most ``p_false``.
- **SpaceSaving** — Metwally et al. (2005). Top-k heavy hitters with a
  deterministic error bound: the reported count of the true k-th most
  frequent item is at least its true count minus ``n / k``.

All four are :class:`Transformer` subclasses: ``learn_one(x)`` inserts
one item, ``merge(other)`` combines two sketches of the same shape.

References
----------
Cormode, G., Muthukrishnan, S. (2005). An improved data stream summary:
    the Count-Min sketch and its applications. J. Algorithms 55(1),
    58-75.
Flajolet, P., Fusy, E., Gandouet, O., Meunier, F. (2007). HyperLogLog:
    the analysis of a near-optimal cardinality estimation algorithm.
    In AofA.
Bloom, B. H. (1970). Space/time trade-offs in hash coding with
    allowable errors. Communications of the ACM 13(7), 422-426.
Metwally, A., Agrawal, D., El Abbadi, A. (2005). Efficient computation
    of frequent and top-k elements in data streams. In ICDE.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any

from dense_armor.base import Transformer


def _hash64(item: Any, seed: int) -> int:
    """Stable 64-bit hash of an item, keyed by ``seed``."""
    h = hashlib.blake2b(
        str(item).encode("utf-8"),
        digest_size=8,
        key=seed.to_bytes(4, "little", signed=False),
    )
    return int.from_bytes(h.digest(), "little")


class CountMinSketch(Transformer):
    """Count-Min frequency sketch (Cormode & Muthukrishnan 2005).

    For any item ``a`` observed ``f_a`` times and any stream length
    ``N``,

    .. math::

        f_a \\le \\hat{f}_a \\le f_a + \\varepsilon N

    with probability at least ``1 - \\delta``, where
    ``w = ceil(e / eps)`` and ``d = ceil(ln(1 / delta))``. The error is
    one-sided: the estimate is never below the true count.

    Args:
        epsilon: additive error factor, in ``(0, 1)``. Default ``0.01``.
        delta: failure probability, in ``(0, 1)``. Default ``0.01``.
        feature: dict key that holds the item. ``None`` reads the
            smallest key.
        seed: hash seed, so different runs use different hash functions
            unless set.

    Examples:
        >>> from dense_armor.utility.stats.sketches import CountMinSketch
        >>> cm = CountMinSketch(epsilon=0.001, delta=0.001)
        >>> for w in ["a", "b", "a", "c", "a", "b", "a"]:
        ...     _ = cm.learn_one({"w": w})
        >>> cm.estimate("a") >= 4
        True

    References:
        Cormode, G., Muthukrishnan, S. (2005). J. Algorithms 55(1),
        58-75.
    """

    def __init__(
        self,
        epsilon: float = 0.01,
        delta: float = 0.01,
        feature: str | None = None,
        seed: int = 42,
    ) -> None:
        if not 0.0 < epsilon < 1.0:
            raise ValueError(f"epsilon must be in (0, 1), got {epsilon}")
        if not 0.0 < delta < 1.0:
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        self.epsilon = epsilon
        self.delta = delta
        self.feature = feature
        self.seed = seed
        self.w_: int = math.ceil(math.e / epsilon)
        self.d_: int = math.ceil(math.log(1.0 / delta))
        self.table_: list[list[int]] = [[0] * self.w_ for _ in range(self.d_)]
        self.n_ = 0

    def _key(self, x: dict) -> str:
        return min(x) if self.feature is None else self.feature

    def _row_col(self, item: Any, row: int) -> int:
        return _hash64(item, self.seed + row) % self.w_

    def learn_one(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "CountMinSketch":
        """Insert one item from ``x``."""
        self._time_step(t)
        item = x[self._key(x)]
        for row in range(self.d_):
            col = self._row_col(item, row)
            self.table_[row][col] += 1
        self.n_ += 1
        return self

    def update(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "CountMinSketch":
        return self.learn_one(x, y=y, t=t)

    def estimate(self, item: Any) -> int:
        """Estimated frequency of ``item``. Never below the true count."""
        return min(
            self.table_[row][self._row_col(item, row)] for row in range(self.d_)
        )

    def merge(self, other: "CountMinSketch") -> "CountMinSketch":
        """Combine two Count-Min sketches with the same shape.

        Args:
            other: another ``CountMinSketch`` with the same ``epsilon``,
                ``delta`` and ``seed``.

        Returns:
            A new ``CountMinSketch`` with the summed counters.

        Raises:
            TypeError: if ``other`` is not a ``CountMinSketch``.
            ValueError: if the shapes differ.
        """
        if not isinstance(other, CountMinSketch):
            raise TypeError(
                f"merge expects CountMinSketch, got {type(other).__name__}"
            )
        if (
            self.w_ != other.w_
            or self.d_ != other.d_
            or self.seed != other.seed
        ):
            raise ValueError("merge requires the same shape and seed")
        out = CountMinSketch(
            epsilon=self.epsilon,
            delta=self.delta,
            feature=self.feature,
            seed=self.seed,
        )
        for row in range(self.d_):
            for col in range(self.w_):
                out.table_[row][col] = (
                    self.table_[row][col] + other.table_[row][col]
                )
        out.n_ = self.n_ + other.n_
        return out

    @property
    def count(self) -> int:
        return self.n_

    @property
    def width(self) -> int:
        return self.w_

    @property
    def depth(self) -> int:
        return self.d_

    def transform_one(self, x: dict, t: float | None = None) -> dict:
        item = x[self._key(x)]
        return {
            "count": self.n_,
            "estimate": self.estimate(item),
            "width": self.w_,
            "depth": self.d_,
        }


class HyperLogLog(Transformer):
    """HyperLogLog distinct-count sketch (Flajolet et al. 2007).

    With ``m = 2^p`` registers, the estimator has a relative standard
    error of about ``1.04 / sqrt(m)``. The implementation uses 64-bit
    hashes, so the large-range correction of the paper is not needed
    for streams below ~10^17 distinct items.

    Args:
        precision: number of index bits, ``p`` in ``[4, 18]``. Default
            ``14``, i.e. ``m = 16384`` registers.
        feature: dict key that holds the item. ``None`` reads the
            smallest key.
        seed: hash seed.

    Examples:
        >>> from dense_armor.utility.stats.sketches import HyperLogLog
        >>> h = HyperLogLog(precision=10)
        >>> for i in range(1000):
        ...     _ = h.learn_one({"w": f"item{i}"})
        >>> abs(h.count() - 1000) / 1000 < 0.1
        True

    References:
        Flajolet, P., Fusy, E., Gandouet, O., Meunier, F. (2007). AofA.
    """

    def __init__(
        self,
        precision: int = 14,
        feature: str | None = None,
        seed: int = 42,
    ) -> None:
        if not 4 <= precision <= 18:
            raise ValueError(f"precision must be in [4, 18], got {precision}")
        self.precision = precision
        self.feature = feature
        self.seed = seed
        self.m_: int = 1 << precision
        self.registers_: list[int] = [0] * self.m_
        self.n_ = 0
        alpha: float
        if self.m_ == 16:
            alpha = 0.673
        elif self.m_ == 32:
            alpha = 0.697
        elif self.m_ == 64:
            alpha = 0.709
        else:
            alpha = 0.7213 / (1.0 + 1.079 / self.m_)
        self.alpha_: float = alpha

    def _key(self, x: dict) -> str:
        return min(x) if self.feature is None else self.feature

    def _rho(self, w: int, bits: int) -> int:
        if w == 0:
            return bits + 1
        shift = bits - w.bit_length() + 1
        return shift if shift > 0 else 1

    def learn_one(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "HyperLogLog":
        """Insert one item."""
        self._time_step(t)
        item = x[self._key(x)]
        h = _hash64(item, self.seed)
        index = h & (self.m_ - 1)
        w = h >> self.precision
        rho = self._rho(w, 64 - self.precision)
        if rho > self.registers_[index]:
            self.registers_[index] = rho
        self.n_ += 1
        return self

    def update(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "HyperLogLog":
        return self.learn_one(x, y=y, t=t)

    def count(self) -> float:
        """Estimated number of distinct items seen so far."""
        z = sum(2.0 ** (-r) for r in self.registers_)
        e = self.alpha_ * self.m_ * self.m_ / z
        if e <= 2.5 * self.m_:
            v = sum(1 for r in self.registers_ if r == 0)
            if v > 0:
                e = self.m_ * math.log(self.m_ / v)
        return e

    def merge(self, other: "HyperLogLog") -> "HyperLogLog":
        """Combine two HyperLogLog sketches with the same precision.

        Args:
            other: another ``HyperLogLog`` with the same ``precision``
                and ``seed``.

        Returns:
            A new ``HyperLogLog`` with the register-wise maximum.

        Raises:
            TypeError: if ``other`` is not a ``HyperLogLog``.
            ValueError: if the precision or seed differ.
        """
        if not isinstance(other, HyperLogLog):
            raise TypeError(
                f"merge expects HyperLogLog, got {type(other).__name__}"
            )
        if self.precision != other.precision or self.seed != other.seed:
            raise ValueError("merge requires the same precision and seed")
        out = HyperLogLog(
            precision=self.precision,
            feature=self.feature,
            seed=self.seed,
        )
        for i in range(self.m_):
            out.registers_[i] = max(self.registers_[i], other.registers_[i])
        out.n_ = self.n_ + other.n_
        return out

    @property
    def count_total(self) -> int:
        return self.n_

    def transform_one(self, x: dict, t: float | None = None) -> dict:
        return {
            "count": self.n_,
            "distinct": self.count(),
            "registers": self.m_,
        }


class BloomFilter(Transformer):
    """Bloom membership filter (Bloom 1970).

    For ``n`` expected items and target false-positive probability
    ``p_false``, the optimal number of bits is
    ``m = ceil(-n · ln p_false / (ln 2)^2)`` and the optimal number of
    hash functions is ``k = round((m / n) · ln 2)``.

    No false negatives: an item that was inserted always tests as
    present. False positives occur with probability at most
    ``p_false``.

    Args:
        n_expected: expected number of distinct items.
        p_false: target false-positive probability.
        feature: dict key that holds the item. ``None`` reads the
            smallest key.
        seed: hash seed.

    Examples:
        >>> from dense_armor.utility.stats.sketches import BloomFilter
        >>> b = BloomFilter(n_expected=1000, p_false=0.01)
        >>> for w in ["a", "b", "c"]:
        ...     _ = b.learn_one({"w": w})
        >>> b.check("a"), b.check("b"), b.check("c")
        (True, True, True)
        >>> b.check("d") in (True, False)
        True

    References:
        Bloom, B. H. (1970). Communications of the ACM 13(7), 422-426.
    """

    def __init__(
        self,
        n_expected: int = 1000,
        p_false: float = 0.01,
        feature: str | None = None,
        seed: int = 42,
    ) -> None:
        if n_expected < 1:
            raise ValueError(f"n_expected must be >= 1, got {n_expected}")
        if not 0.0 < p_false < 1.0:
            raise ValueError(f"p_false must be in (0, 1), got {p_false}")
        self.n_expected = n_expected
        self.p_false = p_false
        self.feature = feature
        self.seed = seed
        self.m_: int = max(
            8,
            math.ceil(-n_expected * math.log(p_false) / (math.log(2) ** 2)),
        )
        self.k_: int = max(1, round((self.m_ / n_expected) * math.log(2)))
        self.bits_: bytearray = bytearray((self.m_ + 7) // 8)
        self.n_ = 0

    def _key(self, x: dict) -> str:
        return min(x) if self.feature is None else self.feature

    def _positions(self, item: Any) -> list[int]:
        return [_hash64(item, self.seed + i) % self.m_ for i in range(self.k_)]

    def learn_one(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "BloomFilter":
        """Insert one item."""
        self._time_step(t)
        item = x[self._key(x)]
        for pos in self._positions(item):
            self.bits_[pos // 8] |= 1 << (pos % 8)
        self.n_ += 1
        return self

    def update(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "BloomFilter":
        return self.learn_one(x, y=y, t=t)

    def check(self, item: Any) -> bool:
        """Membership test: ``True`` if the item is (probably) present."""
        for pos in self._positions(item):
            if not (self.bits_[pos // 8] & (1 << (pos % 8))):
                return False
        return True

    def __contains__(self, item: Any) -> bool:
        return self.check(item)

    def merge(self, other: "BloomFilter") -> "BloomFilter":
        """Combine two Bloom filters with the same shape.

        Args:
            other: another ``BloomFilter`` with the same ``m``, ``k``
                and ``seed``.

        Returns:
            A new ``BloomFilter`` with the bit-wise OR.

        Raises:
            TypeError: if ``other`` is not a ``BloomFilter``.
            ValueError: if the shapes differ.
        """
        if not isinstance(other, BloomFilter):
            raise TypeError(
                f"merge expects BloomFilter, got {type(other).__name__}"
            )
        if (
            self.m_ != other.m_
            or self.k_ != other.k_
            or self.seed != other.seed
        ):
            raise ValueError("merge requires the same shape and seed")
        out = BloomFilter(
            n_expected=self.n_expected,
            p_false=self.p_false,
            feature=self.feature,
            seed=self.seed,
        )
        out.bits_ = bytearray(a | b for a, b in zip(self.bits_, other.bits_))
        out.n_ = self.n_ + other.n_
        return out

    @property
    def count(self) -> int:
        return self.n_

    @property
    def n_bits(self) -> int:
        return self.m_

    @property
    def n_hashes(self) -> int:
        return self.k_

    @property
    def fill_ratio(self) -> float:
        """Fraction of set bits, useful to check the FP rate."""
        return sum(b.bit_count() for b in self.bits_) / self.m_

    def transform_one(self, x: dict, t: float | None = None) -> dict:
        item = x[self._key(x)]
        return {
            "count": self.n_,
            "present": self.check(item),
            "n_bits": self.m_,
            "n_hashes": self.k_,
            "fill_ratio": self.fill_ratio,
        }


class SpaceSaving(Transformer):
    """Space-Saving heavy hitters (Metwally et al. 2005).

    Keeps ``k`` counters. On a new item:

    - if the item is already tracked, increment its counter;
    - if a slot is free, allocate a new one with count 1;
    - otherwise replace the counter with the smallest count ``m`` by
      ``(item, m + 1)``.

    The guarantee: the reported count of the true k-th most frequent
    item is at least its true count minus ``n / k``.

    Args:
        k: number of counters. Default ``10``.
        feature: dict key that holds the item. ``None`` reads the
            smallest key.

    Examples:
        >>> from dense_armor.utility.stats.sketches import SpaceSaving
        >>> s = SpaceSaving(k=3)
        >>> for w in ["a"] * 10 + ["b"] * 5 + ["c"] * 2 + ["d"] * 1:
        ...     _ = s.learn_one({"w": w})
        >>> s.top(1)[0][0]
        'a'

    References:
        Metwally, A., Agrawal, D., El Abbadi, A. (2005). ICDE.
    """

    def __init__(
        self,
        k: int = 10,
        feature: str | None = None,
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        self.feature = feature
        self.counters_: dict[Any, int] = {}
        self.n_ = 0

    def _key(self, x: dict) -> str:
        return min(x) if self.feature is None else self.feature

    def learn_one(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "SpaceSaving":
        """Insert one item."""
        self._time_step(t)
        item = x[self._key(x)]
        self.n_ += 1
        if item in self.counters_:
            self.counters_[item] += 1
            return self
        if len(self.counters_) < self.k:
            self.counters_[item] = 1
            return self
        min_item = min(self.counters_, key=lambda i: self.counters_[i])
        min_count = self.counters_[min_item]
        del self.counters_[min_item]
        self.counters_[item] = min_count + 1
        return self

    def update(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "SpaceSaving":
        return self.learn_one(x, y=y, t=t)

    def estimate(self, item: Any) -> int:
        """Estimated frequency of ``item``, or 0 if not tracked."""
        return self.counters_.get(item, 0)

    def top(self, n: int | None = None) -> list[tuple[Any, int]]:
        """Tracked items sorted by decreasing count.

        Args:
            n: return only the first ``n``. ``None`` returns all
                tracked items.
        """
        items = sorted(
            self.counters_.items(), key=lambda kv: kv[1], reverse=True
        )
        return items if n is None else items[:n]

    def merge(self, other: "SpaceSaving") -> "SpaceSaving":
        """Combine two Space-Saving sketches.

        Merging two Space-Saving summaries does not preserve the same
        error guarantee as a single pass over the union; use it to
        sketch totals, not to reason about exact heavy hitters.

        Args:
            other: another ``SpaceSaving`` with the same ``k``.

        Returns:
            A new ``SpaceSaving`` built by replaying the counters of
            both inputs in decreasing count order.

        Raises:
            TypeError: if ``other`` is not a ``SpaceSaving``.
            ValueError: if ``k`` values differ.
        """
        if not isinstance(other, SpaceSaving):
            raise TypeError(
                f"merge expects SpaceSaving, got {type(other).__name__}"
            )
        if self.k != other.k:
            raise ValueError("merge requires the same k")
        out = SpaceSaving(k=self.k, feature=self.feature)
        out.n_ = self.n_ + other.n_
        combined: dict[Any, int] = {}
        for item, c in self.counters_.items():
            combined[item] = combined.get(item, 0) + c
        for item, c in other.counters_.items():
            combined[item] = combined.get(item, 0) + c
        for item, c in sorted(
            combined.items(), key=lambda kv: kv[1], reverse=True
        )[: self.k]:
            out.counters_[item] = c
        return out

    @property
    def count(self) -> int:
        return self.n_

    @property
    def n_tracked(self) -> int:
        return len(self.counters_)

    def transform_one(self, x: dict, t: float | None = None) -> dict:
        item = x[self._key(x)]
        return {
            "count": self.n_,
            "estimate": self.estimate(item),
            "n_tracked": self.n_tracked,
            "top": self.top(1),
        }
