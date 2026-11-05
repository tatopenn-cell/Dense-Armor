"""Pre-trade limits for one order, from SEC Rule 15c3-5 (c)(1)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    """One decision on one order."""

    approved: bool
    reason: str


class PreTradeChecker:
    """Accept or reject one order with the five controls of (c)(1).

    The controls run in this order and the first one that fails gives
    the reason; an approved order has ``reason == "ok"``:

    1. ``"size"``: ``size > max_size`` (SEC 15c3-5 (c)(1)(ii), size
       parameters, PDF page 128).
    2. ``"price_band"``: ``abs(price / ref_price - 1) > price_band``
       ((c)(1)(ii), price parameters, PDF page 128).
    3. ``"duplicate"``: an approved order with the same ``account``,
       ``side``, ``size`` and ``price`` was approved in the last
       ``duplicate_window_s`` seconds ((c)(1)(ii), duplicative orders, PDF
       page 128).
    4. ``"rate"``: ``max_orders`` or more orders of the account were
       approved in the last ``window_s`` seconds ((c)(1)(ii), over a short
       period of time, PDF page 128).
    5. ``"notional"``: the open notional of the account plus
       ``size * price`` is above ``max_notional`` ((c)(1)(i), pre-set
       credit or capital thresholds, in the aggregate for each customer,
       PDF page 128).

    A rejected order changes no state. An approved order adds
    ``size * price`` to the open notional of its account. The value exactly
    at a limit is approved; one tick above is rejected. Only approved
    orders count for the rate and for the duplicates: that is our choice,
    the rule does not say it.

    Args:
        max_notional: the largest open notional an account can carry, in
            the currency of the prices; strictly positive.
        max_size: the largest size of one order, in shares; strictly
            positive.
        price_band: the largest relative distance of the price from the
            reference price, as a fraction; strictly positive.
        max_orders: the number of approved orders in ``window_s`` after
            which the account is rejected; at least 1.
        window_s: the length of the rate window, in seconds; strictly
            positive.
        duplicate_window_s: the length of the duplicate window, in
            seconds; strictly positive.

    Raises:
        ValueError: if a limit is not strictly positive.

    Examples:
        >>> c = PreTradeChecker(max_notional=1e6, max_size=10.0,
        ...                     price_band=0.01, max_orders=5,
        ...                     window_s=1.0, duplicate_window_s=0.1)
        >>> c.check("a", "buy", 10.0, 100.0, 100.0, t=0.0)
        Decision(approved=True, reason='ok')
        >>> c.check("a", "buy", 10.01, 100.0, 100.0, t=0.1).reason
        'size'
        >>> c.open_notional("a")
        1000.0

    References:
        SEC Release 34-63241, Rule 15c3-5 (c)(1)(i) and (c)(1)(ii), PDF
        page 128: pre-set credit or capital thresholds in the aggregate for
        each customer; price or size parameters, over a short period of
        time, duplicative orders.
    """

    def __init__(
        self,
        max_notional: float,
        max_size: float,
        price_band: float,
        max_orders: int,
        window_s: float,
        duplicate_window_s: float,
    ) -> None:
        for name, v in (
            ("max_notional", max_notional),
            ("max_size", max_size),
            ("price_band", price_band),
            ("window_s", window_s),
            ("duplicate_window_s", duplicate_window_s),
        ):
            if float(v) <= 0.0:
                raise ValueError(f"{name} must be strictly positive, got {v}")
        if int(max_orders) < 1:
            raise ValueError(f"max_orders must be >= 1, got {max_orders}")
        self.max_notional = float(max_notional)
        self.max_size = float(max_size)
        self.price_band = float(price_band)
        self.max_orders = int(max_orders)
        self.window_s = float(window_s)
        self.duplicate_window_s = float(duplicate_window_s)
        self._open: dict[str, float] = {}
        self._rate: dict[str, list[float]] = {}
        self._dup: dict[str, list[tuple[float, float, float, float, str]]] = {}
        self._last_t = -float("inf")

    def _evict(self, t: float) -> None:
        lo = t - self.window_s
        for k in list(self._rate.keys()):
            self._rate[k] = [u for u in self._rate[k] if u >= lo]
            if not self._rate[k]:
                del self._rate[k]
        dlo = t - self.duplicate_window_s
        for k in list(self._dup.keys()):
            self._dup[k] = [r for r in self._dup[k] if r[0] >= dlo]
            if not self._dup[k]:
                del self._dup[k]

    def check(
        self,
        account: str,
        side: str,
        size: float,
        price: float,
        ref_price: float,
        t: float,
    ) -> Decision:
        """Decide one order. A rejected order leaves the state unchanged."""
        if side not in ("buy", "sell"):
            raise ValueError(f'side must be "buy" or "sell", got {side!r}')
        if float(size) <= 0.0:
            raise ValueError(f"size must be positive, got {size}")
        if float(price) <= 0.0:
            raise ValueError(f"price must be positive, got {price}")
        if float(ref_price) <= 0.0:
            raise ValueError(f"ref_price must be positive, got {ref_price}")
        if float(t) < self._last_t:
            raise ValueError(f"t must be >= the previous t ({self._last_t}), got {t}")
        self._last_t = float(t)
        self._evict(float(t))
        if float(size) > self.max_size:
            return Decision(False, "size")
        rel = abs(float(price) / float(ref_price) - 1.0)
        if rel > self.price_band:
            return Decision(False, "price_band")
        for r in self._dup.get(account, []):
            if r[1] == float(size) and r[2] == float(price) and r[4] == side:
                return Decision(False, "duplicate")
        if len(self._rate.get(account, [])) >= self.max_orders:
            return Decision(False, "rate")
        open_now = self._open.get(account, 0.0)
        add = float(size) * float(price)
        if open_now + add > self.max_notional:
            return Decision(False, "notional")
        self._open[account] = open_now + add
        self._rate.setdefault(account, []).append(float(t))
        self._dup.setdefault(account, []).append(
            (float(t), float(size), float(price), float(ref_price), side)
        )
        return Decision(True, "ok")

    def release(self, account: str, notional: float) -> None:
        """Subtract a notional after a fill or a cancel."""
        cur = self._open.get(account, 0.0)
        if float(notional) > cur:
            raise ValueError(
                f"release of {notional} is larger than open notional {cur}"
            )
        self._open[account] = cur - float(notional)

    def open_notional(self, account: str) -> float:
        """The open notional of the account."""
        return float(self._open.get(account, 0.0))
