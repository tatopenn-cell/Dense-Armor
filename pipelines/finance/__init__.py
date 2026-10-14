"""Finance pipelines — four standard use cases on stock-market data.

Every class in this package consumes a stream of prices and volumes
(or a stream of orders) and returns a structured verdict. Four cases
are covered, one per file:

- :class:`~pipelines.finance.sorveglianza.FinancePipeline`
  Watch a market index in real time. Five Dense-Armor tools run in
  sequence: Armatura cleans the tick, CUSUM watches the trend, LODA
  and Half-Space Trees score the day, Orca dampens oversized orders.

- :class:`~pipelines.finance.rischio.RiskPipeline`
  Daily risk snapshot. DDSketch keeps the tail, CUSUM watches the
  volatility. Reports VaR at 99% and 99.9% plus an alarm on the
  volatility trend.

- :class:`~pipelines.finance.esecuzione.ExecutionPipeline`
  Decide whether an order leaves and how big. Size check against a
  rolling mean, then Orca on the suspicious orders only.

- :class:`~pipelines.finance.anomalia_titolo.SingleTickerPipeline`
  Single-ticker strangeness. LODA and Half-Space Trees on the
  (return, volume) pair of one stock.

Examples:
    >>> from pipelines.finance import FinancePipeline
    >>> p = FinancePipeline(range_ret=(-0.05, 0.05),
    ...                     range_vol=(0.5, 2.0), seed=0)
    >>> e = p.process_tick(100.0, 1.0)
    >>> e.tick
    0
    >>> e.discarded_by_armatura
    False
"""

from pipelines.finance.online_analysis import OnlineAnalysis, OnlineResult
from pipelines.finance.static_analysis import StaticAnalysis, StaticResult
from pipelines.finance.single_ticker import SingleTickerPipeline
from pipelines.finance.execution import ExecutionPipeline, OrderResult
from pipelines.finance.risk import RiskResult, RiskPipeline
from pipelines.finance.surveillance import FinancePipeline, TickResult

__all__ = [
    "OnlineAnalysis",
    "StaticAnalysis",
    "SingleTickerPipeline",
    "ExecutionPipeline",
    "OrderResult",
    "RiskResult",
    "FinancePipeline",
    "RiskPipeline",
    "OnlineResult",
    "StaticResult",
    "TickResult",
]
