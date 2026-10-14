"""Finance pipelines — four standard use cases on stock-market data.

Every class in this package consumes a stream of prices and volumes
(or a stream of orders) and returns a structured verdict. Four cases
are covered, one per file:

- :class:`~pipelines.finance.sorveglianza.FinanzaPipeline`
  Watch a market index in real time. Five Dense-Armor tools run in
  sequence: Armatura cleans the tick, CUSUM watches the trend, LODA
  and Half-Space Trees score the day, Orca dampens oversized orders.

- :class:`~pipelines.finance.rischio.RischioPipeline`
  Daily risk snapshot. DDSketch keeps the tail, CUSUM watches the
  volatility. Reports VaR at 99% and 99.9% plus an alarm on the
  volatility trend.

- :class:`~pipelines.finance.esecuzione.EsecuzionePipeline`
  Decide whether an order leaves and how big. Size check against a
  rolling mean, then Orca on the suspicious orders only.

- :class:`~pipelines.finance.anomalia_titolo.AnomaliaTitoloPipeline`
  Single-ticker strangeness. LODA and Half-Space Trees on the
  (return, volume) pair of one stock.

Examples:
    >>> from pipelines.finance import FinanzaPipeline
    >>> p = FinanzaPipeline(range_ret=(-0.05, 0.05),
    ...                     range_vol=(0.5, 2.0), seed=0)
    >>> e = p.process_tick(100.0, 1.0)
    >>> e.tick
    0
    >>> e.scartato_da_armatura
    False
"""

from pipelines.finance.analisi_online import AnalisiOnline, RisultatoOnline
from pipelines.finance.analisi_statica import AnalisiStatica, RisultatoStatico
from pipelines.finance.anomalia_titolo import AnomaliaTitoloPipeline
from pipelines.finance.esecuzione import EsecuzionePipeline, EsitoOrdine
from pipelines.finance.rischio import EsitoRischio, RischioPipeline
from pipelines.finance.sorveglianza import FinanzaPipeline, TickEsito

__all__ = [
    "AnalisiOnline",
    "AnalisiStatica",
    "AnomaliaTitoloPipeline",
    "EsecuzionePipeline",
    "EsitoOrdine",
    "EsitoRischio",
    "FinanzaPipeline",
    "RischioPipeline",
    "RisultatoOnline",
    "RisultatoStatico",
    "TickEsito",
]
