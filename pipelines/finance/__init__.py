"""Finance pipelines — four standard use cases on stock-market data.

Sorveglianza:      FinanzaPipeline        five tools watching a market index
Esecuzione:        EsecuzionePipeline     decide whether an order leaves, and how big
Rischio:           RischioPipeline        daily VaR and volatility alarm
AnomaliaTitolo:    AnomaliaTitoloPipeline single-ticker strangeness
"""

from pipelines.finance.anomalia_titolo import AnomaliaTitoloPipeline
from pipelines.finance.esecuzione import EsecuzionePipeline, EsitoOrdine
from pipelines.finance.rischio import EsitoRischio, RischioPipeline
from pipelines.finance.sorveglianza import FinanzaPipeline, TickEsito

__all__ = [
    "AnomaliaTitoloPipeline",
    "EsecuzionePipeline",
    "EsitoOrdine",
    "EsitoRischio",
    "FinanzaPipeline",
    "RischioPipeline",
    "TickEsito",
]
