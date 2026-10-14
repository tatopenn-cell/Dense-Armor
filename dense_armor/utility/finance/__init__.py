"""Finance pipelines — four standard use cases on stock-market data.

Sorveglianza:      FinanzaPipeline        five tools watching a market index
Esecuzione:        EsecuzionePipeline     decide whether an order leaves, and how big
Rischio:           RischioPipeline        daily VaR and volatility alarm
AnomaliaTitolo:    AnomaliaTitoloPipeline single-ticker strangeness
"""

from dense_armor.utility.finance.anomalia_titolo import AnomaliaTitoloPipeline
from dense_armor.utility.finance.esecuzione import EsecuzionePipeline, EsitoOrdine
from dense_armor.utility.finance.pipeline import FinanzaPipeline, TickEsito
from dense_armor.utility.finance.rischio import EsitoRischio, RischioPipeline

__all__ = [
    "FinanzaPipeline",
    "TickEsito",
    "EsecuzionePipeline",
    "EsitoOrdine",
    "RischioPipeline",
    "EsitoRischio",
    "AnomaliaTitoloPipeline",
]
