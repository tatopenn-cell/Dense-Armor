"""Finance pipeline — five Dense-Armor tools on a stock-market tick stream.

Chains:
    Armatura        discard absurd ticks
    CUSUM           flag a slow change of trend
    LODA + HST      score how anomalous a day is (two votes)
    Quantile sketch size the tail risk
    Orca            dampen oversized orders before they leave
"""

from dense_armor.utility.finance.pipeline import FinanzaPipeline, TickEsito

__all__ = ["FinanzaPipeline", "TickEsito"]
