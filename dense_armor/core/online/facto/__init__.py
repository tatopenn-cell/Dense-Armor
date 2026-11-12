"""Factorization machines: FM (Rendle 2010), FFM (Juan 2016), FwFM (Pan 2018), HOFM (Blondel 2016)."""

from dense_armor.core.online.facto.ffm import FFMClassifier, FFMRegressor
from dense_armor.core.online.facto.fm import FMClassifier, FMRegressor
from dense_armor.core.online.facto.fwfm import FwFMClassifier, FwFMRegressor
from dense_armor.core.online.facto.hofm import HOFMClassifier, HOFMRegressor

__all__ = [
    "FFMClassifier",
    "FFMRegressor",
    "FMClassifier",
    "FMRegressor",
    "FwFMClassifier",
    "FwFMRegressor",
    "HOFMClassifier",
    "HOFMRegressor",
]
