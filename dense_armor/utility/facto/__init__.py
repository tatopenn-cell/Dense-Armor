"""Factorization machines: FM (Rendle 2010), FFM (Juan 2016), FwFM (Pan 2018), HOFM."""

from dense_armor.utility.facto.ffm import FFMClassifier, FFMRegressor
from dense_armor.utility.facto.fm import FMClassifier, FMRegressor
from dense_armor.utility.facto.fwfm import FwFMClassifier, FwFMRegressor

__all__ = [
    "FFMClassifier",
    "FFMRegressor",
    "FMClassifier",
    "FMRegressor",
    "FwFMClassifier",
    "FwFMRegressor",
]
