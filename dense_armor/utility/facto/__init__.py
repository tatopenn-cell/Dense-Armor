"""Factorization machines: FM (Rendle 2010), FFM (Juan 2016), FwFM, HOFM."""

from dense_armor.utility.facto.ffm import FFMClassifier, FFMRegressor
from dense_armor.utility.facto.fm import FMClassifier, FMRegressor

__all__ = ["FFMClassifier", "FFMRegressor", "FMClassifier", "FMRegressor"]
