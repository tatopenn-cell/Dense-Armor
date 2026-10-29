"""Streaming statistics and sketches."""

from dense_armor.utility.stats.dependence import (
    Autocorrelation,
    RollingCorrelation,
    RollingCovariance,
    RunningCorrelation,
    RunningCovariance,
)
from dense_armor.utility.stats.mixture import OnlineGaussianMixture
from dense_armor.utility.stats.moments import (
    EWStats,
    RunningMoments,
    RunningMomentsVector,
)
from dense_armor.utility.stats.quantiles import (
    DDSketch,
    TDigest,
)
from dense_armor.utility.stats.robust import (
    RollingIQR,
    RollingMAD,
    RollingMedian,
    RollingMedianVector,
    RollingQuantile,
)
from dense_armor.utility.stats.similarity import (
    cosine_similarity,
    cosine_similarity_many,
)
from dense_armor.utility.stats.sketches import (
    BloomFilter,
    CountMinSketch,
    HyperLogLog,
    SpaceSaving,
)
from dense_armor.utility.stats.stable import stable_moments, stable_var
from dense_armor.utility.stats.student_t import StudentT

__all__ = [
    "Autocorrelation",
    "BloomFilter",
    "CountMinSketch",
    "DDSketch",
    "EWStats",
    "HyperLogLog",
    "RollingCorrelation",
    "RollingCovariance",
    "RollingIQR",
    "RollingMAD",
    "RollingMedian",
    "RollingMedianVector",
    "RollingQuantile",
    "RunningCorrelation",
    "RunningCovariance",
    "OnlineGaussianMixture",
    "RunningMoments",
    "RunningMomentsVector",
    "SpaceSaving",
    "StudentT",
    "stable_moments",
    "stable_var",
    "TDigest",
    "cosine_similarity",
    "cosine_similarity_many",
]
