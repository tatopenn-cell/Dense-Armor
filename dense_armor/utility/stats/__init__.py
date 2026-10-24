"""Streaming statistics and sketches."""
from dense_armor.utility.stats.bm25 import BM25Index
from dense_armor.utility.stats.dependence import (
    Autocorrelation,
    RollingCorrelation,
    RollingCovariance,
    RunningCorrelation,
    RunningCovariance,
)
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

__all__ = [
    "Autocorrelation",
    "BM25Index",
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
    "RunningMoments",
    "RunningMomentsVector",
    "SpaceSaving",
    "TDigest",
    "cosine_similarity",
    "cosine_similarity_many",
]
