"""Streams: iterators of ``(Signal, y)`` pairs in time order."""

from dense_armor.utility.signals.stream.array import iter_array
from dense_armor.utility.signals.stream.csv import iter_csv
from dense_armor.utility.signals.stream.merge import merge_by_time
from dense_armor.utility.signals.stream.rosbag import iter_rosbag
from dense_armor.utility.signals.stream.shuffle import shuffle

__all__ = [
    "iter_array",
    "iter_csv",
    "iter_rosbag",
    "merge_by_time",
    "shuffle",
]
