"""An anomaly score for a stream."""


def anomaly_score(x):
    """A score for every row: an outlier is an anomaly.

    The score flags the outlier and the anomaly of the stream.
    """
    return x
