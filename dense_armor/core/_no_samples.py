"""The stand-in for the samples of an input that has none."""


class NoSamples:
    """What ``AudioInput`` receives in place of an audio block when the input is not sound.

    ``AudioInput`` hands the samples of the block to a primitive that only has ``update_many``.
    Text and video have no samples, so that path ends here with a clear error instead of an
    ``AttributeError`` on a string or an image.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind

    def to_mono(self) -> None:
        """Refuse: this input has no samples."""
        raise ValueError(
            f"this primitive takes samples (update_many only); {self.kind} has none"
        )
