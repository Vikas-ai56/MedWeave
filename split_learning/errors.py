class SplitLearningError(Exception):
    """Base class for errors raised by the split-learning engine."""


class ShapeError(SplitLearningError, ValueError):
    """A tensor does not have the shape or dtype the split model expects."""


class CheckpointError(SplitLearningError):
    """A checkpoint is unreadable or does not match the model it is loaded into."""


class ProtocolError(SplitLearningError):
    """An exchange call arrived out of order, for the wrong batch or Backbone version."""
