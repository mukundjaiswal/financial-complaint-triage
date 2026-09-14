"""Exception hierarchy."""


class ComplaintTriageError(Exception):
    """Base class for every error raised by this package."""


class ConfigurationError(ComplaintTriageError):
    """Required configuration is missing or invalid."""


class ModelError(ComplaintTriageError):
    """The local model could not be loaded or produced no output."""


class DatasetError(ComplaintTriageError):
    """A dataset is missing or malformed."""
