"""
Module 1: Custom Exceptions
Defines custom exception classes for crypto analyzer.
"""


class InsufficientDataError(Exception):
    """Raised when there is insufficient historical data for analysis."""
    pass


class APIError(Exception):
    """Raised when API requests fail."""
    pass


class RateLimitError(Exception):
    """Raised when API rate limit is exceeded."""
    pass


class InvalidTimeframeError(Exception):
    """Raised when an invalid timeframe is specified."""
    pass


class PatternDetectionError(Exception):
    """Raised when pattern detection fails."""
    pass


class CalculationError(Exception):
    """Raised when technical calculations fail."""
    pass


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass


class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass
