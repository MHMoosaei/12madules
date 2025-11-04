"""
Module 3: Utility Functions

Common utility functions used throughout the analyzer.

NOTE: These functions extract inline logic from the original temp7.py script
into reusable utilities. The logic is identical, but now centralized for
maintainability. For exact backward compatibility, these functions implement
the same algorithms used inline in temp7.py.
"""


import time
from functools import wraps
from typing import Any, Callable
import pandas as pd
import numpy as np


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """
    Decorator to retry function on failure.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator


def safe_float_conversion(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers."""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values."""
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100


def normalize_symbol(symbol: str) -> str:
    """Normalize cryptocurrency symbol."""
    return symbol.upper().strip()


def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Validate that dataframe has required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Returns:
        True if all columns present
    """
    if df is None or df.empty:
        return False
    return all(col in df.columns for col in required_columns)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()


# def find_peaks_troughs(series: pd.Series, order: int = 5) -> tuple:
#     """
#     Find peaks and troughs in a price series.

#     Args:
#         series: Price series
#         order: Window size for peak detection

#     Returns:
#         Tuple of (peaks, troughs) as lists of indices
#     """
#     from scipy.signal import argrelextrema

#     peaks = argrelextrema(series.values, np.greater, order=order)[0]
#     troughs = argrelextrema(series.values, np.less, order=order)[0]

#     return peaks.tolist(), troughs.tolist()

def find_peaks_troughs(series: pd.Series, distance: int = 5, prominence: float = None) -> tuple:
    """
    Find peaks and troughs in a price series using scipy.signal.find_peaks.
    
    Args:
        series: Price series
        distance: Minimum distance between peaks (matches temp7.py behavior)
        prominence: Minimum prominence of peaks (optional)
    
    Returns:
        Tuple of (peaks, troughs) as lists of indices
    """
    from scipy.signal import find_peaks
    
    # Find peaks (local maxima)
    peaks, _ = find_peaks(series.values, distance=distance, prominence=prominence)
    
    # Find troughs (local minima) by inverting the series
    troughs, _ = find_peaks(-series.values, distance=distance, prominence=prominence)
    
    return peaks.tolist(), troughs.tolist()

