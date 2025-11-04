"""
Module 2: Configuration
Central configuration management for crypto analyzer.
"""

import os
from typing import Optional, Dict, List

class Config:
    """
    Configuration class for CryptoTradeAnalyzer.
    Manages all settings including API keys, symbols, timeframes, and paths.
    """
    
    def __init__(self, min_success_rate: int = 70, api_key: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            min_success_rate: Minimum quality score threshold (0-100)
            api_key: CryptoCompare API key (optional, will try environment variable)
        """
        # API Configuration
        self.api_key = api_key or os.getenv('CRYPTO_COMPARE_API_KEY')
        
        if not self.api_key:
            # Fallback to hardcoded key (matches original temp7.py behavior)
            self.api_key = "330301d4e10a22b321623bfbe4e52ce67e010df4ebed49106ed85e73ff650270"  # mh.moosaei
        
        # Quality threshold
        self.min_success_rate = min_success_rate
        
        # Symbols to analyze (from original code)
        self.symbols = [
            "BTC", "ETH", "ETC", "XRP", "SOL", "BNB", "DOGE", "ADA", "EOS", "TRX",
            "AVAX", "LINK", "XLM", "HBAR", "SHIB", "TON", "DOT", "LTC", "BCH", "UNI",
            "S", "AXS", "NEAR", "APT", "AAVE", "POL", "XMR", "RENDER", "FIL", "PEPE",
            "GMT", "ATOM", "SAND", "FLOKI", "APE", "CAKE", "CATI", "CVX", "XAUT", "CHZ",
            "CRV", "LDO", "DYDX", "API3", "ONE", "STORJ", "SNT", "ZRX", "SLP", "T",
            "GRASS", "ARB", "WLD", "X", "WIF", "CELR", "FET", "PENGU", "ALGO", "VET",
            "OP", "INJ", "ICP", "SEI", "SUI", "ENA", "JUP", "PUMP", "TURBO", "MOG",
            "HYPE", "PYTH", "FORM"
        ]
        
        # API URLs
        self.base_url = "https://min-api.cryptocompare.com/data"
        self.alternative_base_url = "https://api.coingecko.com/api/v3"
        
        # Timeframe mappings (from original code)
        self.timeframes = {
            "1m": "histominute",
            "5m": "histominute",
            "15m": "histominute",
            "30m": "histominute",
            "1h": "histohour",
            "4h": "histohour",
            "12h": "histohour",
            "1d": "histoday",
            "1w": "histoday"
        }
        
        # Timeframe multipliers
        self.timeframe_multipliers = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 1, "4h": 4, "12h": 12,
            "1d": 1, "1w": 7
        }
        
        # File paths
        self.signals_csv_path = "crypto_signals.csv"
        
        # Rate limiting
        self.api_call_limit = 15  # Calls per minute
        
        # Analysis parameters
        self.default_limit = 200  # Default candles to fetch
        self.min_data_points = 50  # Minimum data points for analysis
        
        # Pattern success rates (from original code) - using lowercase keys to match pattern detection logic
        self.pattern_success_rates = {
            "double_bottom": 69,
            "double_top": 56,
            "head_and_shoulders": 69,
            "inverse_head_and_shoulders": 71,
            "ascending_triangle": 55,
            "descending_triangle": 57,
            "symmetrical_triangle": 49,
            "falling_wedge": 53,
            "rising_wedge": 52,
            "bull_flag": 55,
            "bear_flag": 55,
            "channel_up": 60,
            "channel_down": 59
        }


# Module-level configuration dictionaries for backward compatibility
api_config = {
    'base_url': 'https://min-api.cryptocompare.com/data',
    'alternative_base_url': 'https://api.coingecko.com/api/v3',
    'api_key': os.getenv('CRYPTO_COMPARE_API_KEY', '330301d4e10a22b321623bfbe4e52ce67e010df4ebed49106ed85e73ff650270'),
    'rate_limit': 15,
    'timeout': 30,
    'retries': 3
}

symbol_config = {
    'symbols': [
        "BTC", "ETH", "ETC", "XRP", "SOL", "BNB", "DOGE", "ADA", "EOS", "TRX",
        "AVAX", "LINK", "XLM", "HBAR", "SHIB", "TON", "DOT", "LTC", "BCH", "UNI",
        "S", "AXS", "NEAR", "APT", "AAVE", "POL", "XMR", "RENDER", "FIL", "PEPE",
        "GMT", "ATOM", "SAND", "FLOKI", "APE", "CAKE", "CATI", "CVX", "XAUT", "CHZ",
        "CRV", "LDO", "DYDX", "API3", "ONE", "STORJ", "SNT", "ZRX", "SLP", "T",
        "GRASS", "ARB", "WLD", "X", "WIF", "CELR", "FET", "PENGU", "ALGO", "VET",
        "OP", "INJ", "ICP", "SEI", "SUI", "ENA", "JUP", "PUMP", "TURBO", "MOG",
        "HYPE", "PYTH", "FORM"
    ],
    'base_currency': 'USDT'
}

timeframe_config = {
    "1m": "histominute",
    "5m": "histominute",
    "15m": "histominute",
    "30m": "histominute",
    "1h": "histohour",
    "4h": "histohour",
    "12h": "histohour",
    "1d": "histoday",
    "1w": "histoday"
}

pattern_config = {
    "double_bottom": 69,
    "double_top": 56,
    "head_and_shoulders": 69,
    "inverse_head_and_shoulders": 71,
    "ascending_triangle": 55,
    "descending_triangle": 57,
    "symmetrical_triangle": 49,
    "falling_wedge": 53,
    "rising_wedge": 52,
    "bull_flag": 55,
    "bear_flag": 55,
    "channel_up": 60,
    "channel_down": 59
}