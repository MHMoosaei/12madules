"""
Module 4: API Client
Handles all API interactions with CryptoCompare and fallback providers.
"""

import time
import requests
import pandas as pd
from typing import Optional
from datetime import datetime

from exceptions import APIError, RateLimitError, InsufficientDataError
from config import api_config, symbol_config, timeframe_config


class APIClient:
    """
    API client for fetching cryptocurrency data from CryptoCompare.
    Includes fallback to CoinGecko API if primary fails.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize API client.

        Args:
            api_key: CryptoCompare API key (optional, uses config if not provided)
        """
        self.api_key = api_key or api_config['api_key']
        self.base_url = api_config['base_url']
        self.alternative_base_url = api_config['alternative_base_url']
        self.rate_limit = api_config['rate_limit']
        self.timeout = api_config.get('timeout', 10)
        self.retries = api_config.get('retries', 3)

        # Rate limiting
        self.last_api_call = 0
        self.api_call_limit = self.rate_limit

        # Timeframe mappings
        self.timeframes = timeframe_config
        self.timeframe_multipliers = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 1, "4h": 4, "12h": 12,
            "1d": 1, "1w": 7
        }

    def rate_limit_api_call(self):
        """Implement rate limiting for API calls."""
        current_time = time.time()
        elapsed = current_time - self.last_api_call

        if elapsed < 0.5:  # At least 0.5 seconds between calls
            time.sleep(0.5 - elapsed)

        self.last_api_call = time.time()

    def get_historical_data(self, 
                           symbol: str, 
                           timeframe: str, 
                           limit: int = 200,
                           end_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """
        Fetch historical price data from CryptoCompare API.

        Args:
            symbol: Cryptocurrency symbol (e.g., "BTC")
            timeframe: Timeframe (e.g., "1d", "4h", "1h")
            limit: Number of data points to retrieve
            end_date: If specified, fetch data up to this date

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        if timeframe not in self.timeframes:
            raise ValueError(f"Invalid timeframe {timeframe}. Valid: {list(self.timeframes.keys())}")

        max_retries = self.retries
        retry_delay = 1  # seconds

        for attempt in range(max_retries):
            try:
                # Apply rate limiting
                self.rate_limit_api_call()

                # Get timeframe category and multiplier
                tf_category = self.timeframes.get(timeframe, 'histoday')
                tf_multiplier = self.timeframe_multipliers.get(timeframe, 1)

                # Build URL and parameters
                url = f"{self.base_url}/{tf_category}"
                params = {
                    'fsym': symbol,
                    'tsym': 'USDT',
                    'limit': limit,
                    'api_key': self.api_key
                }

                # Add end date if specified
                if end_date:
                    params['toTs'] = int(end_date.timestamp())

                # Add aggregation for non-standard timeframes
                if tf_category == 'histominute' and tf_multiplier > 1:
                    params['aggregate'] = tf_multiplier
                elif tf_category == 'histohour' and tf_multiplier > 1:
                    params['aggregate'] = tf_multiplier
                elif tf_category == 'histoday' and tf_multiplier > 1:
                    params['aggregate'] = tf_multiplier

                # Make API request
                response = requests.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()

                data = response.json()

                # Check for API errors
                if data.get('Response') == 'Error':
                    print(f"API error for {symbol}: {data.get('Message')}")
                    return self._fallback_to_alternative_api(symbol, timeframe, limit)

                if 'Data' not in data or not data['Data']:
                    print(f"No data found for {symbol}")
                    return self._fallback_to_alternative_api(symbol, timeframe, limit)

                # Convert to DataFrame
                df = pd.DataFrame(data['Data'])
                df['time'] = pd.to_datetime(df['time'], unit='s')

                # Ensure required columns exist
                required_columns = ['open', 'high', 'low', 'close', 'volumefrom', 'volumeto']
                for col in required_columns:
                    if col not in df.columns:
                        if col in ['volumefrom', 'volumeto']:
                            df[col] = 0
                        else:
                            print(f"Missing required column {col} for {symbol}")
                            return None

                # Rename volume column
                df['volume'] = df['volumefrom']

                # Filter out zero values
                df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]

                if len(df) == 0:
                    print(f"All data filtered out for {symbol}")
                    return None

                return df

            except requests.exceptions.Timeout:
                print(f"Timeout fetching {symbol} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue

            except requests.exceptions.RequestException as e:
                print(f"Network error fetching {symbol}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue

            except Exception as e:
                print(f"Unexpected error fetching {symbol}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue

        # All retries failed
        print(f"All retries failed for {symbol}, attempting fallback API")
        return self._fallback_to_alternative_api(symbol, timeframe, limit)

    def _fallback_to_alternative_api(self, 
                                    symbol: str, 
                                    timeframe: str, 
                                    limit: int) -> Optional[pd.DataFrame]:
        """
        Fallback to CoinGecko API if primary fails.

        Args:
            symbol: Cryptocurrency symbol
            timeframe: Timeframe
            limit: Number of data points

        Returns:
            DataFrame or None
        """
        try:
            print(f"Attempting to fetch {symbol} data from alternative API...")

            # Convert timeframe to days
            days = 30  # Default
            if timeframe == '1d':
                days = limit
            elif timeframe in ['1h', '4h']:
                days = min(90, limit // 24 + 1)  # CoinGecko hourly limited to 90 days

            # CoinGecko uses lowercase symbols
            coin_id = symbol.lower()

            # Build URL
            url = f"{self.alternative_base_url}/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'daily' if timeframe == '1d' else 'hourly'
            }

            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            if 'prices' not in data:
                print(f"Alternative API: No data found for {symbol}")
                return None

            # Convert to DataFrame
            prices = data['prices']  # [[timestamp, price], ...]
            volumes = data.get('total_volumes', [])  # [[timestamp, volume], ...]

            df = pd.DataFrame(prices, columns=['timestamp', 'close'])
            df['time'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Add volume
            if volumes:
                volume_df = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
                df = pd.merge(df, volume_df, on='timestamp')
            else:
                df['volume'] = 0

            # Simplified OHLC (using close for all)
            df['open'] = df['close']
            df['high'] = df['close']
            df['low'] = df['close']

            # Limit to requested rows
            df = df.tail(limit)

            return df

        except Exception as e:
            print(f"Alternative API failed for {symbol}: {e}")
            return None
