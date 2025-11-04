"""
Module 5: Technical Indicators
Comprehensive technical indicator calculations for cryptocurrency analysis
Extracts and organizes all indicator logic from the original analyzer
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Professional-grade technical indicator calculator
    Supports 20+ indicators across multiple timeframes
    """

    def __init__(self):
        """Initialize technical indicators calculator"""
        self.indicator_cache = {}

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for a DataFrame

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with all technical indicators added
        """
        if df is None or len(df) < 30:
            logger.warning("Insufficient data for technical indicators")
            return df

        try:
            df = df.copy()

            # Calculate each indicator group
            df = self._calculate_moving_averages(df)
            df = self._calculate_macd(df)
            df = self._calculate_rsi(df)
            df = self._calculate_stochastic(df)
            df = self._calculate_bollinger_bands(df)
            df = self._calculate_atr(df)
            df = self._calculate_cci(df)
            df = self._calculate_williams_r(df)
            df = self._calculate_ultimate_oscillator(df)
            df = self._calculate_momentum(df)
            df = self._calculate_adx(df)
            df = self._calculate_volume_indicators(df)
            df = self._calculate_trend_indicators(df)
            df = self._calculate_vwap(df)
            df = self._calculate_ma_crossovers(df)
            df = self._calculate_ichimoku(df)

            return df

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}", exc_info=True)
            return df

    def _calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Simple and Exponential Moving Averages"""
        try:
            # Simple Moving Averages (SMA)
            for period in [9, 12, 13, 26, 50, 100, 200]:
                if len(df) >= period:
                    df[f'sma{period}'] = df['close'].rolling(window=period).mean()

            # Exponential Moving Averages (EMA)
            for period in [9, 12, 13, 26, 50, 100, 200]:
                if len(df) >= period:
                    df[f'ema{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        except Exception as e:
            logger.error(f"Error calculating moving averages: {e}")

        return df

    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            if len(df) >= 26:
                # MACD Line = 12-period EMA - 26-period EMA
                ema12 = df['close'].ewm(span=12, adjust=False).mean()
                ema26 = df['close'].ewm(span=26, adjust=False).mean()
                df['macd'] = ema12 - ema26

                # Signal Line = 9-period EMA of MACD
                df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

                # Histogram = MACD - Signal
                df['macd_hist'] = df['macd'] - df['macd_signal']

        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")

        return df

    def _calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI (Relative Strength Index) for multiple periods"""
        try:
            for period in [9, 14, 25]:
                if len(df) >= period + 1:
                    delta = df['close'].diff()
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)

                    avg_gain = gain.rolling(window=period).mean()
                    avg_loss = loss.rolling(window=period).mean()

                    # Calculate RS and RSI
                    rs = avg_gain / avg_loss.replace(0, 1e-10)  # Avoid division by zero
                    df[f'rsi{period}'] = 100 - (100 / (1 + rs))

        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")

        return df

    def _calculate_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Stochastic Oscillator and Stochastic RSI"""
        try:
            # Standard Stochastic Oscillator
            if len(df) >= 14:
                low_14 = df['low'].rolling(window=14).min()
                high_14 = df['high'].rolling(window=14).max()

                df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14).replace(0, 1e-10))
                df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

            # Stochastic RSI
            if 'rsi14' in df.columns:
                rsi_14 = df['rsi14']
                rsi_min = rsi_14.rolling(window=14).min()
                rsi_max = rsi_14.rolling(window=14).max()

                stoch_rsi_k = 100 * ((rsi_14 - rsi_min) / (rsi_max - rsi_min).replace(0, 1e-10))
                df['stoch_rsi_k'] = stoch_rsi_k
                df['stoch_rsi_d'] = stoch_rsi_k.rolling(window=3).mean()

        except Exception as e:
            logger.error(f"Error calculating Stochastic: {e}")

        return df

    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        try:
            if len(df) >= 20:
                df['bb_middle'] = df['close'].rolling(window=20).mean()
                df['bb_std'] = df['close'].rolling(window=20).std()
                df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
                df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
                df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'].replace(0, 1e-10)

                # Bollinger Band crossover signals
                df['price_cross_upper_bb'] = (
                    (df['close'] > df['bb_upper']) & 
                    (df['close'].shift(1) <= df['bb_upper'].shift(1))
                ).astype(int)

                df['price_cross_lower_bb'] = (
                    (df['close'] < df['bb_lower']) & 
                    (df['close'].shift(1) >= df['bb_lower'].shift(1))
                ).astype(int)

        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")

        return df

    def _calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average True Range (ATR)"""
        try:
            if len(df) >= 14:
                high_low = df['high'] - df['low']
                high_close = (df['high'] - df['close'].shift()).abs()
                low_close = (df['low'] - df['close'].shift()).abs()

                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                df['atr'] = true_range.rolling(window=14).mean()
                df['atr_percent'] = (df['atr'] / df['close'].replace(0, 1e-10)) * 100

        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")

        return df

    def _calculate_cci(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Commodity Channel Index (CCI)"""
        try:
            if len(df) >= 20:
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                tp_sma20 = typical_price.rolling(window=20).mean()
                tp_deviation = (typical_price - tp_sma20).abs()
                tp_deviation_mean = tp_deviation.rolling(window=20).mean()

                df['cci20'] = (typical_price - tp_sma20) / (0.015 * tp_deviation_mean.replace(0, 1e-10))

        except Exception as e:
            logger.error(f"Error calculating CCI: {e}")

        return df

    def _calculate_williams_r(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Williams %R"""
        try:
            if len(df) >= 14:
                high_14 = df['high'].rolling(window=14).max()
                low_14 = df['low'].rolling(window=14).min()

                df['williams_r'] = -100 * ((high_14 - df['close']) / (high_14 - low_14).replace(0, 1e-10))

        except Exception as e:
            logger.error(f"Error calculating Williams %R: {e}")

        return df

    def _calculate_ultimate_oscillator(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Ultimate Oscillator"""
        try:
            if len(df) >= 28:
                # Buying Pressure = Close - True Low
                bp = df['close'] - pd.concat([df['low'], df['close'].shift(1)], axis=1).min(axis=1)

                # True Range
                tr = pd.concat([
                    df['high'] - df['low'],
                    (df['high'] - df['close'].shift(1)).abs(),
                    (df['low'] - df['close'].shift(1)).abs()
                ], axis=1).max(axis=1)

                # Calculate averages for 7, 14, and 28 periods
                avg7 = bp.rolling(window=7).sum() / tr.rolling(window=7).sum().replace(0, 1e-10)
                avg14 = bp.rolling(window=14).sum() / tr.rolling(window=14).sum().replace(0, 1e-10)
                avg28 = bp.rolling(window=28).sum() / tr.rolling(window=28).sum().replace(0, 1e-10)

                df['uo'] = 100 * ((4 * avg7) + (2 * avg14) + avg28) / 7

        except Exception as e:
            logger.error(f"Error calculating Ultimate Oscillator: {e}")

        return df

    def _calculate_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Momentum indicator"""
        try:
            df['mom'] = df['close'].diff(14)
        except Exception as e:
            logger.error(f"Error calculating Momentum: {e}")

        return df

    def _calculate_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average Directional Index (ADX)"""
        try:
            if len(df) >= 14:
                plus_dm = df['high'].diff()
                minus_dm = -df['low'].diff()

                plus_dm[plus_dm < 0] = 0
                minus_dm[minus_dm < 0] = 0

                # Conditions for +DM and -DM
                cond1 = (plus_dm > minus_dm) & (plus_dm > 0)
                plus_dm[~cond1] = 0

                cond2 = (minus_dm > plus_dm) & (minus_dm > 0)
                minus_dm[~cond2] = 0

                # True Range
                tr = pd.concat([
                    (df['high'] - df['low']).abs(),
                    (df['high'] - df['close'].shift()).abs(),
                    (df['low'] - df['close'].shift()).abs()
                ], axis=1).max(axis=1)

                # Directional Indicators
                plus_di14 = 100 * (plus_dm.rolling(window=14).sum() / tr.rolling(window=14).sum().replace(0, 1e-10))
                minus_di14 = 100 * (minus_dm.rolling(window=14).sum() / tr.rolling(window=14).sum().replace(0, 1e-10))

                # DX and ADX
                dx = 100 * (abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14).replace(0, 1e-10))
                df['adx'] = dx.rolling(window=14).mean()

                df['plus_di'] = plus_di14
                df['minus_di'] = minus_di14

        except Exception as e:
            logger.error(f"Error calculating ADX: {e}")

        return df

    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators"""
        try:
            # On Balance Volume (OBV)
            obv = np.zeros(len(df))
            for i in range(1, len(df)):
                if df['close'].iloc[i] > df['close'].iloc[i-1]:
                    obv[i] = obv[i-1] + df['volume'].iloc[i]
                elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                    obv[i] = obv[i-1] - df['volume'].iloc[i]
                else:
                    obv[i] = obv[i-1]
            df['obv'] = obv

            # Volume indicators
            df['volume_sma20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma20'].replace(0, 1e-10)
            df['unusual_volume'] = (df['volume_ratio'] > 2).astype(int)
            df['rvol'] = df['volume'] / df['volume'].rolling(20).mean().replace(0, 1e-10)

        except Exception as e:
            logger.error(f"Error calculating volume indicators: {e}")

        return df

    def _calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend detection indicators"""
        try:
            if all(col in df.columns for col in ['ema9', 'ema50', 'ema200']):
                # Short-term trend
                df['short_term_trend'] = np.where(
                    df['ema9'] > df['ema50'], 1,
                    np.where(df['ema9'] < df['ema50'], -1, 0)
                )

                # Medium-term trend
                df['medium_term_trend'] = np.where(
                    df['ema50'] > df['ema200'], 1,
                    np.where(df['ema50'] < df['ema200'], -1, 0)
                )

                # Trend strength
                df['trend_strength'] = df['short_term_trend'] + df['medium_term_trend']

        except Exception as e:
            logger.error(f"Error calculating trend indicators: {e}")

        return df

    def _calculate_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Volume-Weighted Average Price (VWAP)"""
        try:
            df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum().replace(0, 1e-10)
        except Exception as e:
            logger.error(f"Error calculating VWAP: {e}")

        return df

    def _calculate_ma_crossovers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect moving average crossovers"""
        try:
            # SMA crossovers
            for fast, slow in [(5, 10), (10, 20), (20, 50), (50, 200)]:
                fast_col = f'sma{fast}'
                slow_col = f'sma{slow}'
                crossover_col = f'sma{fast}_{slow}_crossover'

                if fast_col in df.columns and slow_col in df.columns:
                    df[crossover_col] = 0

                    # Bullish crossover
                    bullish = (df[fast_col].shift(1) <= df[slow_col].shift(1)) & (df[fast_col] > df[slow_col])
                    df.loc[bullish, crossover_col] = 1

                    # Bearish crossover
                    bearish = (df[fast_col].shift(1) >= df[slow_col].shift(1)) & (df[fast_col] < df[slow_col])
                    df.loc[bearish, crossover_col] = -1

            # EMA crossovers
            for fast, slow in [(9, 12), (12, 26), (26, 50), (50, 200)]:
                fast_col = f'ema{fast}'
                slow_col = f'ema{slow}'
                crossover_col = f'ema{fast}_{slow}_crossover'

                if fast_col in df.columns and slow_col in df.columns:
                    df[crossover_col] = 0

                    bullish = (df[fast_col].shift(1) <= df[slow_col].shift(1)) & (df[fast_col] > df[slow_col])
                    df.loc[bullish, crossover_col] = 1

                    bearish = (df[fast_col].shift(1) >= df[slow_col].shift(1)) & (df[fast_col] < df[slow_col])
                    df.loc[bearish, crossover_col] = -1

        except Exception as e:
            logger.error(f"Error calculating MA crossovers: {e}")

        return df

    def _calculate_ichimoku(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Ichimoku Cloud components"""
        try:
            # Tenkan-sen (Conversion Line): 9-period high-low average
            high_9 = df['high'].rolling(window=9).max()
            low_9 = df['low'].rolling(window=9).min()
            df['tenkan_sen'] = (high_9 + low_9) / 2

            # Kijun-sen (Base Line): 26-period high-low average
            high_26 = df['high'].rolling(window=26).max()
            low_26 = df['low'].rolling(window=26).min()
            df['kijun_sen'] = (high_26 + low_26) / 2

            # Senkou Span A (Leading Span A): Average of Tenkan-sen and Kijun-sen, shifted 26 periods ahead
            df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)

            # Senkou Span B (Leading Span B): 52-period high-low average, shifted 26 periods ahead
            high_52 = df['high'].rolling(window=52).max()
            low_52 = df['low'].rolling(window=52).min()
            df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)

            # Chikou Span (Lagging Span): Close price shifted 26 periods back
            df['chikou_span'] = df['close'].shift(-26)

        except Exception as e:
            logger.error(f"Error calculating Ichimoku: {e}")

        return df
