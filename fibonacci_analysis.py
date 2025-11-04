"""
Module 6: Fibonacci Analysis
Multi-timeframe Fibonacci retracement and extension levels with volume confirmation
Crypto-optimized swing point detection and level clustering
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class FibonacciAnalyzer:
    """
    Professional Fibonacci analysis for cryptocurrency markets
    Implements dynamic swing detection with volume confirmation
    """

    def __init__(self):
        """Initialize Fibonacci analyzer"""
        self.fib_retracements = [0.236, 0.382, 0.5, 0.618, 0.786]
        self.fib_extensions = [1.272, 1.414, 1.618, 2.0, 2.618, 3.618]

    def calculate_fibonacci_levels(self, df: pd.DataFrame) -> List[float]:
        """
        Calculate Fibonacci retracement and extension levels with crypto-specific improvements
        Uses dynamic swing point detection and volume-weighted significance

        Args:
            df: DataFrame with OHLCV data

        Returns:
            List of significant Fibonacci levels
        """
        try:
            if len(df) < 50:
                return []

            # Multi-timeframe analysis for swing points
            lookback_periods = [21, 50, 100]  # Short, medium, long-term swings
            all_fib_levels = []

            for lookback in lookback_periods:
                if len(df) >= lookback:
                    period_df = df.tail(lookback)

                    # Find significant swing points with volume confirmation
                    swing_highs, swing_lows = self._find_volume_confirmed_swings(period_df)

                    if len(swing_highs) >= 1 and len(swing_lows) >= 1:
                        # Get most recent and significant swings
                        major_high = max(swing_highs, key=lambda x: x['volume_strength'])
                        major_low = min(swing_lows, key=lambda x: x['price'])

                        # Ensure we have a meaningful range
                        price_range = major_high['price'] - major_low['price']

                        if price_range > 0:
                            # Calculate retracement levels
                            all_fib_levels.extend(
                                self._calculate_retracement_levels(
                                    major_high, major_low, price_range, lookback
                                )
                            )

                            # Calculate extension levels
                            all_fib_levels.extend(
                                self._calculate_extension_levels(
                                    major_high, major_low, price_range, lookback
                                )
                            )

            # Cluster and weight similar levels
            if not all_fib_levels:
                return []

            clustered_levels = self._cluster_fibonacci_levels(all_fib_levels, df)

            # Return top 12 most significant levels
            return sorted(clustered_levels)[:12]

        except Exception as e:
            logger.error(f"Error in fibonacci calculation: {e}", exc_info=True)
            return []

    def _find_volume_confirmed_swings(self, df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """
        Find swing highs and lows confirmed by volume
        Returns significant swing points with volume strength weighting

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Tuple of (swing_highs, swing_lows) lists
        """
        try:
            if len(df) < 10:
                return [], []

            # Dynamic window based on data length
            window = max(3, len(df) // 15)
            highs = []
            lows = []

            # Calculate volume metrics
            avg_volume = df['volume'].mean()
            volume_std = df['volume'].std()

            for i in range(window, len(df) - window):
                # Check for swing high
                is_swing_high = all(
                    df['high'].iloc[i] >= df['high'].iloc[i-j] for j in range(1, window+1)
                ) and all(
                    df['high'].iloc[i] >= df['high'].iloc[i+j] for j in range(1, window+1)
                )

                if is_swing_high:
                    # Volume confirmation for swing high
                    volume_around_swing = df['volume'].iloc[i-1:i+2].max()
                    volume_strength = min(3.0, volume_around_swing / avg_volume) if avg_volume > 0 else 1.0

                    # Price significance
                    price_context = df['high'].iloc[max(0, i-window*2):i+window*2+1]
                    price_percentile = (
                        (df['high'].iloc[i] - price_context.min()) / 
                        (price_context.max() - price_context.min())
                    ) if price_context.max() > price_context.min() else 0.5

                    highs.append({
                        'price': df['high'].iloc[i],
                        'index': i,
                        'volume': volume_around_swing,
                        'volume_strength': volume_strength,
                        'price_significance': price_percentile,
                        'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i
                    })

                # Check for swing low
                is_swing_low = all(
                    df['low'].iloc[i] <= df['low'].iloc[i-j] for j in range(1, window+1)
                ) and all(
                    df['low'].iloc[i] <= df['low'].iloc[i+j] for j in range(1, window+1)
                )

                if is_swing_low:
                    # Volume confirmation for swing low
                    volume_around_swing = df['volume'].iloc[i-1:i+2].max()
                    volume_strength = min(3.0, volume_around_swing / avg_volume) if avg_volume > 0 else 1.0

                    # Price significance
                    price_context = df['low'].iloc[max(0, i-window*2):i+window*2+1]
                    price_percentile = (
                        (price_context.max() - df['low'].iloc[i]) / 
                        (price_context.max() - price_context.min())
                    ) if price_context.max() > price_context.min() else 0.5

                    lows.append({
                        'price': df['low'].iloc[i],
                        'index': i,
                        'volume': volume_around_swing,
                        'volume_strength': volume_strength,
                        'price_significance': price_percentile,
                        'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i
                    })

            # Filter and rank by significance
            def calculate_significance(swing):
                return (swing['volume_strength'] * 0.6 + swing['price_significance'] * 0.4)

            highs.sort(key=calculate_significance, reverse=True)
            lows.sort(key=calculate_significance, reverse=True)

            # Keep top swings but ensure recent swings are included
            max_swings = min(5, max(1, len(df) // 20))
            recent_threshold = len(df) * 0.7

            filtered_highs = highs[:max_swings]
            filtered_lows = lows[:max_swings]

            # Ensure at least one recent swing
            recent_highs = [s for s in highs if s['index'] >= recent_threshold]
            recent_lows = [s for s in lows if s['index'] >= recent_threshold]

            if recent_highs and not any(s['index'] >= recent_threshold for s in filtered_highs):
                filtered_highs.append(recent_highs[0])

            if recent_lows and not any(s['index'] >= recent_threshold for s in filtered_lows):
                filtered_lows.append(recent_lows[0])

            return filtered_highs, filtered_lows

        except Exception as e:
            logger.error(f"Error in volume-confirmed swing detection: {e}")
            return [], []

    def _calculate_retracement_levels(
        self, 
        major_high: Dict, 
        major_low: Dict, 
        price_range: float, 
        lookback: int
    ) -> List[Dict]:
        """Calculate Fibonacci retracement levels"""
        levels = []

        for fib in self.fib_retracements:
            level = major_high['price'] - (price_range * fib)

            if level > 0:
                # Weight by swing significance and recency
                time_weight = 1.0 if lookback == 21 else 0.7 if lookback == 50 else 0.5
                volume_weight = (major_high['volume_strength'] + major_low['volume_strength']) / 2
                level_weight = time_weight * volume_weight

                levels.append({
                    'price': level,
                    'type': 'retracement',
                    'fib_ratio': fib,
                    'weight': level_weight,
                    'timeframe': f'{lookback}d'
                })

        return levels

    def _calculate_extension_levels(
        self, 
        major_high: Dict, 
        major_low: Dict, 
        price_range: float, 
        lookback: int
    ) -> List[Dict]:
        """Calculate Fibonacci extension levels for breakout targets"""
        levels = []

        for fib in self.fib_extensions:
            # Upside extensions
            level_up = major_high['price'] + (price_range * (fib - 1.0))

            # Downside extensions
            level_down = major_low['price'] - (price_range * (fib - 1.0))

            time_weight = 1.0 if lookback == 21 else 0.7 if lookback == 50 else 0.5

            if level_up > 0:
                volume_weight = major_high['volume_strength']
                levels.append({
                    'price': level_up,
                    'type': 'extension_up',
                    'fib_ratio': fib,
                    'weight': time_weight * volume_weight * 0.8,  # Extensions less certain
                    'timeframe': f'{lookback}d'
                })

            if level_down > 0:
                volume_weight = major_low['volume_strength']
                levels.append({
                    'price': level_down,
                    'type': 'extension_down',
                    'fib_ratio': fib,
                    'weight': time_weight * volume_weight * 0.8,
                    'timeframe': f'{lookback}d'
                })

        return levels

    def _cluster_fibonacci_levels(self, all_fib_levels: List[Dict], df: pd.DataFrame) -> List[float]:
        """
        Cluster and weight similar Fibonacci levels

        Args:
            all_fib_levels: List of Fibonacci level dictionaries
            df: DataFrame for current price context

        Returns:
            List of clustered price levels
        """
        # Sort by price for clustering
        all_fib_levels.sort(key=lambda x: x['price'])

        # Cluster similar levels (within 1% of each other)
        clustered_levels = []
        current_cluster = [all_fib_levels[0]]

        for level in all_fib_levels[1:]:
            cluster_avg = sum(l['price'] for l in current_cluster) / len(current_cluster)

            if abs(level['price'] - cluster_avg) / cluster_avg < 0.01:  # Within 1%
                current_cluster.append(level)
            else:
                # Finalize current cluster
                if current_cluster:
                    cluster_price = sum(
                        l['price'] * l['weight'] for l in current_cluster
                    ) / sum(l['weight'] for l in current_cluster)

                    cluster_weight = sum(l['weight'] for l in current_cluster)

                    # Only include significant levels
                    if cluster_weight > 0.5:
                        clustered_levels.append(cluster_price)

                current_cluster = [level]

        # Don't forget the last cluster
        if current_cluster:
            cluster_price = sum(
                l['price'] * l['weight'] for l in current_cluster
            ) / sum(l['weight'] for l in current_cluster)

            cluster_weight = sum(l['weight'] for l in current_cluster)

            if cluster_weight > 0.5:
                clustered_levels.append(cluster_price)

        # Filter out levels too close to current price (less than 0.5% away)
        current_price = df['close'].iloc[-1]
        filtered_levels = []

        for level in clustered_levels:
            distance = abs(level - current_price) / current_price
            if distance > 0.005:  # At least 0.5% away
                filtered_levels.append(level)

        return filtered_levels
