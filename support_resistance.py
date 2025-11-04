"""
Module 7: Support and Resistance Analysis
Professional-grade S/R identification using volume profile, structural levels,
Fibonacci, pivot points, and confluence-based scoring
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class SupportResistanceAnalyzer:
    """
    Professional support and resistance analyzer
    Uses volume profile as primary source with confluence scoring
    """

    def __init__(self, volume_profile_analyzer=None, fibonacci_analyzer=None):
        """
        Initialize S/R analyzer

        Args:
            volume_profile_analyzer: VolumeProfileAnalyzer instance
            fibonacci_analyzer: FibonacciAnalyzer instance
        """
        self.volume_profile_analyzer = volume_profile_analyzer
        self.fibonacci_analyzer = fibonacci_analyzer

        # Source scores for confluence-based scoring
        self.source_scores = {
            'volume_poc': 10,      # Highest importance
            'volume_vah': 8,       # Value Area is critical
            'volume_val': 8,
            'swing_point_1d': 6,   # Daily structure is primary
            'statistical_1d': 5,
            'fibonacci': 4,        # Fibonacci confluence is strong
            'swing_point_4h': 3,   # 4H structure is secondary
            'statistical_4h': 2,
            'pivot': 2,
            'psychological': 1     # Lowest importance
        }

    def identify_support_resistance(
        self, 
        all_timeframes: Dict[str, pd.DataFrame]
    ) -> Tuple[List[Tuple[float, int]], List[Tuple[float, int]]]:
        """
        Identify support and resistance levels using confluence-based scoring

        Args:
            all_timeframes: Dictionary of DataFrames for different timeframes

        Returns:
            Tuple of (support_levels, resistance_levels) with (price, strength_score)
        """
        df_daily = all_timeframes.get("1d")
        if df_daily is None or df_daily.empty:
            return [], []

        try:
            potential_levels = []
            current_price = df_daily['close'].iloc[-1]

            # 1. Volume Profile levels (primary source)
            if self.volume_profile_analyzer:
                volume_profile_result = self.volume_profile_analyzer.get_professional_volume_profile(df_daily)

                poc_price = volume_profile_result.get("poc")
                vah_price = volume_profile_result.get("vah")
                val_price = volume_profile_result.get("val")

                if poc_price is not None:
                    potential_levels.append((poc_price, 'volume_poc'))
                if vah_price is not None:
                    potential_levels.append((vah_price, 'volume_vah'))
                if val_price is not None:
                    potential_levels.append((val_price, 'volume_val'))

            # 2. Structural levels from daily timeframe
            daily_structural_levels = self._find_structural_levels(df_daily, lookback=30)
            for level, source in daily_structural_levels:
                potential_levels.append((level, source + '_1d'))

            # 3. Structural levels from 4H timeframe if available
            df_4h = all_timeframes.get("4h")
            if df_4h is not None and not df_4h.empty:
                four_hour_structural_levels = self._find_structural_levels(df_4h, lookback=15)
                for level, source in four_hour_structural_levels:
                    potential_levels.append((level, source + '_4h'))

            # 4. Fibonacci levels
            if self.fibonacci_analyzer:
                fib_levels = self.fibonacci_analyzer.calculate_fibonacci_levels(df_daily)
                for level in fib_levels:
                    potential_levels.append((level, 'fibonacci'))

            # 5. Pivot points
            pivot_levels = self._calculate_pivot_points(df_daily)
            for level in pivot_levels:
                potential_levels.append((level, 'pivot'))

            # 6. Psychological levels
            psychological_levels = self._generate_psychological_levels(current_price)
            for level in psychological_levels:
                potential_levels.append((level, 'psychological'))

            # Cluster and score levels
            clustered_levels = self._cluster_and_score_levels(potential_levels)

            # Separate into support and resistance
            support_levels = [(price, score) for price, score in clustered_levels if price < current_price]
            resistance_levels = [(price, score) for price, score in clustered_levels if price > current_price]

            # Sort and return top 8 of each
            support_levels.sort(key=lambda x: x[0], reverse=True)
            resistance_levels.sort(key=lambda x: x[0])

            return support_levels[:8], resistance_levels[:8]

        except Exception as e:
            logger.error(f"Error in identify_support_resistance: {e}", exc_info=True)
            return [], []

    def _find_structural_levels(self, df: pd.DataFrame, lookback: int) -> List[Tuple[float, str]]:
        """
        Find statistical and swing point support/resistance levels

        Args:
            df: DataFrame with OHLCV data
            lookback: Number of periods to look back

        Returns:
            List of (price, source) tuples
        """
        if df is None or len(df) < lookback:
            return []

        potential_levels = []

        # Statistical Support/Resistance
        try:
            for i in range(lookback, len(df) - 5):
                # Support levels
                low_price = df['low'].iloc[i]
                if i + 5 < len(df):
                    recovery = df['close'].iloc[i:i+5].min() > low_price * 1.005

                    if recovery:
                        test_count = 0
                        for j in range(max(0, i-lookback), min(len(df), i+lookback)):
                            if abs(df['low'].iloc[j] - low_price) / low_price < 0.01:
                                test_count += 1

                        if test_count >= 2:
                            potential_levels.append((low_price, 'statistical'))

                # Resistance levels
                high_price = df['high'].iloc[i]
                if i + 5 < len(df):
                    rejection = df['close'].iloc[i:i+5].max() < high_price * 0.995

                    if rejection:
                        test_count = 0
                        for j in range(max(0, i-lookback), min(len(df), i+lookback)):
                            if abs(df['high'].iloc[j] - high_price) / high_price < 0.01:
                                test_count += 1

                        if test_count >= 2:
                            potential_levels.append((high_price, 'statistical'))

        except Exception as e:
            logger.error(f"Statistical levels calculation error: {e}")

        # Swing points
        try:
            for i in range(lookback, len(df) - lookback):
                # Swing highs
                is_high = all(
                    df['high'].iloc[i] >= df['high'].iloc[i-j] for j in range(1, lookback + 1)
                ) and all(
                    df['high'].iloc[i] >= df['high'].iloc[i+j] for j in range(1, lookback + 1)
                )

                if is_high:
                    potential_levels.append((df['high'].iloc[i], 'swing_point'))

                # Swing lows
                is_low = all(
                    df['low'].iloc[i] <= df['low'].iloc[i-j] for j in range(1, lookback + 1)
                ) and all(
                    df['low'].iloc[i] <= df['low'].iloc[i+j] for j in range(1, lookback + 1)
                )

                if is_low:
                    potential_levels.append((df['low'].iloc[i], 'swing_point'))

        except Exception as e:
            logger.error(f"Historical levels calculation error: {e}")

        return potential_levels

    def _calculate_pivot_points(self, df: pd.DataFrame) -> List[float]:
        """Calculate traditional pivot points"""
        pivot_levels = []

        if len(df) >= 2:
            try:
                prev_high = df['high'].iloc[-2]
                prev_low = df['low'].iloc[-2]
                prev_close = df['close'].iloc[-2]

                pivot = (prev_high + prev_low + prev_close) / 3

                # Calculate pivot levels
                levels = [
                    pivot,                              # Central pivot
                    (2 * pivot) - prev_low,            # R1
                    (2 * pivot) - prev_high,           # S1
                    pivot + (prev_high - prev_low),    # R2
                    pivot - (prev_high - prev_low),    # S2
                    pivot + 2 * (prev_high - prev_low), # R3
                    pivot - 2 * (prev_high - prev_low)  # S3
                ]

                pivot_levels = [level for level in levels if level > 0]

            except Exception as e:
                logger.error(f"Pivot calculation error: {e}")

        return pivot_levels

    def _generate_psychological_levels(self, current_price: float) -> List[float]:
        """Generate psychological price levels based on round numbers"""
        levels = []

        if current_price <= 0:
            return levels

        magnitude = 10 ** (len(str(int(current_price))) - 1)
        psychological_multipliers = [0.1, 0.2, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 5.0, 7.5, 10.0]

        for multiplier in psychological_multipliers:
            level = magnitude * multiplier
            if level > 0 and level != current_price:
                levels.append(level)

        return levels

    def _cluster_and_score_levels(self, potential_levels: List[Tuple[float, str]]) -> List[Tuple[float, int]]:
        """
        Cluster similar price levels and calculate confluence scores

        Args:
            potential_levels: List of (price, source) tuples

        Returns:
            List of (price, score) tuples
        """
        if not potential_levels:
            return []

        # Sort by price
        potential_levels.sort(key=lambda x: x[0])

        # Cluster levels within 0.5% of each other
        clustered_levels = []
        current_cluster = [potential_levels[0]]

        for level, source in potential_levels[1:]:
            cluster_avg_price = sum(p for p, s in current_cluster) / len(current_cluster)

            if abs(level - cluster_avg_price) / cluster_avg_price < 0.005:  # Within 0.5%
                current_cluster.append((level, source))
            else:
                # Calculate final level price and score
                final_price = sum(p for p, s in current_cluster) / len(current_cluster)
                final_score = sum(self.source_scores.get(s, 1) for p, s in current_cluster)

                clustered_levels.append((final_price, final_score))
                current_cluster = [(level, source)]

        # Don't forget the last cluster
        if current_cluster:
            final_price = sum(p for p, s in current_cluster) / len(current_cluster)
            final_score = sum(self.source_scores.get(s, 1) for p, s in current_cluster)
            clustered_levels.append((final_price, final_score))

        return clustered_levels

    def is_level_significant(
        self, 
        price_level: float, 
        sr_levels: List[Tuple[float, int]], 
        tolerance_pct: float = 0.015, 
        min_strength: int = 4
    ) -> bool:
        """
        Check if a price level aligns with a significant S/R level

        Args:
            price_level: The price point to check
            sr_levels: List of (price, strength) tuples for S/R
            tolerance_pct: Percentage tolerance for matching (default 1.5%)
            min_strength: Minimum strength score required (default 4)

        Returns:
            True if the price level aligns with a strong S/R zone
        """
        if price_level == 0:
            return False

        for sr_price, sr_strength in sr_levels:
            if abs(price_level - sr_price) / sr_price <= tolerance_pct:
                if sr_strength >= min_strength:
                    return True

        return False
