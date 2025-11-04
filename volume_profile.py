"""
Volume Profile Analysis Module

This module provides professional volume profile analysis for cryptocurrency markets,
including POC, Value Area calculations, accumulation/distribution detection,
and market structure analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class VolumeProfileAnalyzer:
    """Professional Volume Profile Analysis for crypto markets"""

    def __init__(self):
        """Initialize the Volume Profile Analyzer"""
        self.logger = logging.getLogger(__name__)

    def get_professional_volume_profile(self, df: pd.DataFrame, bins: int = 30) -> Dict[str, Any]:
        """
        ENHANCED Professional Volume Profile Analysis for crypto markets
        Returns Point of Control, Value Area High/Low, and volume distribution insights
        Enhanced with crypto-specific accumulation/distribution detection
        """
        try:
            if 'volume' not in df.columns or len(df) < 20:
                return {"poc": None, "vah": None, "val": None, "volume_nodes": []}

            # Get price and volume data
            prices = df['close'].values
            volumes = df['volume'].values
            highs = df['high'].values
            lows = df['low'].values

            # ENHANCEMENT: Dynamic bin sizing based on volatility
            atr = np.mean(highs[-14:] - lows[-14:]) if len(df) >= 14 else (highs[-1] - lows[-1])
            price_range = highs.max() - lows.min()

            # Adjust bins for crypto volatility - more bins for stable periods
            if atr / price_range < 0.02:  # Low volatility
                bins = min(50, bins * 2)
            elif atr / price_range > 0.08:  # High volatility
                bins = max(20, bins // 2)

            # Create price bins using high-low range for more accuracy
            price_min = lows.min()
            price_max = highs.max()
            price_bins = np.linspace(price_min, price_max, bins + 1)

            # ENHANCEMENT: Volume-weighted price distribution
            volume_by_price = np.zeros(bins)
            time_weighted_volume = np.zeros(bins)  # Recent volume gets higher weight
            accumulation_zones = np.zeros(bins)  # Track accumulation vs distribution

            for i in range(len(df)):
                # Use VWAP-style calculation for more accurate representation
                ohlc_avg = (highs[i] + lows[i] + prices[i] + prices[i]) / 4  # Close gets double weight
                volume = volumes[i]

                # Time decay factor - recent data is more important in crypto
                time_weight = 1.0 + (i / len(df)) * 0.5  # Recent data gets up to 50% more weight

                # Find appropriate bin
                bin_index = np.searchsorted(price_bins, ohlc_avg) - 1
                if 0 <= bin_index < bins:
                    volume_by_price[bin_index] += volume
                    time_weighted_volume[bin_index] += volume * time_weight

                    # CRYPTO-SPECIFIC: Detect accumulation vs distribution
                    price_change = 0 if i == 0 else (prices[i] - prices[i-1]) / prices[i-1]

                    # FIX: Safely calculate volume_ratio to prevent division by zero
                    mean_vol_slice = np.mean(volumes[max(0, i-10):i+1]) if i > 0 else 1.0
                    if mean_vol_slice > 0:
                        volume_ratio = volume / mean_vol_slice
                    else:
                        volume_ratio = 1.0  # Default to 1.0 if mean volume is zero

                    # High volume + small price change = accumulation/distribution
                    if volume_ratio > 1.5 and abs(price_change) < 0.02:
                        if price_change >= 0:
                            accumulation_zones[bin_index] += volume * 0.7  # Accumulation
                        else:
                            accumulation_zones[bin_index] -= volume * 0.7  # Distribution

            # ENHANCED: Calculate Point of Control using time-weighted volume
            poc_index = np.argmax(time_weighted_volume)
            poc_price = (price_bins[poc_index] + price_bins[poc_index + 1]) / 2
            poc_strength = time_weighted_volume[poc_index] / np.mean(time_weighted_volume)

            # Calculate Value Area (70% of total volume) using time-weighted data
            total_volume = np.sum(time_weighted_volume)
            target_volume = total_volume * 0.70

            # Find Value Area High and Low
            sorted_indices = np.argsort(time_weighted_volume)[::-1]  # Sort by volume desc
            cumulative_volume = 0
            value_area_bins = []

            for idx in sorted_indices:
                cumulative_volume += time_weighted_volume[idx]
                value_area_bins.append(idx)
                if cumulative_volume >= target_volume:
                    break

            # Value Area High and Low
            vah_price = (price_bins[max(value_area_bins)] + price_bins[max(value_area_bins) + 1]) / 2
            val_price = (price_bins[min(value_area_bins)] + price_bins[min(value_area_bins) + 1]) / 2

            # CRYPTO-ENHANCED: Identify significant volume nodes with accumulation bias
            avg_volume = total_volume / bins
            volume_nodes = []

            for i, vol in enumerate(time_weighted_volume):
                if vol > avg_volume * 1.3:  # Lower threshold for crypto volatility
                    node_price = (price_bins[i] + price_bins[i + 1]) / 2

                    # Calculate accumulation score
                    accumulation_score = accumulation_zones[i] / max(vol, 1)

                    # Classify node type
                    if accumulation_score > 0.3:
                        node_type = "accumulation"  # Smart money buying
                    elif accumulation_score < -0.3:
                        node_type = "distribution"  # Smart money selling
                    else:
                        node_type = "neutral"  # Mixed activity

                    # Calculate proximity to current price for relevance
                    current_price = prices[-1]
                    price_distance = abs(node_price - current_price) / current_price
                    relevance_multiplier = max(0.5, 2.0 - price_distance * 10)  # Closer = more relevant

                    volume_nodes.append({
                        "price": node_price,
                        "volume": vol,
                        "strength": (vol / avg_volume) * relevance_multiplier,
                        "accumulation_score": round(accumulation_score, 3),
                        "node_type": node_type,
                        "distance_pct": round(price_distance * 100, 2)
                    })

            # Sort volume nodes by strength (relevance-adjusted)
            volume_nodes.sort(key=lambda x: x["strength"], reverse=True)

            # CRYPTO-SPECIFIC: Market structure analysis
            market_structure = self._analyze_volume_market_structure(
                time_weighted_volume, accumulation_zones, current_price, price_bins
            )

            return {
                "poc": poc_price,
                "poc_strength": round(poc_strength, 2),
                "vah": vah_price,
                "val": val_price,
                "volume_nodes": volume_nodes[:10],  # Top 10 most relevant nodes
                "total_volume": total_volume,
                "distribution_quality": self._assess_volume_distribution_quality(time_weighted_volume),
                "market_structure": market_structure,
                "accumulation_distribution_ratio": round(np.sum(accumulation_zones) / max(total_volume, 1), 3)
            }

        except Exception as e:
            self.logger.error(f"Error in professional volume profile: {e}")
            return {"poc": None, "vah": None, "val": None, "volume_nodes": []}

    def _assess_volume_distribution_quality(self, volume_by_price: np.ndarray) -> str:
        """Assess the quality of volume distribution for trading signals"""
        try:
            # Calculate distribution metrics
            total_volume = np.sum(volume_by_price)
            if total_volume == 0:
                return "poor"

            # Concentration ratio (how concentrated volume is)
            max_volume = np.max(volume_by_price)
            concentration_ratio = max_volume / total_volume

            # Standard deviation of volume distribution
            volume_std = np.std(volume_by_price)
            volume_mean = np.mean(volume_by_price)
            cv = volume_std / volume_mean if volume_mean > 0 else 0

            # Classification
            if concentration_ratio > 0.3 and cv > 1.5:
                return "excellent"  # High concentration, good for S/R levels
            elif concentration_ratio > 0.2 and cv > 1.0:
                return "good"
            elif concentration_ratio > 0.15:
                return "fair"
            else:
                return "poor"  # Too distributed, weak S/R levels

        except Exception:
            return "unknown"

    def _analyze_volume_market_structure(self, volume_by_price: np.ndarray, 
                                        accumulation_zones: np.ndarray,
                                        current_price: float, 
                                        price_bins: np.ndarray) -> Dict[str, Any]:
        """
        CRYPTO-SPECIFIC: Analyze market structure from volume distribution
        Identifies key support/resistance zones and market phase
        """
        try:
            # Find current price bin
            current_bin = np.searchsorted(price_bins, current_price) - 1
            current_bin = max(0, min(current_bin, len(volume_by_price) - 1))

            # Analyze volume above and below current price
            volume_above = np.sum(volume_by_price[current_bin+1:]) if current_bin < len(volume_by_price) - 1 else 0
            volume_below = np.sum(volume_by_price[:current_bin]) if current_bin > 0 else 0
            total_volume = volume_above + volume_below + volume_by_price[current_bin]

            if total_volume == 0:
                return {"phase": "unknown", "bias": "neutral"}

            # Calculate volume distribution bias
            volume_above_pct = volume_above / total_volume * 100
            volume_below_pct = volume_below / total_volume * 100

            # Analyze accumulation distribution patterns
            acc_above = np.sum(accumulation_zones[current_bin+1:]) if current_bin < len(accumulation_zones) - 1 else 0
            acc_below = np.sum(accumulation_zones[:current_bin]) if current_bin > 0 else 0

            # Determine market phase
            if volume_above_pct > 65:
                if acc_above > 0:
                    phase = "accumulation_above"  # Strength above, likely to move up
                else:
                    phase = "resistance_above"  # Heavy selling above
            elif volume_below_pct > 65:
                if acc_below > 0:
                    phase = "support_below"  # Strong support below
                else:
                    phase = "distribution_below"  # Weak support below
            else:
                phase = "balanced"  # Relatively balanced

            # Overall market bias
            if acc_above + acc_below > 0:
                bias = "bullish_accumulation"
            elif acc_above + acc_below < 0:
                bias = "bearish_distribution"
            else:
                bias = "neutral"

            return {
                "phase": phase,
                "bias": bias,
                "volume_above_pct": round(volume_above_pct, 1),
                "volume_below_pct": round(volume_below_pct, 1),
                "net_accumulation": round(acc_above + acc_below, 2)
            }

        except Exception:
            return {"phase": "unknown", "bias": "neutral"}
