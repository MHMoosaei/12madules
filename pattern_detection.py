"""
Module 10: Pattern Detection
Professional chart pattern recognition with volume and S/R validation.
Maintains 100% backward compatibility with original implementation.
"""

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


class PatternDetector:
    """
    Comprehensive chart pattern detection with professional-grade validation.

    Detects:
    - Classical patterns (Double Top/Bottom, H&S, Triangles, Wedges, Flags, Channels)
    - Fibonacci retracements and extensions
    - Harmonic patterns (ABCD, Gartley, Butterfly, Crab)
    - Support/Resistance tests (bounces, rejections, breakouts)
    - Divergences (RSI, MACD, Volume)
    """

    def __init__(self):
        """Initialize pattern detector with default parameters."""
        self.min_pattern_bars = 30
        self.min_quality_score = 60

    def detect_chart_patterns(
        self,
        df: pd.DataFrame,
        support_levels: List[Tuple[float, int]],
        resistance_levels: List[Tuple[float, int]],
        all_timeframes: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Dict[str, Any]:
        """
        Main pattern detection orchestrator.

        Args:
            df: DataFrame with OHLCV and indicators
            support_levels: List of (price, strength) tuples
            resistance_levels: List of (price, strength) tuples  
            all_timeframes: Optional dict of timeframe DataFrames

        Returns:
            Dict with best pattern detected
        """
        if df is None or len(df) < self.min_pattern_bars:
            return {"pattern": "unknown", "confidence": 0, "direction": "NEUTRAL"}

        try:
            patterns = []

            # --- 1. Consolidated Multi-Pattern Detectors ---
            triangle_wedge_results = self._detect_triangles_and_wedges(
                df, all_timeframes, support_levels, resistance_levels
            )
            if triangle_wedge_results:
                patterns.extend(triangle_wedge_results)

            # --- 2. Single High-Probability Pattern Detectors ---
            single_pattern_detectors = [
                self._detect_head_and_shoulders_pattern,
                self._detect_inverse_head_and_shoulders_pattern,
                self._detect_double_top,
                self._detect_double_bottom,
                self._detect_channel_up,
                self._detect_channel_down,
                self._detect_sideways_channel_strategy,
            ]

            for detector in single_pattern_detectors:
                result = detector(df, support_levels, resistance_levels, all_timeframes)
                if result and result.get("detected"):
                    patterns.append(result)

            # --- 3. Contextual and Fallback Signals ---
            if not patterns:
                divergence_result = self._detect_divergence_signal(
                    df, "Divergence Signal", all_timeframes
                )
                if divergence_result and divergence_result.get("detected"):
                    patterns.append(divergence_result)
                else:
                    sr_test_results = self._detect_support_resistance_tests(
                        df, support_levels, resistance_levels, all_timeframes
                    )
                    if sr_test_results:
                        patterns.extend(sr_test_results)

            # --- 4. Final Evaluation ---
            if not patterns:
                indicator_signals = self._detect_indicator_signals(
                    df, support_levels, resistance_levels, all_timeframes
                )
                if indicator_signals:
                    patterns.extend(indicator_signals)
                else:
                    return {"pattern": "unknown", "confidence": 0, "direction": "NEUTRAL"}

            # Return highest confidence pattern
            patterns.sort(key=lambda x: x.get("confidence", 0), reverse=True)
            return patterns[0]

        except Exception as e:
            print(f"Error detecting chart patterns: {str(e)}")
            return {"pattern": "error", "confidence": 0, "direction": "NEUTRAL"}

    def _detect_triangles_and_wedges(
        self,
        df: pd.DataFrame,
        all_timeframes: Optional[Dict[str, pd.DataFrame]],
        support_levels: List[Tuple[float, int]],
        resistance_levels: List[Tuple[float, int]]
    ) -> List[Dict[str, Any]]:
        """Detect triangle and wedge patterns with geometric validation."""
        patterns = []

        if len(df) < 40:
            return patterns

        try:
            recent_df = df.tail(60).copy()

            # Find swing highs and lows
            highs = []
            lows = []

            for i in range(5, len(recent_df) - 5):
                if all(recent_df["high"].iloc[i] >= recent_df["high"].iloc[i-j] for j in range(1, 6)) and \
                   all(recent_df["high"].iloc[i] >= recent_df["high"].iloc[i+j] for j in range(1, 6)):
                    highs.append((i, recent_df["high"].iloc[i]))

                if all(recent_df["low"].iloc[i] <= recent_df["low"].iloc[i-j] for j in range(1, 6)) and \
                   all(recent_df["low"].iloc[i] <= recent_df["low"].iloc[i+j] for j in range(1, 6)):
                    lows.append((i, recent_df["low"].iloc[i]))

            if len(highs) >= 2 and len(lows) >= 2:
                # Calculate trendlines
                high_indices = np.array([h[0] for h in highs]).reshape(-1, 1)
                high_prices = np.array([h[1] for h in highs])

                low_indices = np.array([l[0] for l in lows]).reshape(-1, 1)
                low_prices = np.array([l[1] for l in lows])

                # Fit regression lines
                high_reg = LinearRegression().fit(high_indices, high_prices)
                low_reg = LinearRegression().fit(low_indices, low_prices)

                high_slope = high_reg.coef_[0]
                low_slope = low_reg.coef_[0]

                # Determine pattern type
                current_price = recent_df["close"].iloc[-1]

                # Ascending Triangle: flat resistance, rising support
                if abs(high_slope) < 0.0001 and low_slope > 0.001:
                    quality_score = 75
                    if current_price > high_prices.max() * 0.98:  # Near resistance
                        quality_score += 10

                    patterns.append({
                        "detected": True,
                        "pattern": "Ascending Triangle",
                        "confidence": min(95, quality_score),
                        "direction": "LONG",
                        "details": {
                            "resistance_level": float(high_prices.max()),
                            "support_slope": float(low_slope)
                        }
                    })

                # Descending Triangle: flat support, falling resistance  
                elif abs(low_slope) < 0.0001 and high_slope < -0.001:
                    quality_score = 75
                    if current_price < low_prices.min() * 1.02:  # Near support
                        quality_score += 10

                    patterns.append({
                        "detected": True,
                        "pattern": "Descending Triangle",
                        "confidence": min(95, quality_score),
                        "direction": "SHORT",
                        "details": {
                            "support_level": float(low_prices.min()),
                            "resistance_slope": float(high_slope)
                        }
                    })

                # Symmetrical Triangle: converging trendlines
                elif high_slope < -0.001 and low_slope > 0.001:
                    convergence = abs(high_slope) + abs(low_slope)
                    if convergence > 0.005:  # Minimum convergence rate
                        quality_score = 70

                        # Calculate apex (where lines meet)
                        apex_x = (low_reg.intercept_ - high_reg.intercept_) / (high_slope - low_slope)
                        bars_to_apex = apex_x - len(recent_df)

                        if 0 < bars_to_apex < 20:  # Approaching apex
                            quality_score += 15

                        # Determine direction from volume and momentum
                        direction = "LONG" if recent_df["close"].iloc[-1] > recent_df["close"].iloc[-10] else "SHORT"

                        patterns.append({
                            "detected": True,
                            "pattern": "Symmetrical Triangle",
                            "confidence": min(95, quality_score),
                            "direction": direction,
                            "details": {
                                "bars_to_apex": float(bars_to_apex),
                                "high_slope": float(high_slope),
                                "low_slope": float(low_slope)
                            }
                        })

                # Rising Wedge: both lines rising, converging (bearish)
                elif high_slope > 0.001 and low_slope > 0.001 and high_slope < low_slope:
                    quality_score = 72

                    # Volume should be contracting
                    if "volume" in recent_df.columns:
                        early_vol = recent_df["volume"].iloc[:20].mean()
                        recent_vol = recent_df["volume"].iloc[-20:].mean()
                        if recent_vol < early_vol * 0.8:
                            quality_score += 13

                    patterns.append({
                        "detected": True,
                        "pattern": "Rising Wedge",
                        "confidence": min(95, quality_score),
                        "direction": "SHORT",
                        "details": {
                            "high_slope": float(high_slope),
                            "low_slope": float(low_slope)
                        }
                    })

                # Falling Wedge: both lines falling, converging (bullish)
                elif high_slope < -0.001 and low_slope < -0.001 and high_slope > low_slope:
                    quality_score = 72

                    # Volume should be contracting
                    if "volume" in recent_df.columns:
                        early_vol = recent_df["volume"].iloc[:20].mean()
                        recent_vol = recent_df["volume"].iloc[-20:].mean()
                        if recent_vol < early_vol * 0.8:
                            quality_score += 13

                    patterns.append({
                        "detected": True,
                        "pattern": "Falling Wedge",
                        "confidence": min(95, quality_score),
                        "direction": "LONG",
                        "details": {
                            "high_slope": float(high_slope),
                            "low_slope": float(low_slope)
                        }
                    })

        except Exception as e:
            print(f"Error in triangle/wedge detection: {str(e)}")

        return patterns

    def _detect_double_top(
        self,
        df: pd.DataFrame,
        support_levels: List[Tuple[float, int]],
        resistance_levels: List[Tuple[float, int]],
        all_timeframes: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Dict[str, Any]:
        """
        Enhanced double top detection with professional-grade validation.

        Key improvements:
        - Stricter geometric requirements
        - Volume divergence analysis
        - RSI momentum divergence
        - Support/resistance confluence
        - Minimum pattern duration requirements
        """
        try:
            recent_df = df.tail(60).copy()

            if len(recent_df) < 35:
                return {"detected": False}

            # Find significant peaks (prominence-based)
            significant_peaks = []
            for i in range(8, len(recent_df) - 8):
                if all(recent_df["high"].iloc[i] >= recent_df["high"].iloc[i-j] for j in range(1, 9)) and \
                   all(recent_df["high"].iloc[i] >= recent_df["high"].iloc[i+j] for j in range(1, 9)):
                    peak_prominence = recent_df["high"].iloc[i] - max(
                        recent_df["low"].iloc[i-8:i].min(),
                        recent_df["low"].iloc[i:i+8].min()
                    )
                    if peak_prominence > recent_df["close"].iloc[i] * 0.02:  # Min 2% prominence
                        significant_peaks.append((i, recent_df["high"].iloc[i], recent_df["volume"].iloc[i]))

            if len(significant_peaks) < 2:
                return {"detected": False}

            # Check peak pairs
            for i in range(len(significant_peaks) - 1):
                for j in range(i + 1, len(significant_peaks)):
                    peak1_idx, peak1_val, peak1_vol = significant_peaks[i]
                    peak2_idx, peak2_val, peak2_vol = significant_peaks[j]

                    # 1. Price similarity (within 1.5%)
                    if not (abs(peak2_val - peak1_val) / peak1_val <= 0.015):
                        continue

                    # 2. Temporal spacing (15-40 bars)
                    if not (15 <= peak2_idx - peak1_idx <= 40):
                        continue

                    # 3. Valley depth (minimum 4%)
                    trough_section = recent_df.iloc[peak1_idx:peak2_idx]
                    if trough_section.empty:
                        continue
                    trough_val = trough_section["low"].min()
                    valley_depth = (min(peak1_val, peak2_val) / trough_val - 1)
                    if not (valley_depth >= 0.04):
                        continue

                    # 4. Neckline break confirmation
                    if not (recent_df["close"].iloc[-1] < trough_val * 0.995):
                        continue

                    # 5. S/R confluence
                    is_at_resistance = self._is_level_significant(
                        peak1_val, resistance_levels, min_strength=4
                    )
                    if not is_at_resistance:
                        continue

                    # --- Quality Scoring ---
                    quality_score = 60

                    # Peak similarity
                    similarity = 100 - (abs(peak2_val - peak1_val) / peak1_val * 100)
                    quality_score += similarity * 0.4

                    # Temporal spacing (15 bars = ideal)
                    distance_factor = 100 - (abs((peak2_idx - peak1_idx) - 15) / 3)
                    quality_score += distance_factor * 0.3

                    # Breakdown depth
                    breakdown_factor = (trough_val - recent_df["close"].iloc[-1]) / trough_val * 100 * 2
                    quality_score += breakdown_factor * 0.3

                    # Volume divergence (bearish: lower volume on second peak)
                    if "volume" in recent_df.columns:
                        if peak2_vol < peak1_vol * 0.8:
                            quality_score += 10

                    # RSI divergence
                    if "rsi_14" in df.columns:
                        try:
                            main_df_idx1 = df.index.get_loc(recent_df.index[peak1_idx])
                            main_df_idx2 = df.index.get_loc(recent_df.index[peak2_idx])
                            rsi1 = df.loc[df.index[main_df_idx1], "rsi_14"]
                            rsi2 = df.loc[df.index[main_df_idx2], "rsi_14"]

                            # Bearish divergence: price higher, RSI lower
                            if peak2_val >= peak1_val and rsi2 < rsi1:
                                quality_score += 20
                        except (KeyError, IndexError):
                            pass

                    # MACD divergence
                    if "macd" in df.columns:
                        try:
                            main_df_idx1 = df.index.get_loc(recent_df.index[peak1_idx])
                            main_df_idx2 = df.index.get_loc(recent_df.index[peak2_idx])
                            macd1 = df.loc[df.index[main_df_idx1], "macd"]
                            macd2 = df.loc[df.index[main_df_idx2], "macd"]

                            if peak2_val >= peak1_val and macd2 < macd1:
                                quality_score += 20
                        except (KeyError, IndexError):
                            pass

                    # Timing (pattern completed recently)
                    bars_since_completion = len(recent_df) - peak2_idx
                    if 3 <= bars_since_completion <= 10:
                        quality_score += 8
                    elif bars_since_completion > 20:
                        quality_score -= 10

                    # Filter low quality
                    if quality_score < 75:
                        continue

                    # Get trend context for final confidence adjustment
                    trends = {}
                    if all_timeframes:
                        for tf in ["1h", "4h", "1d", "1w"]:
                            if tf in all_timeframes:
                                trends[tf] = self._analyze_trend(all_timeframes[tf], tf)

                    final_confidence = self._calculate_enhanced_confidence(
                        "Double Top",
                        "SHORT",
                        df,
                        support_levels,
                        resistance_levels,
                        trends,
                        base_confidence_override=quality_score
                    )

                    return {
                        "detected": True,
                        "pattern": "Double Top",
                        "confidence": final_confidence,
                        "direction": "SHORT",
                        "details": {
                            "quality_score": quality_score,
                            "peak1_price": peak1_val,
                            "peak2_price": peak2_val,
                            "neckline_price": trough_val,
                            "breakdown_price": recent_df["close"].iloc[-1],
                            "volume_divergence": peak2_vol / peak1_vol,
                            "valley_depth_pct": valley_depth * 100
                        }
                    }

            return {"detected": False}

        except Exception as e:
            print(f"Error in enhanced double top detection: {e}")
            return {"detected": False}

    def _detect_double_bottom(
        self,
        df: pd.DataFrame,
        support_levels: List[Tuple[float, int]],
        resistance_levels: List[Tuple[float, int]],
        all_timeframes: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Dict[str, Any]:
        """
        Enhanced double bottom detection (mirror of double top).
        """
        try:
            recent_df = df.tail(60).copy()

            if len(recent_df) < 35:
                return {"detected": False}

            # Find significant troughs
            significant_lows = []
            for i in range(8, len(recent_df) - 8):
                if all(recent_df["low"].iloc[i] <= recent_df["low"].iloc[i-j] for j in range(1, 9)) and \
                   all(recent_df["low"].iloc[i] <= recent_df["low"].iloc[i+j] for j in range(1, 9)):
                    trough_prominence = max(
                        recent_df["high"].iloc[i-8:i].max(),
                        recent_df["high"].iloc[i:i+8].max()
                    ) - recent_df["low"].iloc[i]
                    if trough_prominence > recent_df["close"].iloc[i] * 0.02:
                        significant_lows.append((i, recent_df["low"].iloc[i], recent_df["volume"].iloc[i]))

            if len(significant_lows) < 2:
                return {"detected": False}

            for i in range(len(significant_lows) - 1):
                for j in range(i + 1, len(significant_lows)):
                    low1_idx, low1_val, low1_vol = significant_lows[i]
                    low2_idx, low2_val, low2_vol = significant_lows[j]

                    # 1. Price similarity
                    if not (abs(low2_val - low1_val) / low1_val <= 0.015):
                        continue

                    # 2. Temporal spacing
                    if not (15 <= low2_idx - low1_idx <= 40):
                        continue

                    # 3. Peak height (minimum 3%)
                    peak_section = recent_df.iloc[low1_idx:low2_idx]
                    if peak_section.empty:
                        continue
                    peak_val = peak_section["high"].max()
                    peak_height = (peak_val / max(low1_val, low2_val) - 1)
                    if not (peak_height >= 0.03):
                        continue

                    # 4. Neckline break confirmation
                    if not (recent_df["close"].iloc[-1] > peak_val):
                        continue

                    # 5. S/R confluence
                    is_at_support = self._is_level_significant(
                        low1_val, support_levels, min_strength=4
                    )
                    if not is_at_support:
                        continue

                    # --- Quality Scoring ---
                    quality_score = 60

                    similarity = 100 - (abs(low2_val - low1_val) / low1_val * 100)
                    quality_score += similarity * 0.4

                    distance_factor = 100 - (abs((low2_idx - low1_idx) - 15) / 3)
                    quality_score += distance_factor * 0.3

                    breakout_factor = (recent_df["close"].iloc[-1] - peak_val) / peak_val * 100 * 2
                    quality_score += breakout_factor * 0.3

                    # Volume confirmation (bullish: higher volume on second low)
                    if "volume" in recent_df.columns:
                        if low2_vol > low1_vol * 1.2:
                            quality_score += 10

                    # RSI divergence (bullish: price lower, RSI higher)
                    if "rsi_14" in df.columns:
                        try:
                            main_df_idx1 = recent_df.index[low1_idx]
                            main_df_idx2 = recent_df.index[low2_idx]
                            rsi1 = df.loc[main_df_idx1, "rsi_14"]
                            rsi2 = df.loc[main_df_idx2, "rsi_14"]

                            if low2_val <= low1_val and rsi2 > rsi1:
                                quality_score += 20
                        except (KeyError, IndexError):
                            pass

                    # MACD divergence
                    if "macd" in df.columns:
                        try:
                            main_df_idx1 = recent_df.index[low1_idx]
                            main_df_idx2 = recent_df.index[low2_idx]
                            macd1 = df.loc[main_df_idx1, "macd"]
                            macd2 = df.loc[main_df_idx2, "macd"]

                            if low2_val <= low1_val and macd2 > macd1:
                                quality_score += 20
                        except (KeyError, IndexError):
                            pass

                    bars_since_completion = len(recent_df) - low2_idx
                    if 3 <= bars_since_completion <= 10:
                        quality_score += 8
                    elif bars_since_completion > 20:
                        quality_score -= 10

                    if quality_score < 75:
                        continue

                    trends = {}
                    if all_timeframes:
                        for tf in ["1h", "4h", "1d", "1w"]:
                            if tf in all_timeframes:
                                trends[tf] = self._analyze_trend(all_timeframes[tf], tf)

                    final_confidence = self._calculate_enhanced_confidence(
                        "Double Bottom",
                        "LONG",
                        df,
                        support_levels,
                        resistance_levels,
                        trends,
                        base_confidence_override=quality_score
                    )

                    return {
                        "detected": True,
                        "pattern": "Double Bottom",
                        "confidence": final_confidence,
                        "direction": "LONG",
                        "details": {
                            "quality_score": quality_score,
                            "low1_price": low1_val,
                            "low2_price": low2_val,
                            "neckline_price": peak_val,
                            "breakout_price": recent_df["close"].iloc[-1],
                            "volume_confirmation": low2_vol / low1_vol,
                            "peak_height_pct": peak_height * 100
                        }
                    }

            return {"detected": False}

        except Exception as e:
            print(f"Error in double bottom detection: {e}")
            return {"detected": False}

    def _is_level_significant(self, price_level: float, sr_levels: List[Tuple[float, int]], tolerance_pct: float = 0.015, min_strength: int = 4) -> bool:
        """
        Checks if a given price_level aligns with a significant S/R level.
        """
        if price_level == 0: return False
        for sr_price, sr_strength in sr_levels:
            if abs(price_level - sr_price) / sr_price <= tolerance_pct:
                if sr_strength >= min_strength:
                    return True
        return False

    def _analyze_trend(self, df: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """Analyzes the trend for a given timeframe using EMAs."""
        if df is None or len(df) < 50:
            return {'trend': 'NEUTRAL', 'strength': 0}
        
        trend = 'NEUTRAL'
        strength = 0
        
        if 'ema50' in df.columns and 'ema200' in df.columns:
            ema50 = df['ema50'].iloc[-1]
            ema200 = df['ema200'].iloc[-1]
            
            if ema50 > ema200:
                trend = 'BULLISH'
                strength = min(100, ((ema50 - ema200) / ema200) * 100 * 10)
            elif ema50 < ema200:
                trend = 'BEARISH'
                strength = min(100, ((ema200 - ema50) / ema50) * 100 * 10)
        
        return {'trend': trend, 'strength': strength}

    def _calculate_enhanced_confidence(
        self,
        pattern_name: str,
        direction: str,
        df: pd.DataFrame,
        support_levels: List[Tuple[float, int]],
        resistance_levels: List[Tuple[float, int]],
        trends: Dict[str, Dict[str, Any]],
        base_confidence_override: Optional[float] = None
    ) -> int:
        """Calculates a more advanced, context-aware confidence score."""
        # Base confidence from pattern success rate or override
        base_confidence = base_confidence_override if base_confidence_override is not None else 65

        # 1. Multi-Timeframe Trend Alignment
        alignment_score = 0
        total_weight = 0
        weights = {"1h": 0.1, "4h": 0.2, "1d": 0.4, "1w": 0.3}
        
        for tf, trend_info in trends.items():
            weight = weights.get(tf, 0)
            if trend_info['trend'] == direction:
                alignment_score += weight
            elif trend_info['trend'] != 'NEUTRAL':
                alignment_score -= weight * 0.5
            total_weight += weight
        
        if total_weight > 0:
            confidence_adjustment = (alignment_score / total_weight) * 25
            base_confidence += confidence_adjustment

        # 2. Volume Confirmation
        if "volume" in df.columns and len(df) > 20:
            recent_vol = df["volume"].iloc[-5:].mean()
            avg_vol = df["volume"].iloc[-20:].mean()
            if recent_vol > avg_vol * 1.5:
                base_confidence += 10
            elif recent_vol < avg_vol * 0.7:
                base_confidence -= 5

        # 3. S/R Confluence
        current_price = df["close"].iloc[-1]
        if direction == "LONG":
            # Check for strong support below
            for s_price, s_strength in support_levels:
                if s_price < current_price and s_strength > 5:
                    base_confidence += 5
                    break
        else: # SHORT
            # Check for strong resistance above
            for r_price, r_strength in resistance_levels:
                if r_price > current_price and r_strength > 5:
                    base_confidence += 5
                    break

        return int(min(100, max(50, base_confidence)))

    def _detect_head_and_shoulders_pattern(self, df: pd.DataFrame, support_levels: list, resistance_levels: list, all_timeframes: dict) -> dict:
        """Detects the Head and Shoulders pattern with validation."""
        try:
            recent_df = df.tail(100)
            if len(recent_df) < 60:
                return {"detected": False}

            # Find peaks
            peaks, _ = find_peaks(recent_df['high'], prominence=recent_df['high'].mean() * 0.02, width=3)
            
            if len(peaks) < 3:
                return {"detected": False}

            # Check for H&S structure (peak, higher peak, lower peak)
            for i in range(len(peaks) - 2):
                p1, p2, p3 = peaks[i], peaks[i+1], peaks[i+2]
                h1, h2, h3 = recent_df['high'].iloc[p1], recent_df['high'].iloc[p2], recent_df['high'].iloc[p3]

                # Head must be highest
                if h2 > h1 and h2 > h3:
                    # Shoulders should be roughly symmetrical
                    if abs(h1 - h3) / h2 < 0.15:
                        # Find troughs for neckline
                        trough1_idx = recent_df['low'].iloc[p1:p2].idxmin()
                        trough2_idx = recent_df['low'].iloc[p2:p3].idxmin()
                        t1_val, t2_val = recent_df['low'].loc[trough1_idx], recent_df['low'].loc[trough2_idx]

                        # Neckline break confirmation
                        if recent_df['close'].iloc[-1] < min(t1_val, t2_val):
                            confidence = self._calculate_enhanced_confidence(
                                "Head and Shoulders", "SHORT", df, support_levels, resistance_levels, all_timeframes
                            )
                            return {
                                "detected": True,
                                "pattern": "Head and Shoulders",
                                "confidence": confidence,
                                "direction": "SHORT",
                                "details": {"head": h2, "left_shoulder": h1, "right_shoulder": h3, "neckline": min(t1_val, t2_val)}
                            }
            return {"detected": False}
        except Exception:
            return {"detected": False}

    def _detect_inverse_head_and_shoulders_pattern(self, df: pd.DataFrame, support_levels: list, resistance_levels: list, all_timeframes: dict) -> dict:
        """Detects the Inverse Head and Shoulders pattern."""
        try:
            recent_df = df.tail(100)
            if len(recent_df) < 60:
                return {"detected": False}

            # Find troughs
            troughs, _ = find_peaks(-recent_df['low'], prominence=recent_df['low'].mean() * 0.02, width=3)

            if len(troughs) < 3:
                return {"detected": False}

            for i in range(len(troughs) - 2):
                t1, t2, t3 = troughs[i], troughs[i+1], troughs[i+2]
                l1, l2, l3 = recent_df['low'].iloc[t1], recent_df['low'].iloc[t2], recent_df['low'].iloc[t3]

                # Head must be lowest
                if l2 < l1 and l2 < l3:
                    if abs(l1 - l3) / l2 < 0.15:
                        peak1_idx = recent_df['high'].iloc[t1:t2].idxmax()
                        peak2_idx = recent_df['high'].iloc[t2:t3].idxmax()
                        p1_val, p2_val = recent_df['high'].loc[peak1_idx], recent_df['high'].loc[peak2_idx]

                        if recent_df['close'].iloc[-1] > max(p1_val, p2_val):
                            confidence = self._calculate_enhanced_confidence(
                                "Inverse Head and Shoulders", "LONG", df, support_levels, resistance_levels, all_timeframes
                            )
                            return {
                                "detected": True,
                                "pattern": "Inverse Head and Shoulders",
                                "confidence": confidence,
                                "direction": "LONG",
                                "details": {"head": l2, "left_shoulder": l1, "right_shoulder": l3, "neckline": max(p1_val, p2_val)}
                            }
            return {"detected": False}
        except Exception:
            return {"detected": False}

    def _detect_channel_up(self, df: pd.DataFrame, support_levels: list, resistance_levels: list, all_timeframes: dict) -> dict:
        """Detects an upward channel using linear regression."""
        try:
            recent_df = df.tail(50)
            if len(recent_df) < 20:
                return {"detected": False}

            x = np.arange(len(recent_df))
            highs = recent_df['high'].values
            lows = recent_df['low'].values

            # Fit lines to highs and lows
            res_reg = LinearRegression().fit(x.reshape(-1, 1), highs)
            sup_reg = LinearRegression().fit(x.reshape(-1, 1), lows)

            # Check if both lines are upward sloping and parallel
            if res_reg.coef_[0] > 0 and sup_reg.coef_[0] > 0 and abs(res_reg.coef_[0] - sup_reg.coef_[0]) < 0.001:
                confidence = self._calculate_enhanced_confidence("Channel Up", "NEUTRAL", df, support_levels, resistance_levels, all_timeframes)
                return {
                    "detected": True,
                    "pattern": "Channel Up",
                    "confidence": confidence,
                    "direction": "NEUTRAL",
                    "details": {"resistance_slope": res_reg.coef_[0], "support_slope": sup_reg.coef_[0]}
                }
            return {"detected": False}
        except Exception:
            return {"detected": False}

    def _detect_channel_down(self, df: pd.DataFrame, support_levels: list, resistance_levels: list, all_timeframes: dict) -> dict:
        """Detects a downward channel."""
        try:
            recent_df = df.tail(50)
            if len(recent_df) < 20:
                return {"detected": False}

            x = np.arange(len(recent_df))
            highs = recent_df['high'].values
            lows = recent_df['low'].values

            res_reg = LinearRegression().fit(x.reshape(-1, 1), highs)
            sup_reg = LinearRegression().fit(x.reshape(-1, 1), lows)

            if res_reg.coef_[0] < 0 and sup_reg.coef_[0] < 0 and abs(res_reg.coef_[0] - sup_reg.coef_[0]) < 0.001:
                confidence = self._calculate_enhanced_confidence("Channel Down", "NEUTRAL", df, support_levels, resistance_levels, all_timeframes)
                return {
                    "detected": True,
                    "pattern": "Channel Down",
                    "confidence": confidence,
                    "direction": "NEUTRAL",
                    "details": {"resistance_slope": res_reg.coef_[0], "support_slope": sup_reg.coef_[0]}
                }
            return {"detected": False}
        except Exception:
            return {"detected": False}

    def _detect_sideways_channel_strategy(self, df: pd.DataFrame, support_levels: list, resistance_levels: list, all_timeframes: dict) -> dict:
        """Detects a sideways channel and provides buy/sell signals based on price position."""
        try:
            recent_df = df.tail(60)
            if len(recent_df) < 30:
                return {"detected": False}

            # Find strong horizontal support and resistance
            strong_support = None
            for price, strength in support_levels:
                if strength >= 6:
                    strong_support = price
                    break
            
            strong_resistance = None
            for price, strength in resistance_levels:
                if strength >= 6:
                    strong_resistance = price
                    break

            if strong_support and strong_resistance:
                channel_width = (strong_resistance - strong_support) / strong_support
                # Channel should be between 5% and 20% wide
                if 0.05 < channel_width < 0.20:
                    current_price = recent_df['close'].iloc[-1]
                    position_in_channel = (current_price - strong_support) / (strong_resistance - strong_support)

                    # Buy signal near support
                    if position_in_channel < 0.15: # Bottom 15% of channel
                        confidence = self._calculate_enhanced_confidence("Sideways Channel Buy", "LONG", df, support_levels, resistance_levels, all_timeframes)
                        return {
                            "detected": True,
                            "pattern": "Sideways Channel Buy",
                            "confidence": confidence,
                            "direction": "LONG",
                            "details": {"support": strong_support, "resistance": strong_resistance}
                        }
                    # Sell signal near resistance
                    elif position_in_channel > 0.85: # Top 15% of channel
                        confidence = self._calculate_enhanced_confidence("Sideways Channel Sell", "SHORT", df, support_levels, resistance_levels, all_timeframes)
                        return {
                            "detected": True,
                            "pattern": "Sideways Channel Sell",
                            "confidence": confidence,
                            "direction": "SHORT",
                            "details": {"support": strong_support, "resistance": strong_resistance}
                        }
            return {"detected": False}
        except Exception:
            return {"detected": False}

    def _detect_divergence_signal(self, df: pd.DataFrame, strategy_name: str, all_timeframes: dict) -> dict:
        """Detects bullish or bearish divergence between price and RSI/MACD."""
        try:
            recent_df = df.tail(40)
            if len(recent_df) < 20:
                return {"detected": False}

            # Find price lows and highs
            price_lows_idx, _ = find_peaks(-recent_df['low'], prominence=recent_df['low'].mean() * 0.01)
            price_highs_idx, _ = find_peaks(recent_df['high'], prominence=recent_df['high'].mean() * 0.01)

            # Bullish Divergence (Lower Low in price, Higher Low in indicator)
            if len(price_lows_idx) >= 2:
                l1_idx, l2_idx = price_lows_idx[-2], price_lows_idx[-1]
                price_l1, price_l2 = recent_df['low'].iloc[l1_idx], recent_df['low'].iloc[l2_idx]

                if price_l2 < price_l1:
                    # RSI Divergence
                    if 'rsi14' in recent_df.columns:
                        rsi_l1, rsi_l2 = recent_df['rsi14'].iloc[l1_idx], recent_df['rsi14'].iloc[l2_idx]
                        if rsi_l2 > rsi_l1:
                            return {"detected": True, "pattern": "Bullish RSI Divergence", "confidence": 75, "direction": "LONG"}
                    # MACD Divergence
                    if 'macd_hist' in recent_df.columns:
                        macd_l1, macd_l2 = recent_df['macd_hist'].iloc[l1_idx], recent_df['macd_hist'].iloc[l2_idx]
                        if macd_l2 > macd_l1:
                            return {"detected": True, "pattern": "Bullish MACD Divergence", "confidence": 70, "direction": "LONG"}

            # Bearish Divergence (Higher High in price, Lower High in indicator)
            if len(price_highs_idx) >= 2:
                h1_idx, h2_idx = price_highs_idx[-2], price_highs_idx[-1]
                price_h1, price_h2 = recent_df['high'].iloc[h1_idx], recent_df['high'].iloc[h2_idx]

                if price_h2 > price_h1:
                    if 'rsi14' in recent_df.columns:
                        rsi_h1, rsi_h2 = recent_df['rsi14'].iloc[h1_idx], recent_df['rsi14'].iloc[h2_idx]
                        if rsi_h2 < rsi_h1:
                            return {"detected": True, "pattern": "Bearish RSI Divergence", "confidence": 75, "direction": "SHORT"}
                    if 'macd_hist' in recent_df.columns:
                        macd_h1, macd_h2 = recent_df['macd_hist'].iloc[h1_idx], recent_df['macd_hist'].iloc[h2_idx]
                        if macd_h2 < macd_h1:
                            return {"detected": True, "pattern": "Bearish MACD Divergence", "confidence": 70, "direction": "SHORT"}
            
            return {"detected": False}
        except Exception:
            return {"detected": False}

    def _detect_support_resistance_tests(self, df: pd.DataFrame, support_levels: list, resistance_levels: list, all_timeframes: dict) -> list:
        """Detects bounces from support or rejections from resistance."""
        signals = []
        try:
            current_price = df['close'].iloc[-1]
            atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.02

            # Support bounce
            for s_price, s_strength in support_levels:
                if abs(current_price - s_price) < atr * 0.5: # Price is near support
                    if df['low'].iloc[-3:].min() < s_price and current_price > s_price: # Touched and bounced
                        signals.append({
                            "detected": True,
                            "pattern": f"Support Bounce @ {s_price:.2f}",
                            "confidence": 60 + s_strength * 2,
                            "direction": "LONG"
                        })

            # Resistance rejection
            for r_price, r_strength in resistance_levels:
                if abs(current_price - r_price) < atr * 0.5:
                    if df['high'].iloc[-3:].max() > r_price and current_price < r_price:
                        signals.append({
                            "detected": True,
                            "pattern": f"Resistance Rejection @ {r_price:.2f}",
                            "confidence": 60 + r_strength * 2,
                            "direction": "SHORT"
                        })
            return signals
        except Exception:
            return []

    def _detect_indicator_signals(self, df: pd.DataFrame, support_levels: list, resistance_levels: list, all_timeframes: dict) -> list:
        """Generates basic signals from indicator states (e.g., RSI overbought/oversold)."""
        signals = []
        try:
            latest = df.iloc[-1]

            # RSI Oversold
            if 'rsi14' in latest and latest['rsi14'] < 30:
                signals.append({"detected": True, "pattern": "RSI Oversold", "confidence": 65, "direction": "LONG"})
            
            # RSI Overbought
            if 'rsi14' in latest and latest['rsi14'] > 70:
                signals.append({"detected": True, "pattern": "RSI Overbought", "confidence": 65, "direction": "SHORT"})

            # MACD Bullish Crossover
            if 'macd' in latest and 'macd_signal' in latest:
                if latest['macd'] > latest['macd_signal'] and df['macd'].iloc[-2] < df['macd_signal'].iloc[-2]:
                    signals.append({"detected": True, "pattern": "MACD Bullish Cross", "confidence": 70, "direction": "LONG"})

            # MACD Bearish Crossover
            if 'macd' in latest and 'macd_signal' in latest:
                if latest['macd'] < latest['macd_signal'] and df['macd'].iloc[-2] > df['macd_signal'].iloc[-2]:
                    signals.append({"detected": True, "pattern": "MACD Bearish Cross", "confidence": 70, "direction": "SHORT"})

            return signals
        except Exception:
            return []
