"""
Module 11: Signal Generator
Generates trading signals with optimal entry points, multi-layered targets, and dynamic stops.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from datetime import datetime
import csv


class SignalGenerator:
    """
    Professional signal generator with hierarchical targeting and risk management.

    Features:
    - Optimal entry calculation with S/R consideration
    - Multi-source target generation (R:R, S/R, Fibonacci, measured moves)
    - Dynamic stop-loss with structural anchors
    - Quality scoring and confidence calculation
    - CSV export functionality
    """

    def __init__(self):
        """Initialize signal generator."""
        self.min_rr_ratio = 1.5  # Minimum risk-reward ratio
        self.max_targets = 10  # Maximum number of targets

    def generate_signals(self, 
                        df: pd.DataFrame,
                        pattern_result: Dict[str, Any],
                        support_levels: List[Tuple[float, int]],
                        resistance_levels: List[Tuple[float, int]],
                        all_timeframes: Dict[str, pd.DataFrame],
                        smart_money_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive trading signal.

        Args:
            df: Main timeframe dataframe with indicators
            pattern_result: Chart pattern detection results
            support_levels: List of (price, strength) support tuples
            resistance_levels: List of (price, strength) resistance tuples
            all_timeframes: Dictionary of dataframes for each timeframe
            smart_money_data: Smart money analysis results

        Returns:
            Complete signal dictionary
        """
        try:
            # Extract current price and ATR
            current_price = float(df['close'].iloc[-1])
            atr = float(df['atr'].iloc[-1]) if 'atr' in df.columns else current_price * 0.02

            # Determine direction from pattern
            direction = pattern_result.get('direction', 'NEUTRAL')
            if direction == 'NEUTRAL':
                # Fallback to trend analysis
                direction = self._determine_direction_from_trend(df)

            # Calculate optimal entry point
            pattern_name = pattern_result.get('pattern', 'General')
            optimal_entry = self.calculate_optimal_entry(
                current_price, direction, pattern_name, 
                support_levels, resistance_levels, atr
            )

            # Calculate stop-loss
            stop_loss = self._calculate_dynamic_stop(
                df, current_price, direction, atr, 
                support_levels, resistance_levels, pattern_result
            )

            # Generate targets
            targets = self._generate_hierarchical_targets(
                df, current_price, stop_loss, direction,
                support_levels, resistance_levels, pattern_result
            )

            # Calculate risk-reward ratios
            rr_ratios = self._calculate_risk_reward(current_price, stop_loss, targets, direction)

            # Update targets with R:R ratios
            for i, target in enumerate(targets):
                if i < len(rr_ratios):
                    target['rr_ratio'] = rr_ratios[i]

            # Calculate quality score
            quality_score = self._calculate_quality_score(
                pattern_result, direction, df, support_levels, 
                resistance_levels, all_timeframes, rr_ratios
            )

            # Calculate signal confidence
            confidence = self._calculate_signal_confidence(
                pattern_result, direction, df, support_levels,
                resistance_levels, all_timeframes, smart_money_data
            )

            # Build signal dictionary
            signal = {
                'detected_pattern': pattern_name,
                'direction': direction,
                'confidence': confidence,
                'quality_score': quality_score,
                'current_price': current_price,
                'optimal_entry': optimal_entry,
                'entry_range': (current_price * 0.99, current_price * 1.01),
                'stop_loss': stop_loss,
                'targets': targets,
                'risk_reward_ratio': rr_ratios[0] if rr_ratios else 0,
                'dow_phase': smart_money_data.get('dow_phase', 'Unknown'),
                'support_levels': support_levels[:5],  # Top 5
                'resistance_levels': resistance_levels[:5],  # Top 5
                'pattern_details': pattern_result.get('details', {})
            }

            return signal

        except Exception as e:
            print(f"Error generating signal: {e}")
            return {'error': str(e)}

    def calculate_optimal_entry(self,
                               current_price: float,
                               direction: str,
                               pattern: str,
                               support_levels: List[Tuple[float, int]],
                               resistance_levels: List[Tuple[float, int]],
                               atr: float) -> float:
        """
        Calculate volume-weighted optimal entry with S/R consideration.

        Args:
            current_price: Current market price
            direction: Trade direction (LONG/SHORT)
            pattern: Detected pattern name
            support_levels: List of support levels with strength
            resistance_levels: List of resistance levels with strength
            atr: Average True Range

        Returns:
            Optimal entry price
        """
        try:
            optimal_entry = current_price

            if direction == "LONG":
                # For bullish patterns, wait for pullback to support
                if pattern in ["Double Bottom", "Inverse H&S", "Inverse Head and Shoulders"]:
                    # Find strongest support below current price
                    strong_supports = [lvl for lvl, strg in support_levels 
                                     if strg >= 4 and lvl < current_price]
                    if strong_supports:
                        key_support = max(strong_supports)
                        # Entry at 33% retracement to support
                        optimal_entry = key_support + (current_price - key_support) * 0.33
                    else:
                        optimal_entry = current_price - atr * 0.5

                elif pattern in ["Ascending Triangle", "Bull Flag"]:
                    # Wait for slight pullback
                    optimal_entry = current_price - atr * 0.3

                else:
                    # Default: half ATR below current price
                    optimal_entry = current_price - atr * 0.5

            elif direction == "SHORT":
                # For bearish patterns, wait for bounce to resistance
                if pattern in ["Double Top", "Head and Shoulders"]:
                    # Find strongest resistance above current price
                    strong_resistances = [lvl for lvl, strg in resistance_levels 
                                        if strg >= 4 and lvl > current_price]
                    if strong_resistances:
                        key_resistance = min(strong_resistances)
                        # Entry at 66% bounce to resistance
                        optimal_entry = key_resistance - (key_resistance - current_price) * 0.66
                    else:
                        optimal_entry = current_price + atr * 0.5

                elif pattern in ["Descending Triangle", "Bear Flag"]:
                    # Wait for slight bounce
                    optimal_entry = current_price + atr * 0.3

                else:
                    # Default: half ATR above current price
                    optimal_entry = current_price + atr * 0.5

            # Clamp entry within reasonable range
            if direction == "LONG":
                entry_range_low = current_price * 0.99
                optimal_entry = max(optimal_entry, entry_range_low)
                optimal_entry = min(optimal_entry, current_price)
            elif direction == "SHORT":
                entry_range_high = current_price * 1.01
                optimal_entry = min(optimal_entry, entry_range_high)
                optimal_entry = max(optimal_entry, current_price)

            return optimal_entry

        except Exception as e:
            print(f"Error calculating optimal entry: {e}")
            return current_price

    def _calculate_dynamic_stop(self,
                               df: pd.DataFrame,
                               current_price: float,
                               direction: str,
                               atr: float,
                               support_levels: List[Tuple[float, int]],
                               resistance_levels: List[Tuple[float, int]],
                               pattern_result: Dict[str, Any]) -> float:
        """
        Calculate dynamic stop-loss with structural anchors.

        Priority:
        1. Pattern invalidation level (neckline, breakout level)
        2. Structural swing points (recent highs/lows)
        3. Strong S/R levels with ATR buffer
        4. ATR-based fallback (1.5x-2.5x ATR)
        """
        try:
            stop_loss = 0

            if direction == "LONG":
                # Method 1: Pattern-specific stops
                pattern_details = pattern_result.get('details', {})
                if 'neckline' in pattern_details:
                    # For H&S or Double Bottom, stop below neckline
                    neckline = pattern_details['neckline']
                    stop_loss = neckline - atr * 1.0
                elif 'breakout_level' in pattern_details:
                    # For triangles/wedges, stop below breakout
                    stop_loss = pattern_details['breakout_level'] - atr * 1.0

                # Method 2: Structural anchor (swing low)
                if stop_loss == 0:
                    structural_anchor = self._find_structural_swing_low(df, current_price)
                    if structural_anchor:
                        stop_loss = structural_anchor - atr * 1.5

                # Method 3: Strong support with buffer
                if stop_loss == 0:
                    strong_supports = [lvl for lvl, strg in support_levels 
                                     if strg >= 5 and lvl < current_price]
                    if strong_supports:
                        closest_support = max(strong_supports)
                        stop_loss = closest_support - atr * 1.5

                # Method 4: ATR-based fallback
                if stop_loss == 0:
                    stop_loss = current_price - atr * 2.5

            elif direction == "SHORT":
                # Method 1: Pattern-specific stops
                pattern_details = pattern_result.get('details', {})
                if 'neckline' in pattern_details:
                    neckline = pattern_details['neckline']
                    stop_loss = neckline + atr * 1.0
                elif 'breakout_level' in pattern_details:
                    stop_loss = pattern_details['breakout_level'] + atr * 1.0

                # Method 2: Structural anchor (swing high)
                if stop_loss == 0:
                    structural_anchor = self._find_structural_swing_high(df, current_price)
                    if structural_anchor:
                        stop_loss = structural_anchor + atr * 1.5

                # Method 3: Strong resistance with buffer
                if stop_loss == 0:
                    strong_resistances = [lvl for lvl, strg in resistance_levels 
                                        if strg >= 5 and lvl > current_price]
                    if strong_resistances:
                        closest_resistance = min(strong_resistances)
                        stop_loss = closest_resistance + atr * 1.5

                # Method 4: ATR-based fallback
                if stop_loss == 0:
                    stop_loss = current_price + atr * 2.5

            # Validate risk (1-8% range)
            risk = abs(current_price - stop_loss)
            min_risk = current_price * 0.01
            max_risk = current_price * 0.08

            if risk < min_risk:
                if direction == "LONG":
                    stop_loss = current_price - min_risk
                else:
                    stop_loss = current_price + min_risk
            elif risk > max_risk:
                if direction == "LONG":
                    stop_loss = current_price - max_risk
                else:
                    stop_loss = current_price + max_risk

            return stop_loss

        except Exception as e:
            print(f"Error calculating stop-loss: {e}")
            return current_price * 0.97 if direction == "LONG" else current_price * 1.03

    def _generate_hierarchical_targets(self,
                                      df: pd.DataFrame,
                                      current_price: float,
                                      stop_loss: float,
                                      direction: str,
                                      support_levels: List[Tuple[float, int]],
                                      resistance_levels: List[Tuple[float, int]],
                                      pattern_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate multi-source hierarchical targets.

        Sources (in priority order):
        1. Risk-reward multiples (1.5R, 2R, 2.5R, 3R, 4R, 5R)
        2. Strong S/R levels
        3. Pattern measured moves
        4. Fibonacci extensions (1.272, 1.618, 2.0, 2.618)
        """
        potential_targets = []
        risk = abs(current_price - stop_loss)

        if risk == 0:
            return []

        # Source 1: R:R multiples
        rr_multiples = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
        for r_multiple in rr_multiples:
            if direction == "LONG":
                price = current_price + risk * r_multiple
            else:
                price = current_price - risk * r_multiple

            potential_targets.append({
                'price': price,
                'source': 'risk_multiple',
                'reason': f'{r_multiple}R Target',
                'probability': max(30, 85 - int(r_multiple * 10))
            })

        # Source 2: Structural S/R levels
        levels_to_check = resistance_levels if direction == "LONG" else support_levels
        for level, strength in levels_to_check[:10]:
            if strength >= 4:
                # Check if level is in valid direction
                if (direction == "LONG" and level > current_price * 1.01) or                    (direction == "SHORT" and level < current_price * 0.99):
                    potential_targets.append({
                        'price': level,
                        'source': 'structure',
                        'reason': f'Structural Level (Strength {strength})',
                        'probability': min(90, 50 + strength * 5)
                    })

        # Source 3: Pattern measured moves
        pattern_details = pattern_result.get('details', {})
        if pattern_details and pattern_result.get('detected', False):
            measured_targets = self._calculate_measured_move_targets(
                pattern_result, current_price, direction
            )
            potential_targets.extend(measured_targets)

        # Source 4: Fibonacci extensions
        fib_targets = self._calculate_fibonacci_targets(df, current_price, direction)
        potential_targets.extend(fib_targets)

        # Cluster and finalize targets
        final_targets = self._cluster_and_finalize_targets(
            potential_targets, current_price, direction
        )

        return final_targets[:self.max_targets]

    def _calculate_measured_move_targets(self,
                                        pattern_result: Dict[str, Any],
                                        current_price: float,
                                        direction: str) -> List[Dict[str, Any]]:
        """Calculate pattern-specific measured move targets."""
        targets = []
        pattern_details = pattern_result.get('details', {})
        pattern_name = pattern_result.get('pattern', '')

        try:
            if 'Head and Shoulders' in pattern_name or 'H&S' in pattern_name:
                if 'head_price' in pattern_details and 'neckline' in pattern_details:
                    head = pattern_details['head_price']
                    neckline = pattern_details['neckline']
                    height = abs(head - neckline)

                    # Standard measured moves: 50%, 100%, 161.8%
                    ratios = [0.5, 1.0, 1.618]
                    for i, ratio in enumerate(ratios):
                        if direction == "LONG":
                            target_price = current_price + height * ratio
                        else:
                            target_price = current_price - height * ratio

                        targets.append({
                            'price': target_price,
                            'source': 'measured_move',
                            'reason': f'H&S {int(ratio*100)}% Measured Move',
                            'probability': 80 - i * 10
                        })

            elif 'Double' in pattern_name:
                if 'extreme_price' in pattern_details and 'neckline' in pattern_details:
                    extreme = pattern_details['extreme_price']
                    neckline = pattern_details['neckline']
                    height = abs(extreme - neckline)

                    ratios = [0.5, 1.0]
                    for i, ratio in enumerate(ratios):
                        if direction == "LONG":
                            target_price = current_price + height * ratio
                        else:
                            target_price = current_price - height * ratio

                        targets.append({
                            'price': target_price,
                            'source': 'measured_move',
                            'reason': f'Double {int(ratio*100)}% Measured Move',
                            'probability': 75 - i * 10
                        })

        except Exception as e:
            print(f"Error calculating measured moves: {e}")

        return targets

    def _calculate_fibonacci_targets(self,
                                    df: pd.DataFrame,
                                    current_price: float,
                                    direction: str) -> List[Dict[str, Any]]:
        """Calculate Fibonacci extension targets."""
        targets = []

        try:
            if len(df) < 30:
                return targets

            # Find recent swing high/low
            recent_high = float(df['high'].iloc[-30:].max())
            recent_low = float(df['low'].iloc[-30:].min())
            fib_range = recent_high - recent_low

            if fib_range <= 0:
                return targets

            # Fibonacci extension ratios
            fib_ratios = [1.272, 1.618, 2.0, 2.618]

            for i, ratio in enumerate(fib_ratios):
                if direction == "LONG":
                    fib_target = recent_high + fib_range * (ratio - 1.0)
                    if fib_target > current_price * 1.02:
                        targets.append({
                            'price': fib_target,
                            'source': 'fibonacci',
                            'reason': f'{ratio} Fib Extension',
                            'probability': 65 - i * 5
                        })
                else:
                    fib_target = recent_low - fib_range * (ratio - 1.0)
                    if fib_target < current_price * 0.98:
                        targets.append({
                            'price': fib_target,
                            'source': 'fibonacci',
                            'reason': f'{ratio} Fib Extension',
                            'probability': 65 - i * 5
                        })

        except Exception as e:
            print(f"Error calculating Fibonacci targets: {e}")

        return targets

    def _cluster_and_finalize_targets(self,
                                     potential_targets: List[Dict[str, Any]],
                                     current_price: float,
                                     direction: str) -> List[Dict[str, Any]]:
        """Cluster nearby targets and finalize list."""
        if not potential_targets:
            return []

        # Sort by price
        potential_targets.sort(key=lambda x: x['price'])

        # Cluster targets within 1.5% of each other
        clustered = []
        cluster_threshold = 0.015  # 1.5%

        i = 0
        while i < len(potential_targets):
            current_target = potential_targets[i]
            cluster_members = [current_target]

            # Find nearby targets
            j = i + 1
            while j < len(potential_targets):
                if abs(potential_targets[j]['price'] - current_target['price']) / current_target['price'] < cluster_threshold:
                    cluster_members.append(potential_targets[j])
                    j += 1
                else:
                    break

            # Create clustered target (weighted average)
            total_prob = sum(m['probability'] for m in cluster_members)
            avg_price = sum(m['price'] * m['probability'] for m in cluster_members) / total_prob if total_prob > 0 else current_target['price']
            avg_prob = total_prob / len(cluster_members)

            # Combine reasons
            reasons = list(set(m['reason'] for m in cluster_members))
            combined_reason = reasons[0] if len(reasons) == 1 else 'Multi-source confluence'

            clustered.append({
                'price': avg_price,
                'reason': combined_reason,
                'probability': min(95, avg_prob * 1.1),  # Boost for confluence
                'percent': (avg_price / current_price - 1) * 100
            })

            i = j

        # Sort and return
        if direction == "LONG":
            clustered.sort(key=lambda x: x['price'])
        else:
            clustered.sort(key=lambda x: x['price'], reverse=True)

        return clustered

    def _calculate_risk_reward(self,
                              entry: float,
                              stop_loss: float,
                              targets: List[Dict[str, Any]],
                              direction: str) -> List[float]:
        """Calculate risk-reward ratios for each target."""
        risk = abs(entry - stop_loss)
        if risk == 0:
            return [0] * len(targets)

        rr_ratios = []
        for target in targets:
            reward = abs(target['price'] - entry)
            rr_ratio = reward / risk
            rr_ratios.append(round(rr_ratio, 2))

        return rr_ratios

    def _calculate_quality_score(self,
                                pattern_result: Dict[str, Any],
                                direction: str,
                                df: pd.DataFrame,
                                support_levels: List[Tuple[float, int]],
                                resistance_levels: List[Tuple[float, int]],
                                all_timeframes: Dict[str, pd.DataFrame],
                                rr_ratios: List[float]) -> int:
        """
        Calculate overall signal quality score (0-100).

        Scoring components:
        - Multi-timeframe alignment: 30 points
        - Pattern quality: 20 points
        - S/R confluence: 15 points
        - Risk-reward ratio: 15 points
        - Volume confirmation: 10 points
        - Indicator alignment: 10 points
        """
        score = 0

        try:
            # 1. Multi-timeframe trend alignment (30 points)
            tf_score = self._score_timeframe_alignment(all_timeframes, direction)
            score += int(tf_score * 30)

            # 2. Pattern quality (20 points)
            pattern_quality = pattern_result.get('quality', 0)
            score += int((pattern_quality / 100) * 20)

            # 3. S/R confluence (15 points)
            sr_score = self._score_sr_confluence(support_levels, resistance_levels)
            score += int(sr_score * 15)

            # 4. Risk-reward ratio (15 points)
            if rr_ratios:
                best_rr = max(rr_ratios[:3])  # Best of first 3 targets
                if best_rr >= 3.0:
                    score += 15
                elif best_rr >= 2.0:
                    score += 12
                elif best_rr >= 1.5:
                    score += 8

            # 5. Volume confirmation (10 points)
            volume_score = self._score_volume_confirmation(df, direction)
            score += int(volume_score * 10)

            # 6. Indicator alignment (10 points)
            indicator_score = self._score_indicator_alignment(df, direction)
            score += int(indicator_score * 10)

        except Exception as e:
            print(f"Error calculating quality score: {e}")

        return min(100, max(0, score))

    def _calculate_signal_confidence(self,
                                    pattern_result: Dict[str, Any],
                                    direction: str,
                                    df: pd.DataFrame,
                                    support_levels: List[Tuple[float, int]],
                                    resistance_levels: List[Tuple[float, int]],
                                    all_timeframes: Dict[str, pd.DataFrame],
                                    smart_money_data: Dict[str, Any]) -> int:
        """Calculate signal confidence (0-100)."""
        base_confidence = pattern_result.get('confidence', 60)

        # Adjust for trend alignment
        tf_alignment = self._score_timeframe_alignment(all_timeframes, direction)
        confidence_adjustment = int((tf_alignment - 0.5) * 20)

        # Adjust for Dow Theory phase
        dow_phase = smart_money_data.get('dow_phase', 'Unknown')
        if direction == 'LONG' and dow_phase == 'Accumulation':
            confidence_adjustment += 10
        elif direction == 'SHORT' and dow_phase == 'Distribution':
            confidence_adjustment += 10

        final_confidence = base_confidence + confidence_adjustment
        return min(100, max(0, final_confidence))

    def _determine_direction_from_trend(self, df: pd.DataFrame) -> str:
        """Determine trade direction from trend analysis."""
        try:
            if 'ema_20' in df.columns and 'ema_50' in df.columns:
                ema20 = float(df['ema_20'].iloc[-1])
                ema50 = float(df['ema_50'].iloc[-1])

                if ema20 > ema50 * 1.02:
                    return 'LONG'
                elif ema20 < ema50 * 0.98:
                    return 'SHORT'
        except:
            pass

        return 'NEUTRAL'

    def _find_structural_swing_low(self, df: pd.DataFrame, current_price: float) -> Optional[float]:
        """Find recent structural swing low below current price."""
        try:
            if len(df) < 20:
                return None

            lows = df['low'].iloc[-20:].values
            for i in range(len(lows) - 2, 0, -1):
                if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                    swing_low = float(lows[i])
                    if swing_low < current_price:
                        return swing_low
        except:
            pass

        return None

    def _find_structural_swing_high(self, df: pd.DataFrame, current_price: float) -> Optional[float]:
        """Find recent structural swing high above current price."""
        try:
            if len(df) < 20:
                return None

            highs = df['high'].iloc[-20:].values
            for i in range(len(highs) - 2, 0, -1):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    swing_high = float(highs[i])
                    if swing_high > current_price:
                        return swing_high
        except:
            pass

        return None

    def _score_timeframe_alignment(self, all_timeframes: Dict[str, pd.DataFrame], direction: str) -> float:
        """Score multi-timeframe trend alignment (0-1)."""
        if not all_timeframes:
            return 0.5

        aligned_count = 0
        total_count = 0

        for tf, df in all_timeframes.items():
            if 'ema_20' in df.columns and 'ema_50' in df.columns and len(df) > 0:
                ema20 = float(df['ema_20'].iloc[-1])
                ema50 = float(df['ema_50'].iloc[-1])

                if direction == 'LONG' and ema20 > ema50:
                    aligned_count += 1
                elif direction == 'SHORT' and ema20 < ema50:
                    aligned_count += 1

                total_count += 1

        return aligned_count / total_count if total_count > 0 else 0.5

    def _score_sr_confluence(self, support_levels: List[Tuple[float, int]], 
                            resistance_levels: List[Tuple[float, int]]) -> float:
        """Score S/R level confluence (0-1)."""
        if not support_levels and not resistance_levels:
            return 0.3

        # Count high-strength levels
        strong_supports = sum(1 for _, strg in support_levels if strg >= 5)
        strong_resistances = sum(1 for _, strg in resistance_levels if strg >= 5)
        total_strong = strong_supports + strong_resistances

        if total_strong >= 4:
            return 1.0
        elif total_strong >= 2:
            return 0.7
        else:
            return 0.4

    def _score_volume_confirmation(self, df: pd.DataFrame, direction: str) -> float:
        """Score volume confirmation (0-1)."""
        try:
            if 'volume' not in df.columns or len(df) < 20:
                return 0.5

            recent_volume = float(df['volume'].iloc[-1])
            avg_volume = float(df['volume'].iloc[-20:].mean())

            if recent_volume > avg_volume * 1.5:
                return 1.0
            elif recent_volume > avg_volume * 1.2:
                return 0.8
            elif recent_volume > avg_volume:
                return 0.6
            else:
                return 0.4
        except:
            return 0.5

    def _score_indicator_alignment(self, df: pd.DataFrame, direction: str) -> float:
        """Score technical indicator alignment (0-1)."""
        aligned_count = 0
        total_count = 0

        try:
            # RSI alignment
            if 'rsi_14' in df.columns:
                rsi = float(df['rsi_14'].iloc[-1])
                if direction == 'LONG' and rsi > 50:
                    aligned_count += 1
                elif direction == 'SHORT' and rsi < 50:
                    aligned_count += 1
                total_count += 1

            # MACD alignment
            if 'macd' in df.columns and 'macd_signal' in df.columns:
                macd = float(df['macd'].iloc[-1])
                signal = float(df['macd_signal'].iloc[-1])
                if direction == 'LONG' and macd > signal:
                    aligned_count += 1
                elif direction == 'SHORT' and macd < signal:
                    aligned_count += 1
                total_count += 1
        except:
            pass

        return aligned_count / total_count if total_count > 0 else 0.5

    def save_to_csv(self, signal: Dict[str, Any], csv_path: str):
        """
        Save signal to CSV file.

        Args:
            signal: Complete signal dictionary
            csv_path: Path to CSV file
        """
        try:
            if 'error' in signal:
                print(f"Cannot save signal with error: {signal['error']}")
                return

            # Extract data
            symbol = signal.get('symbol', 'UNKNOWN')
            timestamp = signal.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
            signal_date = dt.strftime('%Y-%m-%d')
            signal_time = dt.strftime('%H:%M')

            # Build row
            row = {
                'quality_score': signal.get('quality_score', 0),
                'confidence': signal.get('confidence', 0),
                'blank': '',
                'detected_pattern': signal.get('detected_pattern', 'Unknown'),
                'signal_date': signal_date,
                'signal_time': signal_time,
                'crypto_pair': f'{symbol}USDT',
                'direction': signal.get('direction', 'NEUTRAL'),
                'start_entry_range': signal.get('entry_range', (0, 0))[0],
                'end_entry_range': signal.get('entry_range', (0, 0))[1],
                'middle_of_entry_range': signal.get('optimal_entry', 0)
            }

            # Add targets (up to 10)
            targets = signal.get('targets', [])
            for i in range(1, 11):
                if i <= len(targets):
                    row[f'Target{i}'] = targets[i-1].get('price', '')
                    row[f'Target{i}Probability'] = targets[i-1].get('probability', '')
                    row[f'Target{i}Reason'] = targets[i-1].get('reason', '')
                else:
                    row[f'Target{i}'] = ''
                    row[f'Target{i}Probability'] = ''
                    row[f'Target{i}Reason'] = ''

            # Add stop-loss and R:R
            row['StopLoss'] = signal.get('stop_loss', 0)
            row['RiskRewardRatio'] = signal.get('risk_reward_ratio', 0)

            # Add trends
            trends = signal.get('trends', {})
            row['1hTrend'] = trends.get('1h', {}).get('trend', 'NEUTRAL')
            row['4hTrend'] = trends.get('4h', {}).get('trend', 'NEUTRAL')
            row['1dTrend'] = trends.get('1d', {}).get('trend', 'NEUTRAL')
            row['1wTrend'] = trends.get('1w', {}).get('trend', 'NEUTRAL')
            row['dow_phase'] = signal.get('dow_phase', 'Unknown')

            # Write to CSV
            import os
            file_exists = os.path.isfile(csv_path)

            with open(csv_path, 'a', newline='') as csvfile:
                fieldnames = list(row.keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                if not file_exists:
                    writer.writeheader()

                writer.writerow(row)

            print(f"Signal for {symbol} saved to CSV.")

        except Exception as e:
            print(f"Error saving signal to CSV: {e}")
