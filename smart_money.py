"""
Module 9: Smart Money Analysis
Implements Smart Money Concepts (SMC) and Dow Theory phase identification.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


class SmartMoneyAnalyzer:
    """
    Smart Money and Dow Theory analyzer.

    Detects:
    - Accumulation and Distribution phases
    - Dow Theory market cycles
    - Institutional activity patterns
    - Money flow analysis
    """

    def __init__(self):
        """Initialize Smart Money Analyzer."""
        pass

    def analyze_money_flow(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Main entry point for smart money analysis.

        Args:
            df: DataFrame with OHLCV and indicator data

        Returns:
            Dictionary with Dow phase, smart money signals, and confidence
        """
        try:
            if df is None or len(df) < 50:
                return {
                    'dow_phase': 'Unknown',
                    'confidence': 0,
                    'accumulation_signals': [],
                    'distribution_signals': []
                }

            # Get Dow Theory phase
            dow_phase = self.dow_theory_phase(df)

            # Get smart money activity signals
            smart_signals = self.detect_smart_money_activity(df)

            # Calculate money flow index
            mfi_data = self.calculate_money_flow_analysis(df)

            # Determine confidence based on signal strength
            confidence = smart_signals.get('confidence', 0)
            if mfi_data.get('status') == 'success':
                mfi = mfi_data.get('mfi', 50)
                # Adjust confidence based on MFI
                if dow_phase == 'Accumulation' and mfi < 30:
                    confidence += 15
                elif dow_phase == 'Distribution' and mfi > 70:
                    confidence += 15

            return {
                'dow_phase': dow_phase,
                'confidence': min(100, confidence),
                'smart_money_signals': smart_signals,
                'money_flow_index': mfi_data.get('mfi', 50),
                'buying_pressure': mfi_data.get('buying_pressure', 50),
                'selling_pressure': mfi_data.get('selling_pressure', 50),
                'accumulation_signals': self._get_accumulation_signals(smart_signals, dow_phase),
                'distribution_signals': self._get_distribution_signals(smart_signals, dow_phase)
            }

        except Exception as e:
            print(f"Error in analyze_money_flow: {e}")
            return {
                'dow_phase': 'Unknown',
                'confidence': 0,
                'accumulation_signals': [],
                'distribution_signals': []
            }

    def dow_theory_phase(self, df: pd.DataFrame) -> str:
        """
        Determine Dow Theory market phase.

        Phases:
        1. Accumulation - Smart money buying at low prices
        2. Markup - Public participation, prices rising
        3. Distribution - Smart money selling at highs
        4. Markdown - Public selling, prices declining

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Phase string
        """
        try:
            if len(df) < 100:
                return 'Undefined'

            # Analyze phase indicators
            phase_indicators = self._analyze_phase_indicators(df)

            # Get the dominant phase
            phases = {
                'Accumulation': phase_indicators.get('accumulation_score', 0),
                'Markup': phase_indicators.get('markup_score', 0),
                'Distribution': phase_indicators.get('distribution_score', 0),
                'Markdown': phase_indicators.get('markdown_score', 0)
            }

            max_phase = max(phases, key=phases.get)
            max_score = phases[max_phase]

            # Require minimum score threshold
            if max_score < 40:
                return 'Undefined'

            return max_phase

        except Exception as e:
            print(f"Error in dow_theory_phase: {e}")
            return 'Undefined'

    def _analyze_phase_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze indicators for each Dow Theory phase."""
        try:
            recent_df = df.tail(60).copy()

            # Price analysis
            current_price = float(recent_df['close'].iloc[-1])
            price_high = float(recent_df['high'].max())
            price_low = float(recent_df['low'].min())
            price_range = price_high - price_low

            if price_range == 0:
                price_position = 0.5
            else:
                price_position = (current_price - price_low) / price_range

            # Volume analysis
            recent_volume = recent_df['volume'].iloc[-10].mean()
            longer_volume = recent_df['volume'].iloc[-30:].mean()
            volume_ratio = recent_volume / longer_volume if longer_volume > 0 else 1

            # Price momentum
            price_5d_ago = float(recent_df['close'].iloc[-6]) if len(recent_df) >= 6 else current_price
            price_momentum = (current_price - price_5d_ago) / price_5d_ago if price_5d_ago > 0 else 0

            # Smart money signals
            smart_signals = self.detect_smart_money_activity(df)

            # Initialize phase scores
            indicators = {
                'accumulation_score': 0,
                'markup_score': 0,
                'distribution_score': 0,
                'markdown_score': 0
            }

            # 1. Accumulation scoring
            if price_position < 0.3:  # Near lows
                indicators['accumulation_score'] += 30
            if smart_signals.get('accumulation', False):
                indicators['accumulation_score'] += 25
            if volume_ratio < 0.8:  # Decreasing volume
                indicators['accumulation_score'] += 20
            if -0.1 < price_momentum < 0.05:  # Stable/slightly up
                indicators['accumulation_score'] += 25

            # 2. Markup scoring
            if price_momentum > 0.05:  # Rising prices
                indicators['markup_score'] += 35
            if volume_ratio > 1.2:  # Increasing volume
                indicators['markup_score'] += 30
            if 0.3 < price_position < 0.8:  # Mid-range prices
                indicators['markup_score'] += 20
            if 'rsi_14' in recent_df.columns and 50 < recent_df['rsi_14'].iloc[-1] < 75:
                indicators['markup_score'] += 15

            # 3. Distribution scoring
            if price_position > 0.7:  # Near highs
                indicators['distribution_score'] += 30
            if volume_ratio > 1.1 and abs(price_momentum) < 0.03:  # Volume up, price flat
                indicators['distribution_score'] += 35
            if smart_signals.get('distribution', False):
                indicators['distribution_score'] += 25
            if 'rsi_14' in recent_df.columns and recent_df['rsi_14'].iloc[-1] > 70:
                indicators['distribution_score'] += 10

            # 4. Markdown scoring
            if price_momentum < -0.05:  # Declining prices
                indicators['markdown_score'] += 35
            if volume_ratio > 1.3:  # High volume
                indicators['markdown_score'] += 30
            if price_position < 0.6:  # Below mid-range
                indicators['markdown_score'] += 20
            if 'rsi_14' in recent_df.columns and recent_df['rsi_14'].iloc[-1] < 30:
                indicators['markdown_score'] += 15

            return indicators

        except Exception as e:
            print(f"Error analyzing phase indicators: {e}")
            return {
                'accumulation_score': 0,
                'markup_score': 0,
                'distribution_score': 0,
                'markdown_score': 0
            }

    def detect_smart_money_activity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect potential smart money/institutional activity.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dictionary with smart money signals
        """
        try:
            signals = {
                'accumulation': False,
                'distribution': False,
                'unusual_activity': False,
                'stealth_buying': False,
                'whale_activity': False,
                'strength': 0,
                'confidence': 0
            }

            if len(df) < 20:
                return signals

            # Recent volume analysis
            recent_volume = df['volume'].iloc[-5:]
            avg_volume_20 = df['volume'].iloc[-20:].mean()
            avg_volume_10 = df['volume'].iloc[-10:].mean()

            # Price analysis
            recent_prices = df['close'].iloc[-10:]
            price_change_5d = (df['close'].iloc[-1] - df['close'].iloc[-6]) / df['close'].iloc[-6] if len(df) >= 6 else 0
            price_volatility = recent_prices.std() / recent_prices.mean() if recent_prices.mean() > 0 else 0

            # Volume patterns
            volume_consistency = np.sum(recent_volume > avg_volume_20 * 1.1) / len(recent_volume)

            # Stealth buying detection
            if (volume_consistency > 0.6 and 
                price_volatility < 0.03 and 
                abs(price_change_5d) < 0.05):
                signals['stealth_buying'] = True
                signals['strength'] += 4
                signals['confidence'] += 25

            # Whale activity detection
            max_recent_volume = recent_volume.max()
            volume_spike_ratio = max_recent_volume / avg_volume_20 if avg_volume_20 > 0 else 1

            if volume_spike_ratio > 3.0:
                spike_day_idx = recent_volume.idxmax()
                spike_day_price_change = (df.loc[spike_day_idx, 'close'] - df.loc[spike_day_idx, 'open']) / df.loc[spike_day_idx, 'open']

                if abs(spike_day_price_change) > 0.03:
                    signals['whale_activity'] = True
                    signals['strength'] += 5
                    signals['confidence'] += 30

                    if spike_day_price_change > 0:
                        signals['accumulation'] = True
                    else:
                        signals['distribution'] = True

            # Accumulation pattern
            volume_above_avg = np.sum(recent_volume > avg_volume_20) / len(recent_volume)
            if (volume_above_avg > 0.7 and 
                price_change_5d > 0 and price_change_5d < 0.08 and 
                price_volatility < 0.05):
                signals['accumulation'] = True
                signals['strength'] += 3
                signals['confidence'] += 20

            # Distribution pattern
            if (volume_above_avg > 0.6 and 
                price_change_5d < 0 and price_change_5d > -0.08 and 
                price_volatility > 0.03):
                signals['distribution'] = True
                signals['strength'] += 3
                signals['confidence'] += 20

            # High volume with minimal price impact (absorption)
            avg_volume_impact = abs(price_change_5d) / (np.mean(recent_volume / avg_volume_20))
            if (np.mean(recent_volume / avg_volume_20) > 1.5 and 
                avg_volume_impact < 0.01 and 
                abs(price_change_5d) < 0.03):
                signals['unusual_activity'] = True
                signals['strength'] += 2
                signals['confidence'] += 15

            # Cap confidence at 100
            signals['confidence'] = min(100, signals['confidence'])

            return signals

        except Exception as e:
            print(f"Error in smart money detection: {e}")
            return {
                'accumulation': False,
                'distribution': False,
                'unusual_activity': False,
                'stealth_buying': False,
                'whale_activity': False,
                'strength': 0,
                'confidence': 0
            }

    def calculate_money_flow_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Money Flow Index and buying/selling pressure.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dictionary with money flow analysis
        """
        try:
            if len(df) < 20:
                return {'status': 'insufficient_data'}

            # Calculate typical price
            typical_price = (df['high'] + df['low'] + df['close']) / 3

            # Money Flow Multiplier
            mf_multiplier = []
            for i in range(len(df)):
                if i == 0:
                    mf_multiplier.append(0)
                else:
                    high_low_range = df['high'].iloc[i] - df['low'].iloc[i]
                    if high_low_range == 0:
                        mf_multiplier.append(0)
                    else:
                        multiplier = ((df['close'].iloc[i] - df['low'].iloc[i]) - 
                                    (df['high'].iloc[i] - df['close'].iloc[i])) / high_low_range
                        mf_multiplier.append(multiplier)

            # Money Flow Volume
            mf_volume = [mult * vol for mult, vol in zip(mf_multiplier, df['volume'])]

            # Positive and Negative Money Flow
            positive_mf = [vol if mult > 0 else 0 for mult, vol in zip(mf_multiplier, df['volume'])]
            negative_mf = [vol if mult < 0 else 0 for mult, vol in zip(mf_multiplier, df['volume'])]

            # Money Flow Index (14-period)
            period = min(14, len(df) - 1)
            if period < 5:
                return {'status': 'insufficient_data'}

            positive_sum = sum(positive_mf[-period:])
            negative_sum = sum(negative_mf[-period:])

            if negative_sum == 0:
                mfi = 100
            else:
                money_ratio = positive_sum / negative_sum
                mfi = 100 - (100 / (1 + money_ratio))

            # Buying/Selling Pressure (last 5 periods)
            recent_positive = sum(positive_mf[-5:])
            recent_negative = sum(negative_mf[-5:])
            total_recent = recent_positive + recent_negative

            if total_recent > 0:
                buying_pressure = (recent_positive / total_recent) * 100
                selling_pressure = (recent_negative / total_recent) * 100
            else:
                buying_pressure = selling_pressure = 50

            # Money flow trend
            if buying_pressure > 60:
                money_flow_trend = 'bullish'
            elif selling_pressure > 60:
                money_flow_trend = 'bearish'
            else:
                money_flow_trend = 'neutral'

            return {
                'status': 'success',
                'mfi': round(mfi, 2),
                'buying_pressure': round(buying_pressure, 2),
                'selling_pressure': round(selling_pressure, 2),
                'money_flow_trend': money_flow_trend
            }

        except Exception as e:
            print(f"Error in money flow analysis: {e}")
            return {'status': 'error'}

    def _get_accumulation_signals(self, smart_signals: Dict[str, Any], dow_phase: str) -> List[str]:
        """Extract accumulation signals."""
        signals = []

        if dow_phase == 'Accumulation':
            signals.append('Dow Theory Accumulation Phase')

        if smart_signals.get('accumulation', False):
            signals.append('Smart Money Accumulation Detected')

        if smart_signals.get('stealth_buying', False):
            signals.append('Stealth Buying Pattern')

        if smart_signals.get('whale_activity', False) and smart_signals.get('accumulation', False):
            signals.append('Large Buyer Activity')

        return signals

    def _get_distribution_signals(self, smart_signals: Dict[str, Any], dow_phase: str) -> List[str]:
        """Extract distribution signals."""
        signals = []

        if dow_phase == 'Distribution':
            signals.append('Dow Theory Distribution Phase')

        if smart_signals.get('distribution', False):
            signals.append('Smart Money Distribution Detected')

        if smart_signals.get('whale_activity', False) and smart_signals.get('distribution', False):
            signals.append('Large Seller Activity')

        if smart_signals.get('unusual_activity', False):
            signals.append('Unusual Volume Activity')

        return signals
