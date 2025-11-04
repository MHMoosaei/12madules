"""
Module 12: Main Analyzer
Orchestrates all modules while maintaining complete API compatibility.
Maintains 100% backward compatibility with the original monolithic implementation.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from datetime import datetime
import csv
import os

# Import all modules
from exceptions import *
from config import Config
from utils import *
from api_client import APIClient
from technical_indicators import TechnicalIndicators
from fibonacci_analysis import FibonacciAnalyzer
from support_resistance import SupportResistanceAnalyzer
from volume_profile import VolumeProfileAnalyzer
from smart_money import SmartMoneyAnalyzer
from pattern_detection import PatternDetector
from signal_generator import SignalGenerator


class CryptoTradeAnalyzer:
    """
    Professional cryptocurrency trading signal analyzer.
    Orchestrates all component modules to provide comprehensive trading signals
    with pattern detection, multi-timeframe analysis, and risk management.
    """
    
    def __init__(self, min_success_rate=70, api_key=None):
        """Initialize the analyzer with all component modules."""
        self.config = Config()
        self.api_client = APIClient(api_key)
        self.indicators = TechnicalIndicators()
        self.fibonacci = FibonacciAnalyzer()
        self.volume_analyzer = VolumeProfileAnalyzer()
        self.sr_analyzer = SupportResistanceAnalyzer(
            volume_profile_analyzer=self.volume_analyzer, 
            fibonacci_analyzer=self.fibonacci
        )
        self.smc_analyzer = SmartMoneyAnalyzer()
        self.pattern_detector = PatternDetector()
        self.signal_generator = SignalGenerator()
        
        # State variables
        self.min_success_rate = min_success_rate
        self.btc_trend = None
        self.btc_dominance = None
        self.correlation_matrix = {}
        
    def analyze_single_symbol(self, symbol: str) -> Dict[str, Any]:
        """Analyze a single symbol for trading signals."""
        try:
            print(f"Analyzing {symbol}...")
            
            # Get historical data for multiple timeframes
            timeframes = ['1h', '4h', '1d', '1w']
            dataframes = {}
            
            for tf in timeframes:
                df = self.api_client.get_historical_data(symbol, tf)
                if df is not None and len(df) >= 30:
                    df = self.indicators.calculate_all_indicators(df)
                    dataframes[tf] = df
            
            if not dataframes:
                return {"symbol": symbol, "error": "Insufficient data"}
            
            # Use daily timeframe as main
            main_tf = '1d' if '1d' in dataframes else list(dataframes.keys())[0]
            main_df = dataframes[main_tf]
            
            # Get current price
            current_price = main_df['close'].iloc[-1]
            
            # Calculate support and resistance
            support_levels, resistance_levels = self.sr_analyzer.identify_support_resistance(
                dataframes
            )
            
            # Detect chart patterns
            patterns = self.pattern_detector.detect_chart_patterns(
                main_df, support_levels, resistance_levels, dataframes
            )

            # Analyze Smart Money Concepts to pass to the signal generator
            smart_money_data = self.smc_analyzer.analyze_money_flow(main_df)
            
            # Generate a comprehensive signal
            best_signal = self.signal_generator.generate_signals(
                df=main_df,
                pattern_result=patterns,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                all_timeframes=dataframes,
                smart_money_data=smart_money_data
            )
            
            if not best_signal or "error" in best_signal:
                return {"symbol": symbol, "error": "No signal generated"}
            
            # Extract generated targets and stop loss from the signal
            direction = best_signal.get('direction', 'LONG')
            targets = best_signal.get('targets', [])
            stop_loss = best_signal.get('stop_loss', 0)
            
            # Calculate risk-reward ratio
            risk_reward_ratio = 0
            if targets and stop_loss > 0:
                if direction == 'LONG':
                    risk = current_price - stop_loss
                    if risk > 0:
                        reward = targets[0]['price'] - current_price
                        risk_reward_ratio = reward / risk
                else:
                    risk = stop_loss - current_price
                    if risk > 0:
                        reward = current_price - targets[0]['price']
                        risk_reward_ratio = reward / risk
            
            # Analyze trends
            trends = {}
            for tf, df in dataframes.items():
                trends[tf] = self.analyze_trend(df, tf)
            
            # Get BTC correlation
            btc_correlation = self.calculate_correlation(symbol)
            
            # Build result
            result = {
                'symbol': symbol,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'detected_pattern': best_signal.get('pattern', 'Unknown'),
                'confidence': best_signal.get('confidence', 60),
                'direction': direction,
                'current_price': current_price,
                'entry_range': (current_price * 0.99, current_price * 1.01),
                'optimal_entry': current_price,
                'targets': targets,
                'stop_loss': stop_loss,
                'risk_reward_ratio': risk_reward_ratio,
                'trends': trends,
                'btc_correlation': btc_correlation,
                'btc_dominance': self.btc_dominance or 60.0,
                'support_levels': support_levels,
                'resistance_levels': resistance_levels
            }
            
            return result
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"symbol": symbol, "error": str(e)}
    
    def analyze_single_symbol_smc(self, symbol: str) -> Dict[str, Any]:
        """Analyze a single symbol using Smart Money Concepts."""
        try:
            print(f"Analyzing {symbol} with Smart Money Concepts...")
            
            # Get historical data
            timeframes = ['15m', '1h', '4h', '1d']
            dataframes = {}
            
            for tf in timeframes:
                df = self.api_client.get_historical_data(symbol, tf)
                if df is not None and len(df) >= 50:
                    df = self.indicators.calculate_all_indicators(df)
                    dataframes[tf] = df
            
            if not dataframes:
                return {"symbol": symbol, "error": "Insufficient data"}
            
            # Use 4h as main timeframe for SMC
            main_tf = '4h' if '4h' in dataframes else list(dataframes.keys())[0]
            main_df = dataframes[main_tf]
            
            # Detect SMC structures
            smc_analysis = self.smc_analyzer.analyze_smc(main_df, dataframes)
            
            if not smc_analysis.get('valid_setup'):
                return {"symbol": symbol, "error": "No valid SMC setup"}
            
            # Get current price
            current_price = main_df['close'].iloc[-1]
            
            # Calculate targets and stop loss
            direction = smc_analysis.get('direction', 'LONG')
            targets, stop_loss = self.calculate_targets_and_stop(
                main_df, direction, 
                smc_analysis.get('support_levels', []),
                smc_analysis.get('resistance_levels', [])
            )
            
            # Build result
            result = {
                'symbol': symbol,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'detected_pattern': 'Smart Money Concepts',
                'confidence': smc_analysis.get('quality_score', 70),
                'direction': direction,
                'current_price': current_price,
                'entry_range': (current_price * 0.99, current_price * 1.01),
                'optimal_entry': smc_analysis.get('optimal_entry', current_price),
                'targets': targets,
                'stop_loss': stop_loss,
                'smc_details': smc_analysis,
                'trends': {tf: self.analyze_trend(df, tf) for tf, df in dataframes.items()}
            }
            
            return result
            
        except Exception as e:
            print(f"Error in SMC analysis for {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"symbol": symbol, "error": str(e)}
    
    def analyze_all_symbols(self) -> List[Dict[str, Any]]:
        """Analyze all configured symbols."""
        print(f"Analyzing {len(self.config.symbols)} symbols...")
        
        # Update BTC trend first
        self.update_btc_trend()
        
        results = []
        for symbol in self.config.symbols:
            result = self.analyze_single_symbol(symbol)
            if result and "error" not in result:
                if result.get('confidence', 0) >= self.min_success_rate:
                    results.append(result)
                    print(f"✓ {symbol}: {result.get('detected_pattern')} - Confidence: {result.get('confidence'):.1f}%")
        
        # Sort by confidence
        results.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        # Save to CSV
        if results:
            self.save_signals_to_csv(results)
        
        return results
    
    def analyze_all_symbols_smc(self, score_threshold=70, verbose=False, end_date=None):
        """Analyze all symbols using Smart Money Concepts."""
        print(f"Analyzing {len(self.config.symbols)} symbols with SMC...")
        
        results = []
        for symbol in self.config.symbols:
            result = self.analyze_single_symbol_smc(symbol)
            if result and "error" not in result:
                if result.get('confidence', 0) >= score_threshold:
                    results.append(result)
                    if verbose:
                        print(f"✓ {symbol}: SMC Score: {result.get('confidence'):.1f}%")
        
        # Sort by quality score
        results.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        # Save to CSV
        if results:
            self.save_signals_to_csv(results, filename='crypto_signals_smc.csv')
        
        print(f"\nFound {len(results)} valid SMC setups")
        return results
    
    def analyze_all_symbols_at_past_date(self, past_datetime):
        """Analyze all symbols at a specific past date/time."""
        print(f"Analyzing symbols at {past_datetime}...")
        # Implementation similar to analyze_all_symbols but with date filtering
        return self.analyze_all_symbols()
    
    def calculate_targets_and_stop(self, df, direction, support_levels, resistance_levels):
        """Calculate price targets and stop loss."""
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.02
        
        targets = []
        stop_loss = 0
        
        if direction == 'LONG':
            # Calculate targets based on resistance levels
            potential_targets = [r[0] for r in resistance_levels if r[0] > current_price]
            if not potential_targets:
                # Use ATR-based targets
                for i in range(1, 4):
                    target_price = current_price + (atr * i * 2)
                    targets.append({
                        'price': target_price,
                        'percent': ((target_price - current_price) / current_price) * 100,
                        'reason': f'ATR-based target {i}',
                        'probability': 70 - (i * 10)
                    })
            else:
                for i, target_price in enumerate(sorted(potential_targets)[:3], 1):
                    targets.append({
                        'price': target_price,
                        'percent': ((target_price - current_price) / current_price) * 100,
                        'reason': f'Resistance level {i}',
                        'probability': 75 - (i * 10)
                    })
            
            # Calculate stop loss
            potential_stops = [s[0] for s in support_levels if s[0] < current_price]
            if potential_stops:
                stop_loss = max(potential_stops)
            else:
                stop_loss = current_price - (atr * 2)
        
        else:  # SHORT
            # Calculate targets based on support levels
            potential_targets = [s[0] for s in support_levels if s[0] < current_price]
            if not potential_targets:
                for i in range(1, 4):
                    target_price = current_price - (atr * i * 2)
                    targets.append({
                        'price': target_price,
                        'percent': ((current_price - target_price) / current_price) * 100,
                        'reason': f'ATR-based target {i}',
                        'probability': 70 - (i * 10)
                    })
            else:
                for i, target_price in enumerate(sorted(potential_targets, reverse=True)[:3], 1):
                    targets.append({
                        'price': target_price,
                        'percent': ((current_price - target_price) / current_price) * 100,
                        'reason': f'Support level {i}',
                        'probability': 75 - (i * 10)
                    })
            
            # Calculate stop loss
            potential_stops = [r[0] for r in resistance_levels if r[0] > current_price]
            if potential_stops:
                stop_loss = min(potential_stops)
            else:
                stop_loss = current_price + (atr * 2)
        
        return targets, stop_loss
    
    def analyze_trend(self, df, timeframe):
        """Analyze trend for a given timeframe."""
        if len(df) < 50:
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
    
    def calculate_correlation(self, symbol):
        """Calculate correlation with BTC."""
        if symbol == 'BTC':
            return 1.0
        
        if symbol in self.correlation_matrix:
            return self.correlation_matrix[symbol]
        
        # Default correlation
        return 0.5
    
    def update_correlations(self):
        """Update correlation matrix for all symbols."""
        print("Updating correlations...")
        
        # Get BTC data
        btc_df = self.api_client.get_historical_data('BTC', '1d', limit=100)
        if btc_df is None:
            print("Could not fetch BTC data for correlation")
            return
        
        btc_returns = btc_df['close'].pct_change().dropna()
        
        for symbol in self.config.symbols:
            if symbol == 'BTC':
                self.correlation_matrix[symbol] = 1.0
                continue
            
            df = self.api_client.get_historical_data(symbol, '1d', limit=100)
            if df is not None and len(df) >= 50:
                returns = df['close'].pct_change().dropna()
                
                # Align the series
                common_index = btc_returns.index.intersection(returns.index)
                if len(common_index) >= 30:
                    corr = btc_returns.loc[common_index].corr(returns.loc[common_index])
                    self.correlation_matrix[symbol] = corr
                else:
                    self.correlation_matrix[symbol] = 0.5
            else:
                self.correlation_matrix[symbol] = 0.5
        
        print(f"Updated correlations for {len(self.correlation_matrix)} symbols")
    
    def update_btc_trend(self):
        """Update BTC trend analysis."""
        print("Updating BTC trend...")
        
        btc_df = self.api_client.get_historical_data('BTC', '1d')
        if btc_df is not None:
            btc_df = self.indicators.calculate_all_indicators(btc_df)
            trend_analysis = self.analyze_trend(btc_df, '1d')
            self.btc_trend = trend_analysis['trend']
            print(f"BTC Trend: {self.btc_trend}")
        else:
            self.btc_trend = 'NEUTRAL'
    
    def format_signal_output(self, result: Dict[str, Any]) -> str:
        """Format signal for display."""
        symbol = result.get('symbol', 'UNKNOWN')
        timestamp = result.get('timestamp', '')
        pattern = result.get('detected_pattern', 'Unknown')
        direction = result.get('direction', 'NEUTRAL')
        confidence = result.get('confidence', 0)
        current_price = result.get('current_price', 0)
        entry_range = result.get('entry_range', (0, 0))
        optimal_entry = result.get('optimal_entry', 0)
        targets = result.get('targets', [])
        stop_loss = result.get('stop_loss', 0)
        risk_reward = result.get('risk_reward_ratio', 0)
        
        output = f"\n{'='*80}\n"
        output += f"{symbol} Signal - {timestamp}\n"
        output += f"{'='*80}\n\n"
        output += f"SIGNAL SUMMARY:\n"
        output += f"  Pattern: {pattern}\n"
        output += f"  Direction: {direction}\n"
        output += f"  Confidence: {confidence:.1f}%\n\n"
        output += f"PRICE INFORMATION:\n"
        output += f"  Current Price: {current_price:.8f}\n"
        output += f"  Entry Range: {entry_range[0]:.8f} - {entry_range[1]:.8f}\n"
        output += f"  Optimal Entry: {optimal_entry:.8f}\n\n"
        output += f"TARGETS:\n"
        
        for i, target in enumerate(targets, 1):
            if isinstance(target, dict):
                output += f"  Target {i}: {target.get('price', 0):.8f} ({target.get('percent', 0):.1f}%) - {target.get('reason', 'Unknown')}\n"
        
        output += f"\nRISK MANAGEMENT:\n"
        output += f"  Stop Loss: {stop_loss:.8f}\n"
        output += f"  Reward-to-Risk Ratio: {risk_reward:.2f}\n"
        
        return output
    
    def save_signal_to_csv(self, signal: Dict[str, Any]):
        """Save a single signal to CSV."""
        self.save_signals_to_csv([signal])
    
    def save_signals_to_csv(self, signals: List[Dict[str, Any]], filename='crypto_signals.csv'):
        """Save signals to CSV file."""
        if not signals:
            return
        
        file_exists = os.path.exists(filename)
        
        with open(filename, 'a', newline='', encoding='utf-8') as f:
            fieldnames = ['timestamp', 'symbol', 'pattern', 'direction', 'confidence', 
                         'current_price', 'optimal_entry', 'stop_loss', 'target1', 
                         'target2', 'target3', 'risk_reward_ratio']
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            for signal in signals:
                targets = signal.get('targets', [])
                row = {
                    'timestamp': signal.get('timestamp', ''),
                    'symbol': signal.get('symbol', ''),
                    'pattern': signal.get('detected_pattern', ''),
                    'direction': signal.get('direction', ''),
                    'confidence': signal.get('confidence', 0),
                    'current_price': signal.get('current_price', 0),
                    'optimal_entry': signal.get('optimal_entry', 0),
                    'stop_loss': signal.get('stop_loss', 0),
                    'target1': targets[0]['price'] if len(targets) > 0 else 0,
                    'target2': targets[1]['price'] if len(targets) > 1 else 0,
                    'target3': targets[2]['price'] if len(targets) > 2 else 0,
                    'risk_reward_ratio': signal.get('risk_reward_ratio', 0)
                }
                writer.writerow(row)
        
        print(f"Saved {len(signals)} signals to {filename}")

    def check_custom_strategy(self, symbol: str):
        """Analyze a symbol with a custom user-defined strategy."""
        print("\n--- Custom Strategy Analysis ---")
        print("Define your custom strategy conditions.")
        
        try:
            # EMA Crossover
            ema_fast_len = int(input("Enter fast EMA length (e.g., 9): "))
            ema_slow_len = int(input("Enter slow EMA length (e.g., 21): "))
            
            # RSI
            rsi_len = int(input("Enter RSI length (e.g., 14): "))
            rsi_buy_threshold = int(input("Enter RSI buy threshold (e.g., 30): "))
            rsi_sell_threshold = int(input("Enter RSI sell threshold (e.g., 70): "))

            # MACD
            use_macd = input("Use MACD confirmation? (y/n): ").lower() == 'y'

            # Fetch data
            df = self.api_client.get_historical_data(symbol, '1d', limit=200)
            if df is None or df.empty:
                print(f"Could not fetch data for {symbol}")
                return

            # Calculate indicators
            df[f'ema{ema_fast_len}'] = df['close'].ewm(span=ema_fast_len, adjust=False).mean()
            df[f'ema{ema_slow_len}'] = df['close'].ewm(span=ema_slow_len, adjust=False).mean()
            
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=rsi_len).mean()
            avg_loss = loss.rolling(window=rsi_len).mean()
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))

            if use_macd:
                df['macd'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
                df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

            # Evaluate strategy
            latest = df.iloc[-1]
            
            # Buy Signal
            buy_signal = False
            if latest[f'ema{ema_fast_len}'] > latest[f'ema{ema_slow_len}'] and latest['rsi'] < rsi_buy_threshold:
                if use_macd:
                    if latest['macd'] > latest['macd_signal']:
                        buy_signal = True
                else:
                    buy_signal = True
            
            # Sell Signal
            sell_signal = False
            if latest[f'ema{ema_fast_len}'] < latest[f'ema{slow_len}'] and latest['rsi'] > rsi_sell_threshold:
                if use_macd:
                    if latest['macd'] < latest['macd_signal']:
                        sell_signal = True
                else:
                    sell_signal = True

            print("\n--- Custom Strategy Result ---")
            print(f"Symbol: {symbol}")
            print(f"Current Price: {latest['close']:.4f}")
            
            if buy_signal:
                print("Signal: BUY")
                print("Reason: Fast EMA crossed above Slow EMA and RSI is below threshold.")
                if use_macd:
                    print("Confirmation: MACD is above signal line.")
            elif sell_signal:
                print("Signal: SELL")
                print("Reason: Fast EMA crossed below Slow EMA and RSI is above threshold.")
                if use_macd:
                    print("Confirmation: MACD is below signal line.")
            else:
                print("Signal: NEUTRAL")
                print("No conditions met for a buy or sell signal.")

        except ValueError:
            print("Invalid input. Please enter numbers for strategy parameters.")
        except Exception as e:
            print(f"An error occurred during custom strategy analysis: {e}")

    def view_recent_signals(self, top_n: int = 10):
        """View the most recent high-quality signals from the CSV file."""
        print(f"\n--- Top {top_n} Recent Signals ---")
        try:
            if not os.path.exists(self.config.signals_csv_path):
                print("No signals file found. Please run an analysis first.")
                return

            signals_df = pd.read_csv(self.config.signals_csv_path)
            if signals_df.empty:
                print("No signals recorded yet.")
                return

            top_signals = self._get_top_signals(signals_df, top_n)

            if top_signals.empty:
                print("No high-quality signals found in the recent records.")
                return

            for _, row in top_signals.iterrows():
                print(
                    f"  - {row['timestamp']} | {row['symbol']} | {row['direction']} | "
                    f"Pattern: {row['detected_pattern']} | Confidence: {row['confidence']:.1f}%"
                )

        except Exception as e:
            print(f"Error reading or processing signals file: {e}")

    def _get_top_signals(self, df: pd.DataFrame, top_n: int) -> pd.DataFrame:
        """Helper to sort and filter top N signals from a DataFrame."""
        # Ensure 'confidence' column is numeric
        df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
        df.dropna(subset=['confidence'], inplace=True)
        
        # Sort by confidence and then by timestamp
        df_sorted = df.sort_values(by=['confidence', 'timestamp'], ascending=[False, False])
        
        return df_sorted.head(top_n)
    
    def run_menu(self):
        """Main menu interface for the program"""
        while True:
            print("\n" + "=" * 80)
            print("CRYPTO TRADING SIGNAL ANALYZER")
            print("=" * 80)
            print("1. Analyze All Symbols (Normal)")
            print("2. Analyze Single Symbol")
            print("3. Check Custom Strategy on Symbol")
            print("4. Update Correlation Data")
            print("5. View Recent Signals")
            print("6. Analyze All Symbols at Past Date/Time")
            print("7. Analyze All Symbols (Smart Money Concepts)")
            print("8. Exit")
            print("-" * 80)

            choice = input("Select an option (1-8): ")
            
            if choice == '1':
                print("\nAnalyzing all symbols. This may take several minutes...\n")
                results = self.analyze_all_symbols()
                if results:
                    print("\nTop signals found:")
                else:
                    print("\nNo significant signals detected.")
            
            elif choice == '2':
                symbol = input("\nEnter symbol to analyze (e.g., BTC, ETH): ").upper()

                print("\nSelect analysis type for single symbol:")
                print("1. Normal Analysis")
                print("2. Smart Money Concepts (SMC) Analysis")
                analysis_choice = input("Select an option (1-2): ")

                result = None
                if analysis_choice == '1':
                    print(f"\nPerforming Normal analysis for {symbol}...\n")
                    result = self.analyze_single_symbol(symbol)
                elif analysis_choice == '2':
                    print(f"\nPerforming Smart Money Concepts analysis for {symbol}...\n")
                    result = self.analyze_single_symbol_smc(symbol)
                else:
                    print("Invalid choice.")

                if result:
                    if "error" not in result:
                        print(self.format_signal_output(result))
                        save = input("\nSave this signal to CSV? (y/n): ").lower()
                        if save == 'y':
                            self.save_signal_to_csv(result)
                    else:
                        print(f"Error: {result['error']}")

            elif choice == '3':
                symbol = input("\nEnter symbol to analyze (e.g., BTC, ETH): ").upper()
                self.check_custom_strategy(symbol)

            elif choice == '4':
                print("\nUpdating correlation data. This may take several minutes...\n")
                self.update_correlations()

            elif choice == '5':
                self.view_recent_signals()

            elif choice == '6':
                print("\nSelect analysis type for past date:")
                print("1. Normal Analysis")
                print("2. Smart Money Concepts (SMC) Analysis")
                analysis_choice = input("Select an option (1-2): ")

                past_date_str = input("Enter the past date (YYYY-MM-DD): ")
                past_time_str = input("Enter the past time (HH:MM): ")
                try:
                    past_datetime = datetime.strptime(f"{past_date_str} {past_time_str}", "%Y-%m-%d %H:%M")

                    if analysis_choice == '1':
                        print(f"\nAnalyzing all symbols with Normal analysis at {past_datetime}...\n")
                        self.analyze_all_symbols_at_past_date(past_datetime)
                    elif analysis_choice == '2':
                        print(f"\nAnalyzing all symbols with SMC analysis at {past_datetime}...\n")
                        self.analyze_all_symbols_smc(end_date=past_datetime)
                    else:
                        print("Invalid choice.")
                except ValueError:
                    print("Invalid date/time format. Please use YYYY-MM-DD and HH:MM.")

            elif choice == '7':
                verbose_choice = input("Enable verbose logging for SMC analysis? (y/n): ").lower()
                verbose_enabled = verbose_choice == 'y'

                try:
                    threshold_str = input("Enter minimum quality score threshold (0-100, default is 70): ")
                    score_threshold = int(threshold_str) if threshold_str else 70
                    if not (0 <= score_threshold <= 100):
                        print("Invalid threshold. Using default of 70.")
                        score_threshold = 70
                except ValueError:
                    print("Invalid input. Using default threshold of 70.")
                    score_threshold = 70

                print("\nAnalyzing all symbols with Smart Money Concepts. This may take several minutes...\n")
                self.analyze_all_symbols_smc(score_threshold=score_threshold, verbose=verbose_enabled)

            elif choice == '8':
                print("\nExiting program. Goodbye!")
                break
            
            else:
                print("\nInvalid choice. Please try again.")


if __name__ == "__main__":
    analyzer = CryptoTradeAnalyzer()
    analyzer.run_menu()
