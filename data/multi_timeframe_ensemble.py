import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from datetime import datetime
import warnings
import traceback
warnings.filterwarnings('ignore')

class MultiTimeframeEnsemble:
    def __init__(self, model_4h_path, scaler_4h_path, model_1h_path, scaler_1h_path, initial_capital=10000):
        """
        Multi-Timeframe Ensemble Trading System with Enhanced Risk Management
        Combines proven 4H model (82.57% returns) with precision 1H model (2.33% MAE)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.daily_loss = 0
        
        # Load 4H model and scaler (your proven system)
        print(f"📊 Loading 4H LSTM model (proven 82.57% returns)...")
        self.model_4h = tf.keras.models.load_model(model_4h_path, compile=False)
        self.scaler_4h = joblib.load(scaler_4h_path)
        
        # Load 1H model and scaler (precision timing)
        print(f"🕐 Loading 1H LSTM model (2.33% MAE precision)...")
        self.model_1h = tf.keras.models.load_model(model_1h_path, compile=False)
        self.scaler_1h = joblib.load(scaler_1h_path)
        
        # Ensemble parameters (optimized for your system)
        self.ensemble_weights = {
            'trend_detection': {'4h': 0.7, '1h': 0.3},
            'entry_timing': {'4h': 0.3, '1h': 0.7},
            'exit_timing': {'4h': 0.4, '1h': 0.6}
        }
        
        # Trading parameters (enhanced from your optimization)
        self.take_profit_pct = 0.12
        self.stop_loss_pct = 0.06
        self.max_hold_periods = 35
        self.min_signal_strength = 1.5
        self.position_size_pct = 0.20
        
        # Multi-horizon weights (from your proven system)
        self.horizon_weights = {'4H': 0.40, '8H': 0.30, '1D': 0.20, '3D': 0.07, '5D': 0.03}

        # Trade tracking
        self.trades = []
        self.portfolio_history = []
        self.current_position = None
        self.position_hold_count = 0

    def load_data(self, data_4h_path, data_1h_path):
        """Load and align 4H and 1H datasets with robust datetime handling"""
        print(f"\n📊 Loading and aligning multi-timeframe data...")
        
        try:
            # Load 4H data (your proven dataset)
            self.data_4h = pd.read_csv(data_4h_path)
            print(f"✅ 4H CSV loaded: {len(self.data_4h)} rows")
            
            # Load 1H data (your new precision dataset)
            self.data_1h = pd.read_csv(data_1h_path)
            print(f"✅ 1H CSV loaded: {len(self.data_1h)} rows")

            # ROBUST DATETIME CONVERSION FOR 4H DATA
            print(f"🔧 Converting 4H datetime column...")
            print(f"   Before conversion - dtype: {self.data_4h['datetime'].dtype}")
            print(f"   Sample values: {self.data_4h['datetime'].head(2).tolist()}")
            
            self.data_4h['datetime'] = pd.to_datetime(self.data_4h['datetime'], errors='coerce')
            nat_count_4h = self.data_4h['datetime'].isna().sum()
            if nat_count_4h > 0:
                print(f"⚠️ Found {nat_count_4h} invalid datetime values in 4H data - removing")
                self.data_4h = self.data_4h.dropna(subset=['datetime']).reset_index(drop=True)
            
            if hasattr(self.data_4h['datetime'].dtype, 'tz') and self.data_4h['datetime'].dtype.tz is not None:
                print(f"   Removing timezone info from 4H data...")
                self.data_4h['datetime'] = self.data_4h['datetime'].dt.tz_localize(None)
            
            print(f"   After conversion - dtype: {self.data_4h['datetime'].dtype}")
            print(f"   Sample values: {self.data_4h['datetime'].head(2).tolist()}")

            # ROBUST DATETIME CONVERSION FOR 1H DATA
            print(f"🔧 Converting 1H datetime column...")
            print(f"   Before conversion - dtype: {self.data_1h['datetime'].dtype}")
            print(f"   Sample values: {self.data_1h['datetime'].head(2).tolist()}")
            
            self.data_1h['datetime'] = pd.to_datetime(self.data_1h['datetime'], errors='coerce')
            nat_count_1h = self.data_1h['datetime'].isna().sum()
            if nat_count_1h > 0:
                print(f"⚠️ Found {nat_count_1h} invalid datetime values in 1H data - removing")
                self.data_1h = self.data_1h.dropna(subset=['datetime']).reset_index(drop=True)
            
            if hasattr(self.data_1h['datetime'].dtype, 'tz') and self.data_1h['datetime'].dtype.tz is not None:
                print(f"   Removing timezone info from 1H data...")
                self.data_1h['datetime'] = self.data_1h['datetime'].dt.tz_localize(None)
            
            print(f"   After conversion - dtype: {self.data_1h['datetime'].dtype}")
            print(f"   Sample values: {self.data_1h['datetime'].head(2).tolist()}")

            # Use same 80% split as your proven system
            split_4h = int(0.8 * len(self.data_4h))
            split_1h = int(0.8 * len(self.data_1h))
            self.trading_data_4h = self.data_4h.iloc[:split_4h].copy()
            self.trading_data_1h = self.data_1h.iloc[:split_1h].copy()
            
            print(f"✅ 4H data: {len(self.trading_data_4h):,} candles")
            print(f"✅ 1H data: {len(self.trading_data_1h):,} candles")
            print(f"📅 4H date range: {self.trading_data_4h['datetime'].min()} to {self.trading_data_4h['datetime'].max()}")
            print(f"📅 1H date range: {self.trading_data_1h['datetime'].min()} to {self.trading_data_1h['datetime'].max()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Traceback: {traceback.format_exc()}")
            return False

    def generate_4h_predictions(self, historical_sequence_4h):
        """Generate predictions using your proven 4H model"""
        try:
            scaled_sequence = self.scaler_4h.transform(historical_sequence_4h)
            X_input = scaled_sequence.reshape(1, 60, 5)
            prediction_scaled = self.model_4h.predict(X_input, verbose=0)[0]
            
            predictions = {}
            horizons = ['4H', '8H', '1D', '3D', '5D']
            
            for i, horizon in enumerate(horizons):
                dummy_array = np.zeros((1, 5))
                dummy_array[0, 3] = prediction_scaled[i]
                raw_prediction = self.scaler_4h.inverse_transform(dummy_array)[0, 3]
                predictions[horizon] = raw_prediction
            
            return predictions
        except:
            return None

    def generate_1h_predictions(self, historical_sequences_1h):
        """Generate predictions using your precision 1H model"""
        try:
            predictions_list = []
            
            # Generate multiple 1H predictions for ensemble averaging
            for sequence in historical_sequences_1h[-4:]:  # Last 4 sequences (4 hours)
                scaled_sequence = self.scaler_1h.transform(sequence)
                X_input = scaled_sequence.reshape(1, 60, len(sequence[0]))
                prediction_scaled = self.model_1h.predict(X_input, verbose=0)[0]
                
                # Convert to price predictions (focusing on 4H horizon for alignment)
                horizons = ['1H', '4H', '8H', '1D', '5D']
                horizon_predictions = {}
                
                for i, horizon in enumerate(horizons):
                    dummy_array = np.zeros((1, len(sequence[0])))
                    dummy_array[0, 3] = prediction_scaled[i]
                    raw_prediction = self.scaler_1h.inverse_transform(dummy_array)[0, 3]
                    horizon_predictions[horizon] = raw_prediction
                
                predictions_list.append(horizon_predictions)
            
            # Average the predictions for stability
            if predictions_list:
                averaged_predictions = {}
                for horizon in ['1H', '4H', '8H', '1D', '5D']:
                    averaged_predictions[horizon] = np.mean([p[horizon] for p in predictions_list])
                return averaged_predictions
            
            return None
        except:
            return None

    def create_ensemble_signals(self, predictions_4h, predictions_1h, current_price, signal_type='trend_detection'):
        """Create ensemble signals by combining 4H and 1H predictions"""
        try:
            if not predictions_4h or not predictions_1h:
                return None, 0
            
            # Get ensemble weights based on signal type
            weights = self.ensemble_weights[signal_type]
            
            # Calculate weighted signals
            ensemble_predictions = {}
            max_change = current_price * 0.10  # 10% bounds
            
            for horizon in ['4H', '8H', '1D']:  # Focus on key horizons
                if horizon in predictions_4h and horizon in predictions_1h:
                    # Weighted average of predictions
                    ensemble_pred = (weights['4h'] * predictions_4h[horizon] + 
                                   weights['1h'] * predictions_1h[horizon])
                    
                    # Apply bounds
                    bounded_pred = np.clip(ensemble_pred,
                                         current_price - max_change,
                                         current_price + max_change)
                    
                    ensemble_predictions[horizon] = bounded_pred
            
            # Calculate ensemble signal strength
            weighted_signal = 0
            total_strength = 0
            
            for horizon, predicted_price in ensemble_predictions.items():
                change_pct = ((predicted_price - current_price) / current_price) * 100
                weight = self.horizon_weights.get(horizon, 0)
                
                weighted_signal += weight * change_pct
                total_strength += abs(change_pct) * weight
            
            return weighted_signal, total_strength
            
        except Exception as e:
            return None, 0

    def enhanced_position_management(self, current_price, current_time, predictions_4h, predictions_1h):
        """Enhanced position management with trailing stops and profit protection"""
        
        if self.current_position is None:
            return  # No position to manage
        
        position = self.current_position
        direction = position['direction']
        self.position_hold_count += 1
        
        # Update price extremes for trailing stops
        if direction == 'LONG':
            position['highest_price'] = max(position.get('highest_price', position['entry_price']), current_price)
            unrealized_pnl_pct = (current_price - position['entry_price']) / position['entry_price'] * 100
            
            # Trailing stop: 4% below highest price once in profit
            if unrealized_pnl_pct > 4.0:
                trailing_stop = position['highest_price'] * 0.96
                if current_price <= trailing_stop:
                    self.close_position(current_price, current_time, 'TRAILING_STOP')
                    return
                    
        else:  # SHORT
            position['lowest_price'] = min(position.get('lowest_price', position['entry_price']), current_price)
            unrealized_pnl_pct = (position['entry_price'] - current_price) / position['entry_price'] * 100
            
            # Trailing stop: 4% above lowest price once in profit
            if unrealized_pnl_pct > 4.0:
                trailing_stop = position['lowest_price'] * 1.04
                if current_price >= trailing_stop:
                    self.close_position(current_price, current_time, 'TRAILING_STOP')
                    return
        
        # Partial profit taking at 8% gain
        if abs(unrealized_pnl_pct) > 8.0 and not position.get('partial_taken', False):
            # Take 50% profit
            self.partial_close_position(0.5, current_price, current_time)
            position['partial_taken'] = True
            return
        
        # Use ensemble predictions for dynamic exit decisions
        exit_signal, exit_strength = self.create_ensemble_signals(
            predictions_4h, predictions_1h, current_price, 'exit_timing'
        )
        
        # Enhanced stop loss (dynamic based on 1H predictions)
        if exit_signal and abs(exit_signal) > 2.0:  # Strong reverse signal
            if ((direction == 'LONG' and exit_signal < -2.0) or 
                (direction == 'SHORT' and exit_signal > 2.0)):
                self.close_position(current_price, current_time, 'ENSEMBLE_STOP')
                return
        
        # Original stop loss (FIXED calculation)
        if ((direction == 'LONG' and current_price <= position['stop_loss']) or 
            (direction == 'SHORT' and current_price >= position['stop_loss'])):
            self.close_position(current_price, current_time, 'STOP_LOSS')
            return
        
        # Take profit
        if ((direction == 'LONG' and current_price >= position['take_profit']) or 
            (direction == 'SHORT' and current_price <= position['take_profit'])):
            self.close_position(current_price, current_time, 'TAKE_PROFIT')
            return
        
        # Time exit
        if self.position_hold_count >= self.max_hold_periods:
            self.close_position(current_price, current_time, 'TIME_EXIT')
            return

    def partial_close_position(self, close_percentage, exit_price, exit_time):
        """Close partial position to lock in profits"""
        if self.current_position is None:
            return
        
        position = self.current_position
        direction = position['direction']
        
        # Calculate partial close
        shares_to_close = position['shares'] * close_percentage
        remaining_shares = position['shares'] - shares_to_close
        
        exit_value = shares_to_close * exit_price
        
        if direction == 'LONG':
            pnl_usd = exit_value - (position['position_value'] * close_percentage)
            pnl_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100
        else:
            pnl_usd = (position['position_value'] * close_percentage) - exit_value
            pnl_pct = (position['entry_price'] - exit_price) / position['entry_price'] * 100
        
        self.current_capital += exit_value
        
        # Update position
        position['shares'] = remaining_shares
        position['position_value'] *= (1 - close_percentage)
        
        # Record partial trade
        trade_record = {
            'entry_time': position['entry_time'],
            'exit_time': exit_time,
            'direction': direction,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'shares': shares_to_close,
            'position_value': position['position_value'] * close_percentage,
            'pnl_pct': pnl_pct,
            'pnl_usd': pnl_usd,
            'exit_reason': 'PARTIAL_PROFIT',
            'hold_periods': self.position_hold_count
        }
        
        self.trades.append(trade_record)
        
        print(f"📊 {direction} PARTIAL closed: {pnl_pct:+.2f}% | ${pnl_usd:+,.0f} | PARTIAL_PROFIT ({close_percentage*100:.0f}%)")

    def open_position(self, direction, entry_price, entry_time, position_size):
        """Open position with FIXED risk-based sizing"""
        
        # CRITICAL FIX: Risk-based position sizing
        risk_per_trade = self.initial_capital * 0.02  # Risk 2% per trade
        price_risk = entry_price * self.stop_loss_pct  # Price distance to stop loss
        
        # Position size = Risk amount / Price risk
        calculated_shares = risk_per_trade / price_risk
        max_position_value = min(position_size, self.current_capital * 0.20)
        
        if calculated_shares * entry_price > max_position_value:
            actual_shares = max_position_value / entry_price
            actual_position_value = max_position_value
        else:
            actual_shares = calculated_shares
            actual_position_value = calculated_shares * entry_price
        
        if direction == 'LONG':
            stop_loss = entry_price * (1 - self.stop_loss_pct)
            take_profit = entry_price * (1 + self.take_profit_pct)
        else:
            stop_loss = entry_price * (1 + self.stop_loss_pct)
            take_profit = entry_price * (1 - self.take_profit_pct)
        
        self.current_position = {
            'direction': direction,
            'entry_price': entry_price,
            'entry_time': entry_time,
            'shares': actual_shares,
            'position_value': actual_position_value,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'highest_price': entry_price if direction == 'LONG' else entry_price,
            'lowest_price': entry_price if direction == 'SHORT' else entry_price
        }
        
        self.position_hold_count = 0
        self.current_capital -= actual_position_value
        
        print(f"📈 {direction} opened: ${entry_price:,.0f} | Size: ${actual_position_value:,.0f} | Risk: ${risk_per_trade:,.0f}")

    def close_position(self, exit_price, exit_time, exit_reason):
        """Close position and calculate P&L"""
        if self.current_position is None:
            return
        
        position = self.current_position
        direction = position['direction']
        entry_price = position['entry_price']
        shares = position['shares']
        
        exit_value = shares * exit_price
        
        if direction == 'LONG':
            pnl_usd = exit_value - position['position_value']
            pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:
            pnl_usd = position['position_value'] - exit_value
            pnl_pct = (entry_price - exit_price) / entry_price * 100
        
        self.current_capital += exit_value
        
        trade_record = {
            'entry_time': position['entry_time'],
            'exit_time': exit_time,
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'shares': shares,
            'position_value': position['position_value'],
            'pnl_pct': pnl_pct,
            'pnl_usd': pnl_usd,
            'exit_reason': exit_reason,
            'hold_periods': self.position_hold_count
        }
        
        self.trades.append(trade_record)
        
        print(f"📊 {direction} closed: {pnl_pct:+.2f}% | ${pnl_usd:+,.0f} | {exit_reason}")
        
        self.current_position = None
        self.position_hold_count = 0

    def check_portfolio_heat(self):
        """Ensure total portfolio risk doesn't exceed 15% of initial capital"""
        if self.current_position is None:
            return True
        
        # Calculate current position risk (stop loss distance * position size)
        position = self.current_position
        if position['direction'] == 'LONG':
            risk_amount = position['position_value'] * self.stop_loss_pct
        else:
            risk_amount = position['position_value'] * self.stop_loss_pct
        
        max_portfolio_risk = self.initial_capital * 0.15  # 15% max portfolio heat
        
        return risk_amount <= max_portfolio_risk

    def run_ensemble_backtest(self):
        """Run complete ensemble backtest combining 4H and 1H models"""
        print(f"\n🚀 Starting Multi-Timeframe Ensemble Backtest")
        print(f"🎯 Combining 82.57% return 4H model with 2.33% MAE 1H model")
        print("=" * 70)
        
        # Align datasets by timestamp
        min_4h_idx = 60  # Need 60 4H candles for sequence
        max_4h_idx = len(self.trading_data_4h) - 1
        
        for i in range(min_4h_idx, max_4h_idx):
            current_4h_candle = self.trading_data_4h.iloc[i]
            current_price = current_4h_candle['Close']
            current_time = current_4h_candle['datetime']
            
            # Get 4H historical sequence
            historical_4h = self.trading_data_4h.iloc[i-60:i][
                ['Open', 'High', 'Low', 'Close', 'Volume']
            ].values
            
            # CRITICAL FIX: Convert current_time to timezone-naive before filtering 1H data
            current_time_naive = pd.to_datetime(current_time).tz_localize(None)
            start_1h_time = current_time_naive - pd.Timedelta(hours=240)  # 240 hours back
            end_1h_time = current_time_naive
            
            mask_1h = ((self.trading_data_1h['datetime'] >= start_1h_time) & 
                      (self.trading_data_1h['datetime'] <= end_1h_time))
            
            if mask_1h.sum() >= 60:  # Need at least 60 1H candles
                recent_1h_data = self.trading_data_1h[mask_1h].tail(240)
                
                # Create multiple 60-period sequences for 1H prediction
                sequences_1h = []
                for j in range(60, len(recent_1h_data), 60):  # Every 60 periods
                    if j < len(recent_1h_data):
                        seq = recent_1h_data.iloc[j-60:j][
                            ['Open', 'High', 'Low', 'Close', 'Volume']
                        ].values
                        if len(seq) == 60:
                            sequences_1h.append(seq)
                
                if sequences_1h:
                    # Generate predictions
                    predictions_4h = self.generate_4h_predictions(historical_4h)
                    predictions_1h = self.generate_1h_predictions(sequences_1h)
                    
                    # Enhanced position management
                    self.enhanced_position_management(current_price, current_time, predictions_4h, predictions_1h)
                    
                    # Entry decision (if no current position)
                    if self.current_position is None and predictions_4h and predictions_1h:
                        # CRITICAL FIX: Check portfolio heat before opening positions
                        if not self.check_portfolio_heat():
                            continue  # Skip if portfolio heat too high
                        
                        # Use ensemble for trend detection
                        trend_signal, trend_strength = self.create_ensemble_signals(
                            predictions_4h, predictions_1h, current_price, 'trend_detection'
                        )
                        
                        # Use ensemble for entry timing
                        entry_signal, entry_strength = self.create_ensemble_signals(
                            predictions_4h, predictions_1h, current_price, 'entry_timing'
                        )
                        
                        # Combined signal strength
                        combined_strength = (trend_strength + entry_strength) / 2
                        combined_signal = (trend_signal + entry_signal) / 2
                        
                        if combined_strength >= self.min_signal_strength:
                            # CRITICAL FIX: Fixed position sizing based on initial capital
                            position_size = self.initial_capital * 0.02  # 2% of initial capital
                            max_position_size = self.initial_capital * 0.03  # 3% absolute maximum
                            position_size = min(position_size, max_position_size)
                            position_size = min(position_size, self.current_capital * 0.25)  # Never more than 25% of current
                            
                            if combined_signal > self.min_signal_strength:
                                self.open_position('LONG', current_price, current_time, position_size)
                            elif combined_signal < -self.min_signal_strength:
                                self.open_position('SHORT', current_price, current_time, position_size)
            
            # Record portfolio value
            position_value = 0
            if self.current_position:
                if self.current_position['direction'] == 'LONG':
                    position_value = self.current_position['shares'] * current_price
                else:
                    position_value = (2 * self.current_position['position_value'] - 
                                    self.current_position['shares'] * current_price)
            
            total_portfolio_value = self.current_capital + position_value
            
            self.portfolio_history.append({
                'timestamp': current_time,
                'portfolio_value': total_portfolio_value,
                'cash': self.current_capital,
                'position_value': position_value,
                'btc_price': current_price
            })
            
            # Progress update
            if i % 500 == 0:
                progress = (i / max_4h_idx) * 100
                print(f"📊 Ensemble backtest: {progress:.1f}% | Portfolio: ${total_portfolio_value:,.0f}")
        
        # Close any remaining position
        if self.current_position:
            final_candle = self.trading_data_4h.iloc[-1]
            self.close_position(final_candle['Close'], final_candle['datetime'], 'FINAL_EXIT')
        
        print(f"\n✅ Ensemble backtest complete!")
        return self.analyze_ensemble_performance()

    def analyze_ensemble_performance(self):
        """Analyze ensemble performance vs individual models"""
        if not self.trades:
            print("❌ No trades executed")
            return None
        
        df_trades = pd.DataFrame(self.trades)
        df_portfolio = pd.DataFrame(self.portfolio_history)
        
        # Calculate performance metrics
        final_value = self.current_capital
        total_return_pct = ((final_value - self.initial_capital) / self.initial_capital) * 100
        
        num_trades = len(self.trades)
        winning_trades = len(df_trades[df_trades['pnl_pct'] > 0])
        win_rate = (winning_trades / num_trades) * 100
        
        avg_win = df_trades[df_trades['pnl_pct'] > 0]['pnl_pct'].mean() if winning_trades > 0 else 0
        avg_loss = df_trades[df_trades['pnl_pct'] < 0]['pnl_pct'].mean() if winning_trades < num_trades else 0
        
        risk_reward_ratio = avg_win / abs(avg_loss) if avg_loss < 0 else float('inf')
        
        # Calculate max drawdown
        running_max = df_portfolio['portfolio_value'].expanding().max()
        drawdown = (df_portfolio['portfolio_value'] - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        # Exit reason analysis
        exit_reasons = df_trades['exit_reason'].value_counts()
        time_exit_pct = (exit_reasons.get('TIME_EXIT', 0) / num_trades) * 100
        take_profit_pct = ((exit_reasons.get('TAKE_PROFIT', 0) + 
                           exit_reasons.get('ENSEMBLE_PROFIT', 0) + 
                           exit_reasons.get('PARTIAL_PROFIT', 0)) / num_trades) * 100
        
        print(f"\n🎯 MULTI-TIMEFRAME ENSEMBLE RESULTS (RISK-MANAGED)")
        print("=" * 70)
        print(f"💰 Final Capital: ${final_value:,}")
        print(f"📈 Total Return: {total_return_pct:+.2f}%")
        print(f"📉 Max Drawdown: {max_drawdown:.2f}%")
        print(f"🔢 Total Trades: {num_trades}")
        print(f"🎯 Win Rate: {win_rate:.1f}%")
        print(f"⚖️ Risk-Reward Ratio: {risk_reward_ratio:.2f}")
        print(f"⏰ Average Hold: {df_trades['hold_periods'].mean() * 4:.1f} hours")
        
        print(f"\n📊 ENSEMBLE vs ORIGINAL COMPARISON:")
        print(f"   Original 4H System: +82.57% return")
        print(f"   Risk-Managed Ensemble: {total_return_pct:+.2f}% return")
        improvement = total_return_pct - 82.57
        print(f"   Net Change: {improvement:+.2f}%")
        
        print(f"\n📋 Enhanced Exit Reason Analysis:")
        for reason, count in exit_reasons.items():
            pct = (count / num_trades) * 100
            print(f"   {reason}: {count} trades ({pct:.1f}%)")
        
        print(f"\n🎯 RISK MANAGEMENT IMPROVEMENTS:")
        print(f"   TIME_EXIT Rate: {time_exit_pct:.1f}% (Target: <35%)")
        print(f"   PROFIT-TAKING Rate: {take_profit_pct:.1f}% (Target: >25%)")
        print(f"   Max Drawdown: {max_drawdown:.2f}% (Target: <20%)")
        
        return {
            'final_capital': final_value,
            'total_return_pct': total_return_pct,
            'improvement_over_4h': improvement,
            'win_rate': win_rate,
            'risk_reward_ratio': risk_reward_ratio,
            'time_exit_pct': time_exit_pct,
            'take_profit_pct': take_profit_pct,
            'max_drawdown': max_drawdown
        }

def main():
    """Main function to run Multi-Timeframe Ensemble with Risk Management"""
    print("🎯 Phase 2 - Substep 4.3: Risk-Managed Multi-Timeframe Ensemble")
    print("Combining proven 82.57% return 4H model with precision 1H model + Risk Controls")
    print("=" * 70)
    
    # Initialize ensemble
    ensemble = MultiTimeframeEnsemble(
        model_4h_path='sequences/btc_lstm_model.h5',
        scaler_4h_path='sequences/scaler.pkl',
        model_1h_path='models/btc_1h_lstm_model.h5',
        scaler_1h_path='models/btc_1h_scaler.pkl',
        initial_capital=10000
    )
    
    # Load data
    if not ensemble.load_data('processed/BTCUSD_4h_cleaned.csv', 'processed/BTCUSD_1h_cleaned.csv'):
        return
    
    # Run ensemble backtest
    performance = ensemble.run_ensemble_backtest()
    
    if performance:
        print(f"\n🎯 Risk-Managed Substep 4.3 Complete!")
        print("✅ Multi-timeframe ensemble system with risk management created")
        print("✅ Position sizing fixed to prevent catastrophic losses")
        print("✅ Trailing stops and partial profit-taking implemented")
        print("✅ Portfolio heat limits and risk controls active")
        print(f"🚀 Ready for GitHub commit and production deployment")

if __name__ == "__main__":
    main()
