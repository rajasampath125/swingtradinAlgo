import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class OptimizedParameterValidator:
    def __init__(self, model_path, scaler_path, data_path, initial_capital=10000):
        """
        Validate optimized parameters on full dataset
        Tests #1 ranked parameters from quick optimization on complete historical data
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.data_path = data_path
        self.initial_capital = initial_capital

        self.current_capital = initial_capital
        
        # BEST PARAMETERS from quick optimization (#1 ranked)
        self.take_profit_pct = 0.12      # 12% vs original 8%
        self.stop_loss_pct = 0.06        # 6% vs original 5%  
        self.max_hold_periods = 35       # 35 vs original 30
        self.min_signal_strength = 2.0   # 2.0 vs original 1.5
        self.position_size_pct = 0.20    # 20% vs original 15%
        
        # Strategy components
        self.horizon_weights = {
            '4H': 0.40, '8H': 0.30, '1D': 0.20, '3D': 0.07, '5D': 0.03
        }
        
        # Trade tracking
        self.trades = []
        self.portfolio_history = []
        self.current_position = None
        self.position_hold_count = 0
        
        print(f"🎯 Optimized Parameter Validator Initialized")
        print(f"💰 Initial Capital: ${initial_capital:,}")
        print(f"⚙️ Testing optimized parameters on FULL dataset")
    
    def load_model_and_data(self):
        """Load model, scaler, and FULL trading dataset"""
        try:
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            self.scaler = joblib.load(self.scaler_path)
            
            # Load COMPLETE dataset (not reduced like quick optimization)
            data = pd.read_csv(self.data_path)
            data['datetime'] = pd.to_datetime(data['datetime'])
            data = data.sort_values('datetime').reset_index(drop=True)
            
            # Use same 80% split as original backtesting
            split_idx = int(0.8 * len(data))
            self.trading_data = data.iloc[:split_idx].copy()
            
            print(f"✅ Model, scaler, and FULL dataset loaded successfully")
            print(f"📊 Full dataset: {len(self.trading_data)} periods")
            print(f"📅 Date range: {self.trading_data['datetime'].min()} to {self.trading_data['datetime'].max()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading components: {e}")
            return False
    
    def generate_predictions(self, historical_sequence):
        """Generate LSTM predictions using trained model"""
        try:
            scaled_sequence = self.scaler.transform(historical_sequence)
            X_input = scaled_sequence.reshape(1, 60, 5)
            prediction_scaled = self.model.predict(X_input, verbose=0)[0]
            
            predictions = {}
            horizons = ['4H', '8H', '1D', '3D', '5D']
            
            for i, horizon in enumerate(horizons):
                dummy_array = np.zeros((1, 5))
                dummy_array[0, 3] = prediction_scaled[i]
                raw_prediction = self.scaler.inverse_transform(dummy_array)[0, 3]
                predictions[horizon] = raw_prediction
            
            return predictions
        except Exception as e:
            return None
    
    def apply_bounds_and_calculate_signals(self, predictions, current_price):
        """Apply bounds and generate weighted trading signals"""
        max_change = current_price * 0.10  # 10% bounds
        weighted_signal = 0
        total_strength = 0
        
        for horizon, predicted_price in predictions.items():
            # Apply bounds
            bounded_price = np.clip(predicted_price,
                                  current_price - max_change,
                                  current_price + max_change)
            
            # Calculate signal strength
            change_pct = ((bounded_price - current_price) / current_price) * 100
            
            # Weight by horizon accuracy
            weight = self.horizon_weights[horizon]
            weighted_signal += weight * change_pct
            total_strength += abs(change_pct) * weight
        
        return weighted_signal, total_strength
    
    def calculate_position_size(self, signal_strength):
        """Calculate position size based on signal strength"""
        base_size = self.current_capital * self.position_size_pct
        strength_multiplier = min(signal_strength / 3.0, 1.8)  # Cap at 1.8x
        position_size = base_size * strength_multiplier
        return min(position_size, self.current_capital * 0.25)  # Max 25% per trade
    
    def open_position(self, direction, entry_price, entry_time, position_size):
        """Open new trading position with optimized parameters"""
        shares = position_size / entry_price
        
        if direction == 'LONG':
            stop_loss = entry_price * (1 - self.stop_loss_pct)
            take_profit = entry_price * (1 + self.take_profit_pct)
        else:  # SHORT
            stop_loss = entry_price * (1 + self.stop_loss_pct)
            take_profit = entry_price * (1 - self.take_profit_pct)
        
        self.current_position = {
            'direction': direction,
            'entry_price': entry_price,
            'entry_time': entry_time,
            'shares': shares,
            'position_value': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
        
        self.position_hold_count = 0
        self.current_capital -= position_size
        
        print(f"📈 {direction} opened: ${entry_price:,.0f} | Size: ${position_size:,.0f}")
    
    def check_exit_conditions(self, current_price, current_time):
        """Check if position should be closed using optimized parameters"""
        if self.current_position is None:
            return False
        
        position = self.current_position
        direction = position['direction']
        self.position_hold_count += 1
        
        # Check stop loss (6% vs original 5%)
        if ((direction == 'LONG' and current_price <= position['stop_loss']) or 
            (direction == 'SHORT' and current_price >= position['stop_loss'])):
            self.close_position(current_price, current_time, 'STOP_LOSS')
            return True
        
        # Check take profit (12% vs original 8%)
        if ((direction == 'LONG' and current_price >= position['take_profit']) or 
            (direction == 'SHORT' and current_price <= position['take_profit'])):
            self.close_position(current_price, current_time, 'TAKE_PROFIT')
            return True
        
        # Check maximum hold period (35 vs original 30)
        if self.position_hold_count >= self.max_hold_periods:
            self.close_position(current_price, current_time, 'TIME_EXIT')
            return True
        
        return False
    
    def close_position(self, exit_price, exit_time, exit_reason):
        """Close position and calculate P&L"""
        if self.current_position is None:
            return
        
        position = self.current_position
        direction = position['direction']
        entry_price = position['entry_price']
        shares = position['shares']
        
        # Calculate P&L
        exit_value = shares * exit_price
        
        if direction == 'LONG':
            pnl_usd = exit_value - position['position_value']
            pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:  # SHORT
            pnl_usd = position['position_value'] - exit_value  
            pnl_pct = (entry_price - exit_price) / entry_price * 100
        
        self.current_capital += exit_value
        
        # Record trade
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
        
        # Clear position
        self.current_position = None
        self.position_hold_count = 0
    
    def execute_trading_decision(self, current_price, current_time, weighted_signal, total_strength):
        """Execute trading decisions using optimized signal threshold"""
        
        # Only trade if no position and signal exceeds optimized threshold (2.0 vs 1.5)
        if (self.current_position is None and 
            total_strength >= self.min_signal_strength and
            abs(weighted_signal) >= self.min_signal_strength):
            
            position_size = self.calculate_position_size(total_strength)
            
            # Long signal
            if weighted_signal > self.min_signal_strength:
                self.open_position('LONG', current_price, current_time, position_size)
            
            # Short signal  
            elif weighted_signal < -self.min_signal_strength:
                self.open_position('SHORT', current_price, current_time, position_size)
    
    def run_full_validation(self):
        """Run complete validation on full dataset"""
        print(f"\n🚀 Starting Full Dataset Validation")
        print(f"Strategy: Optimized Parameters on Complete Historical Data")
        print("=" * 70)
        
        for i in range(60, len(self.trading_data)):
            current_candle = self.trading_data.iloc[i]
            current_price = current_candle['Close']
            current_time = current_candle['datetime']
            
            # Check exit conditions first
            self.check_exit_conditions(current_price, current_time)
            
            # Generate predictions and signals
            historical_sequence = self.trading_data.iloc[i-60:i][
                ['Open', 'High', 'Low', 'Close', 'Volume']
            ].values
            
            predictions = self.generate_predictions(historical_sequence)
            if predictions:
                weighted_signal, total_strength = self.apply_bounds_and_calculate_signals(
                    predictions, current_price)
                
                # Execute trading decision
                self.execute_trading_decision(current_price, current_time, 
                                            weighted_signal, total_strength)
            
            # Record portfolio value
            position_value = 0
            if self.current_position:
                if self.current_position['direction'] == 'LONG':
                    position_value = self.current_position['shares'] * current_price
                else:  # SHORT
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
            if i % 1000 == 0:
                progress = (i / len(self.trading_data)) * 100
                print(f"📊 Validation progress: {progress:.1f}% | Portfolio: ${total_portfolio_value:,.0f}")
        
        # Close any remaining position
        if self.current_position:
            final_candle = self.trading_data.iloc[-1]
            self.close_position(final_candle['Close'], final_candle['datetime'], 'FINAL_EXIT')
        
        print(f"\n✅ Full Dataset Validation Complete!")
        return self.analyze_comprehensive_performance()
    
    def analyze_comprehensive_performance(self):
        """Comprehensive performance analysis with comparisons"""
        if not self.trades:
            print("❌ No trades executed")
            return None
        
        df_trades = pd.DataFrame(self.trades)
        df_portfolio = pd.DataFrame(self.portfolio_history)
        
        # Calculate key performance metrics
        final_value = self.current_capital
        total_return_pct = ((final_value - self.initial_capital) / self.initial_capital) * 100
        
        num_trades = len(self.trades)
        winning_trades = len(df_trades[df_trades['pnl_pct'] > 0])
        losing_trades = num_trades - winning_trades
        win_rate = (winning_trades / num_trades) * 100 if num_trades > 0 else 0
        
        avg_win = df_trades[df_trades['pnl_pct'] > 0]['pnl_pct'].mean() if winning_trades > 0 else 0
        avg_loss = df_trades[df_trades['pnl_pct'] < 0]['pnl_pct'].mean() if losing_trades > 0 else 0
        
        max_win = df_trades['pnl_pct'].max()
        max_loss = df_trades['pnl_pct'].min()
        
        avg_hold_time = df_trades['hold_periods'].mean() * 4  # Convert to hours
        
        # Calculate additional metrics
        profit_factor = (df_trades[df_trades['pnl_usd'] > 0]['pnl_usd'].sum() / 
                        abs(df_trades[df_trades['pnl_usd'] < 0]['pnl_usd'].sum())) if losing_trades > 0 else float('inf')
        
        # Calculate max drawdown
        running_max = df_portfolio['portfolio_value'].expanding().max()
        drawdown = (df_portfolio['portfolio_value'] - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        # Risk-reward ratio
        risk_reward_ratio = avg_win / abs(avg_loss) if avg_loss < 0 else float('inf')
        
        print(f"\n🎯 FULL DATASET VALIDATION RESULTS")
        print("=" * 70)
        print(f"💰 Initial Capital: ${self.initial_capital:,}")
        print(f"💰 Final Capital: ${final_value:,}")
        print(f"📈 Total Return: {total_return_pct:+.2f}%")
        print(f"📉 Max Drawdown: {max_drawdown:.2f}%")
        print(f"🔢 Total Trades: {num_trades}")
        print(f"🎯 Win Rate: {win_rate:.1f}% ({winning_trades}W/{losing_trades}L)")
        print(f"⬆️ Average Win: {avg_win:.2f}%")
        print(f"⬇️ Average Loss: {avg_loss:.2f}%")
        print(f"⚖️ Risk-Reward Ratio: {risk_reward_ratio:.2f}")
        print(f"🚀 Best Trade: {max_win:.2f}%")
        print(f"💥 Worst Trade: {max_loss:.2f}%")
        print(f"⚖️ Profit Factor: {profit_factor:.2f}")
        print(f"⏰ Average Hold: {avg_hold_time:.1f} hours")
        
        # Break down by exit reason
        exit_reasons = df_trades['exit_reason'].value_counts()
        print(f"\n📋 Exit Reason Breakdown:")
        for reason, count in exit_reasons.items():
            pct = (count / num_trades) * 100
            print(f"  {reason}: {count} trades ({pct:.1f}%)")
        
        # COMPREHENSIVE COMPARISON ANALYSIS
        print(f"\n📊 PERFORMANCE COMPARISON ANALYSIS")
        print("=" * 70)
        
        # Comparison vs Original System
        print(f"📈 vs ORIGINAL SYSTEM:")
        print(f"   Total Return: {total_return_pct:+.2f}% vs +63.84% (Original)")
        print(f"   Risk-Reward: {risk_reward_ratio:.2f} vs 1.12 (Original)")
        print(f"   Win Rate: {win_rate:.1f}% vs 50.5% (Original)")
        print(f"   Total Trades: {num_trades} vs 410 (Original)")
        
        # Comparison vs Quick Optimization
        print(f"\n⚡ vs QUICK OPTIMIZATION:")
        print(f"   Total Return: {total_return_pct:+.2f}% vs +1.28% (Quick)")
        print(f"   Dataset Size: {len(self.trading_data)} vs 2,000 periods (Quick)")
        print(f"   Trade Count: {num_trades} vs ~47 trades (Quick)")
        
        # TIME_EXIT analysis (our target metric)
        time_exit_pct = (exit_reasons.get('TIME_EXIT', 0) / num_trades) * 100
        take_profit_pct = (exit_reasons.get('TAKE_PROFIT', 0) / num_trades) * 100
        
        print(f"\n🎯 TARGET METRIC ANALYSIS:")
        print(f"   TIME_EXIT Rate: {time_exit_pct:.1f}% vs 41.7% (Original target: <35%)")
        print(f"   TAKE_PROFIT Rate: {take_profit_pct:.1f}% vs 22.2% (Original target: >25%)")
        
        # Success/Failure Assessment
        print(f"\n🏆 OPTIMIZATION SUCCESS ASSESSMENT:")
        
        success_criteria = []
        if total_return_pct > 50:
            success_criteria.append("✅ Return > 50%")
        else:
            success_criteria.append("❌ Return < 50%")
            
        if risk_reward_ratio > 1.3:
            success_criteria.append("✅ Risk-Reward > 1.3")
        else:
            success_criteria.append("❌ Risk-Reward < 1.3")
            
        if time_exit_pct < 35:
            success_criteria.append("✅ TIME_EXIT < 35%")
        else:
            success_criteria.append("❌ TIME_EXIT > 35%")
            
        if take_profit_pct > 25:
            success_criteria.append("✅ TAKE_PROFIT > 25%")
        else:
            success_criteria.append("❌ TAKE_PROFIT < 25%")
        
        for criterion in success_criteria:
            print(f"   {criterion}")
        
        # Final recommendation
        successful_criteria = len([c for c in success_criteria if "✅" in c])
        if successful_criteria >= 3:
            print(f"\n🎉 RECOMMENDATION: OPTIMIZATION SUCCESSFUL!")
            print(f"   Implement optimized parameters for improved performance")
        elif successful_criteria >= 2:
            print(f"\n⚡ RECOMMENDATION: PARTIAL SUCCESS")
            print(f"   Consider targeted parameter refinement or Phase 2 enhancement")
        else:
            print(f"\n🔄 RECOMMENDATION: EXPAND PARAMETER SEARCH")
            print(f"   Current optimization insufficient - try broader parameter ranges")
        
        return {
            'final_capital': final_value,
            'total_return_pct': total_return_pct,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'risk_reward_ratio': risk_reward_ratio,
            'profit_factor': profit_factor,
            'time_exit_pct': time_exit_pct,
            'take_profit_pct': take_profit_pct,
            'trades': self.trades,
            'portfolio_history': self.portfolio_history
        }

def main():
    """Main function to run full dataset validation"""
    print("🎯 FULL DATASET VALIDATION - Substep 3.2 Validation")
    print("Testing optimized parameters on complete historical data")
    print("=" * 70)
    
    # Initialize validator with optimized parameters
    validator = OptimizedParameterValidator(
        model_path='sequences/btc_lstm_model.h5',
        scaler_path='sequences/scaler.pkl',
        data_path='processed/BTCUSD_4h_cleaned.csv',
        initial_capital=10000
    )
    
    # Load model and data
    if not validator.load_model_and_data():
        return
    
    # Run full validation
    performance = validator.run_full_validation()
    
    if performance:
        print(f"\n✅ Full Dataset Validation Complete!")
        print(f"📊 Results will guide next phase of optimization strategy")

if __name__ == "__main__":
    main()
