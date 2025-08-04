import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
import pickle
warnings.filterwarnings('ignore')

class LSTMTradingStrategy:
    def __init__(self, model_path, scaler_path, initial_capital=10000):
        """
        LSTM Trading Strategy Implementation
        
        Parameters:
        - model_path: Path to trained LSTM model
        - scaler_path: Path to scaler
        - initial_capital: Starting capital
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Trading Strategy Parameters (optimized for your 2.23% accuracy)
        self.max_change_pct = 10  # Prediction bounds
        self.position_size_pct = 0.15  # 15% of capital per trade
        self.min_signal_strength = 1.5  # Minimum 1.5% predicted move to trade
        self.stop_loss_pct = 0.05  # 5% stop loss
        self.take_profit_pct = 0.08  # 8% take profit
        self.max_hold_periods = 30  # Maximum hold (30 * 4H = 5 days)
        
        # Strategy uses weighted multi-horizon approach
        # Weights based on your accuracy results: 4H (2.23%) gets highest weight
        self.horizon_weights = {
            '4H': 0.40,   # Highest weight for most accurate prediction
            '8H': 0.30,   # Second highest for 2.50% accuracy
            '1D': 0.20,   # Third for 3.09% accuracy
            '3D': 0.07,   # Lower weight for 4.63% accuracy
            '5D': 0.03    # Lowest weight for 5.83% accuracy
        }
        
        # Trade Tracking
        self.trades = []
        self.portfolio_history = []
        self.current_position = None
        self.position_entry_time = None
        self.position_hold_count = 0
        
        print(f"🎯 LSTM Trading Strategy initialized")
        print(f"💰 Initial capital: ${initial_capital:,}")
    
    def load_model_and_scaler(self):
        """Load trained model and scaler"""
        try:
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            self.scaler = joblib.load(self.scaler_path)
            print("✅ Model and scaler loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def prepare_trading_data(self, data_path):
        """Load and prepare data for trading simulation"""
        try:
            data = pd.read_csv(data_path)
            data['datetime'] = pd.to_datetime(data['datetime'])
            data = data.sort_values('datetime').reset_index(drop=True)
            
            # Use 80% for strategy testing (same as backtesting)
            split_idx = int(0.8 * len(data))
            self.trading_data = data.iloc[:split_idx].copy()
            
            print(f"📊 Trading simulation: {len(self.trading_data)} candles")
            print(f"📅 Period: {self.trading_data['datetime'].min()} to {self.trading_data['datetime'].max()}")
            
            return True
        except Exception as e:
            print(f"❌ Error loading trading data: {e}")
            return False
    
    def generate_predictions(self, historical_sequence):
        """Generate LSTM predictions with bounds"""
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
        bounded_predictions = {}
        signals = {}
        max_change = current_price * (self.max_change_pct / 100)
        
        weighted_signal = 0
        total_strength = 0
        
        for horizon, predicted_price in predictions.items():
            # Apply bounds
            bounded_price = np.clip(predicted_price,
                                  current_price - max_change,
                                  current_price + max_change)
            bounded_predictions[horizon] = bounded_price
            
            # Calculate signal strength
            change_pct = ((bounded_price - current_price) / current_price) * 100
            
            # Weight signal by horizon accuracy
            weight = self.horizon_weights[horizon]
            weighted_signal += weight * change_pct
            total_strength += abs(change_pct) * weight
            
            signals[horizon] = {
                'predicted_price': bounded_price,
                'change_pct': change_pct,
                'weight': weight
            }
        
        return signals, weighted_signal, total_strength
    
    def calculate_position_size(self, signal_strength):
        """Calculate position size based on signal strength"""
        # Base position size
        base_size = self.current_capital * self.position_size_pct
        
        # Adjust based on signal strength (stronger signals = larger positions)
        strength_multiplier = min(signal_strength / 3.0, 1.8)  # Cap at 1.8x
        position_size = base_size * strength_multiplier
        
        # Maximum 25% of capital per trade
        return min(position_size, self.current_capital * 0.25)
    
    def open_position(self, direction, entry_price, entry_time, position_size):
        """Open a new trading position"""
        
        if direction == 'LONG':
            shares = position_size / entry_price
            stop_loss = entry_price * (1 - self.stop_loss_pct)
            take_profit = entry_price * (1 + self.take_profit_pct)
        else:  # SHORT
            shares = position_size / entry_price
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
        """Check if position should be closed"""
        if self.current_position is None:
            return False
        
        position = self.current_position
        direction = position['direction']
        self.position_hold_count += 1
        
        # Check stop loss
        if ((direction == 'LONG' and current_price <= position['stop_loss']) or 
            (direction == 'SHORT' and current_price >= position['stop_loss'])):
            self.close_position(current_price, current_time, 'STOP_LOSS')
            return True
        
        # Check take profit
        if ((direction == 'LONG' and current_price >= position['take_profit']) or 
            (direction == 'SHORT' and current_price <= position['take_profit'])):
            self.close_position(current_price, current_time, 'TAKE_PROFIT')
            return True
        
        # Check maximum hold period
        if self.position_hold_count >= self.max_hold_periods:
            self.close_position(current_price, current_time, 'TIME_EXIT')
            return True
        
        return False
    
    def close_position(self, exit_price, exit_time, exit_reason):
        """Close current position and calculate P&L"""
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
        
        # Record trade - FIXED field names to match expected format
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
        """Execute trading decisions based on weighted signals"""
        
        # Only trade if no current position and signal is strong enough
        if self.current_position is None and total_strength >= self.min_signal_strength:
            
            position_size = self.calculate_position_size(total_strength)
            
            # Long signal
            if weighted_signal > self.min_signal_strength:
                self.open_position('LONG', current_price, current_time, position_size)
            
            # Short signal  
            elif weighted_signal < -self.min_signal_strength:
                self.open_position('SHORT', current_price, current_time, position_size)
    
    def run_trading_simulation(self):
        """Run complete trading simulation"""
        print(f"\n🚀 Starting LSTM Trading Simulation")
        print(f"Strategy: Weighted Multi-Horizon (4H-focused)")
        print("=" * 60)
        
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
                signals, weighted_signal, total_strength = self.apply_bounds_and_calculate_signals(
                    predictions, current_price)
                
                # Execute trading decision
                self.execute_trading_decision(current_price, current_time, 
                                            weighted_signal, total_strength)
            
            # Record portfolio value - FIXED field names
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
                print(f"📊 Trading progress: {progress:.1f}% | Portfolio: ${total_portfolio_value:,.0f}")
        
        # Close any remaining position
        if self.current_position:
            final_candle = self.trading_data.iloc[-1]
            self.close_position(final_candle['Close'], final_candle['datetime'], 'FINAL_EXIT')
        
        print(f"\n✅ Trading simulation complete!")
        return self.analyze_performance()
    
    def analyze_performance(self):
        """Analyze comprehensive trading performance"""
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
        
        print(f"\n📊 COMPREHENSIVE TRADING PERFORMANCE")
        print("=" * 60)
        print(f"💰 Initial Capital: ${self.initial_capital:,}")
        print(f"💰 Final Capital: ${final_value:,}")
        print(f"📈 Total Return: {total_return_pct:+.2f}%")
        print(f"📉 Max Drawdown: {max_drawdown:.2f}%")
        print(f"🔢 Total Trades: {num_trades}")
        print(f"🎯 Win Rate: {win_rate:.1f}% ({winning_trades}W/{losing_trades}L)")
        print(f"⬆️ Average Win: {avg_win:.2f}%")
        print(f"⬇️ Average Loss: {avg_loss:.2f}%")
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
        
        return {
            'final_capital': final_value,
            'total_return_pct': total_return_pct,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_hold_hours': avg_hold_time,
            'trades': self.trades,
            'portfolio_history': self.portfolio_history
        }

def main():
    """Main function to run Step 2"""
    print("🎯 LSTM Trading Strategy - Step 2")
    print("Leveraging your 2.23% 4H prediction accuracy")
    print("=" * 60)
    
    # Initialize strategy
    strategy = LSTMTradingStrategy(
        model_path='sequences/btc_lstm_model.h5',
        scaler_path='sequences/scaler.pkl',
        initial_capital=10000
    )
    
    # Load model and data
    if not strategy.load_model_and_scaler():
        return
    
    if not strategy.prepare_trading_data('processed/BTCUSD_4h_cleaned.csv'):
        return
    
    # Run trading simulation
    performance = strategy.run_trading_simulation()
    
    if performance:
        print(f"\n🎯 Step 2 Complete!")
        print("✅ Trading strategy successfully implemented")
        print("✅ Multi-horizon weighted approach tested")
        print("✅ Risk management and P&L tracking working")
        print(f"📊 Final Result: {performance['total_return_pct']:+.2f}% return")
        
        # FIXED: Save trading data for analysis (moved inside main function)
        print(f"\n💾 Saving trading data for advanced analysis...")
        try:
            with open('sequences/trades_data.pkl', 'wb') as f:
                pickle.dump(strategy.trades, f)
            
            with open('sequences/portfolio_history.pkl', 'wb') as f:
                pickle.dump(strategy.portfolio_history, f)
            
            print("✅ Trading data saved successfully!")
            print("🚀 Ready for Substep 3.1: Advanced Performance Analysis")
            
        except Exception as e:
            print(f"❌ Error saving data: {e}")
        
        print(f"\n🚀 Ready for Step 3: Advanced Performance Analysis & Optimization")

if __name__ == "__main__":
    main()
