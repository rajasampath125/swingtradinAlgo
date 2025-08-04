import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from itertools import product
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ParameterOptimizer:
    def __init__(self, model_path, scaler_path, data_path, initial_capital=10000):
        """Parameter optimization for LSTM trading strategy"""
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.data_path = data_path
        self.initial_capital = initial_capital
        
        # Load model and data once
        self.load_model_and_data()
        
        # Parameter ranges to test
        self.param_ranges = {
            'take_profit_pct': [0.06, 0.08, 0.10, 0.12],  # Current: 0.08
            'stop_loss_pct': [0.04, 0.05, 0.06, 0.07],    # Current: 0.05
            'max_hold_periods': [25, 30, 35, 40],          # Current: 30
            'min_signal_strength': [1.0, 1.5, 2.0, 2.5],  # Current: 1.5
            'position_size_pct': [0.10, 0.15, 0.20, 0.25] # Current: 0.15
        }
        
        self.results = []
        
        print(f"🎯 Parameter Optimizer initialized")
        print(f"💰 Testing {len(list(product(*self.param_ranges.values())))} parameter combinations")
    
    def load_model_and_data(self):
        """Load model, scaler, and trading data"""
        try:
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            self.scaler = joblib.load(self.scaler_path)
            
            # Load and prepare data
            data = pd.read_csv(self.data_path)
            data['datetime'] = pd.to_datetime(data['datetime'])
            data = data.sort_values('datetime').reset_index(drop=True)
            
            # Use same 80% split as original backtesting
            split_idx = int(0.8 * len(data))
            self.trading_data = data.iloc[:split_idx].copy()
            
            print(f"✅ Model, scaler, and data loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error loading components: {e}")
            return False
    
    def test_parameter_combination(self, params):
        """Test a single parameter combination"""
        try:
            # Initialize trading variables
            current_capital = self.initial_capital
            trades = []
            current_position = None
            position_hold_count = 0
            
            # Horizon weights (keep same as original)
            horizon_weights = {
                '4H': 0.40, '8H': 0.30, '1D': 0.20, '3D': 0.07, '5D': 0.03
            }
            
            # Run trading simulation with new parameters
            for i in range(60, len(self.trading_data)):
                current_candle = self.trading_data.iloc[i]
                current_price = current_candle['Close']
                current_time = current_candle['datetime']
                
                # Check exit conditions
                if current_position:
                    position_hold_count += 1
                    direction = current_position['direction']
                    
                    # Stop loss check
                    if ((direction == 'LONG' and current_price <= current_position['stop_loss']) or 
                        (direction == 'SHORT' and current_price >= current_position['stop_loss'])):
                        trades.append(self.close_position(current_position, current_price, 'STOP_LOSS'))
                        current_capital = trades[-1]['exit_capital']
                        current_position = None
                        position_hold_count = 0
                    
                    # Take profit check
                    elif ((direction == 'LONG' and current_price >= current_position['take_profit']) or 
                          (direction == 'SHORT' and current_price <= current_position['take_profit'])):
                        trades.append(self.close_position(current_position, current_price, 'TAKE_PROFIT'))
                        current_capital = trades[-1]['exit_capital']
                        current_position = None
                        position_hold_count = 0
                    
                    # Time exit check
                    elif position_hold_count >= params['max_hold_periods']:
                        trades.append(self.close_position(current_position, current_price, 'TIME_EXIT'))
                        current_capital = trades[-1]['exit_capital']
                        current_position = None
                        position_hold_count = 0
                
                # Generate new position if none exists
                if not current_position:
                    # Generate LSTM predictions
                    historical_sequence = self.trading_data.iloc[i-60:i][
                        ['Open', 'High', 'Low', 'Close', 'Volume']
                    ].values
                    
                    predictions = self.generate_predictions(historical_sequence)
                    if predictions:
                        signals, weighted_signal, total_strength = self.calculate_signals(
                            predictions, current_price, horizon_weights)
                        
                        # Open position based on signal
                        if (total_strength >= params['min_signal_strength'] and
                            abs(weighted_signal) >= params['min_signal_strength']):
                            
                            position_size = current_capital * params['position_size_pct']
                            position_size = min(position_size, current_capital * 0.3)  # Max 30%
                            
                            if weighted_signal > 0:  # LONG
                                current_position = self.open_position(
                                    'LONG', current_price, current_time, position_size, params)
                                current_capital -= position_size
                            elif weighted_signal < 0:  # SHORT
                                current_position = self.open_position(
                                    'SHORT', current_price, current_time, position_size, params)
                                current_capital -= position_size
            
            # Close any remaining position
            if current_position:
                final_candle = self.trading_data.iloc[-1]
                trades.append(self.close_position(current_position, final_candle['Close'], 'FINAL_EXIT'))
                current_capital = trades[-1]['exit_capital']
            
            # Calculate performance metrics
            if trades:
                return self.calculate_performance_metrics(trades, params)
            else:
                return None
                
        except Exception as e:
            print(f"❌ Error testing parameters: {e}")
            return None
    
    def generate_predictions(self, historical_sequence):
        """Generate LSTM predictions"""
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
        except:
            return None
    
    def calculate_signals(self, predictions, current_price, horizon_weights):
        """Calculate weighted trading signals"""
        max_change = current_price * 0.10  # 10% bounds
        weighted_signal = 0
        total_strength = 0
        signals = {}
        
        for horizon, predicted_price in predictions.items():
            bounded_price = np.clip(predicted_price,
                                  current_price - max_change,
                                  current_price + max_change)
            
            change_pct = ((bounded_price - current_price) / current_price) * 100
            weight = horizon_weights[horizon]
            
            weighted_signal += weight * change_pct
            total_strength += abs(change_pct) * weight
            
            signals[horizon] = {
                'predicted_price': bounded_price,
                'change_pct': change_pct,
                'weight': weight
            }
        
        return signals, weighted_signal, total_strength
    
    def open_position(self, direction, entry_price, entry_time, position_size, params):
        """Open trading position with parameters"""
        shares = position_size / entry_price
        
        if direction == 'LONG':
            stop_loss = entry_price * (1 - params['stop_loss_pct'])
            take_profit = entry_price * (1 + params['take_profit_pct'])
        else:  # SHORT
            stop_loss = entry_price * (1 + params['stop_loss_pct'])
            take_profit = entry_price * (1 - params['take_profit_pct'])
        
        return {
            'direction': direction,
            'entry_price': entry_price,
            'entry_time': entry_time,
            'shares': shares,
            'position_value': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
    
    def close_position(self, position, exit_price, exit_reason):
        """Close position and calculate P&L"""
        direction = position['direction']
        entry_price = position['entry_price']
        shares = position['shares']
        exit_value = shares * exit_price
        
        if direction == 'LONG':
            pnl_usd = exit_value - position['position_value']
            pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:  # SHORT
            pnl_usd = position['position_value'] - exit_value
            pnl_pct = (entry_price - exit_price) / entry_price * 100
        
        return {
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'pnl_usd': pnl_usd,
            'exit_reason': exit_reason,
            'exit_capital': position['position_value'] + pnl_usd
        }
    
    def calculate_performance_metrics(self, trades, params):
        """Calculate comprehensive performance metrics"""
        if not trades:
            return None
        
        df_trades = pd.DataFrame(trades)
        
        # Basic metrics
        total_pnl = df_trades['pnl_usd'].sum()
        final_capital = self.initial_capital + total_pnl
        total_return_pct = (total_pnl / self.initial_capital) * 100
        
        num_trades = len(trades)
        winning_trades = len(df_trades[df_trades['pnl_pct'] > 0])
        win_rate = (winning_trades / num_trades) * 100
        
        avg_win = df_trades[df_trades['pnl_pct'] > 0]['pnl_pct'].mean() if winning_trades > 0 else 0
        avg_loss = df_trades[df_trades['pnl_pct'] < 0]['pnl_pct'].mean() if winning_trades < num_trades else 0
        
        # Risk-reward ratio
        risk_reward_ratio = avg_win / abs(avg_loss) if avg_loss < 0 else float('inf')
        
        # Exit reason breakdown
        exit_reasons = df_trades['exit_reason'].value_counts()
        time_exit_pct = (exit_reasons.get('TIME_EXIT', 0) / num_trades) * 100
        take_profit_pct = (exit_reasons.get('TAKE_PROFIT', 0) / num_trades) * 100
        stop_loss_pct = (exit_reasons.get('STOP_LOSS', 0) / num_trades) * 100
        
        return {
            'params': params,
            'final_capital': final_capital,
            'total_return_pct': total_return_pct,
            'total_pnl': total_pnl,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'risk_reward_ratio': risk_reward_ratio,
            'time_exit_pct': time_exit_pct,
            'take_profit_pct': take_profit_pct,
            'stop_loss_pct': stop_loss_pct
        }
    
    def run_optimization(self):
        """Run full parameter optimization"""
        print(f"\n🚀 Starting Parameter Optimization")
        print("=" * 60)
        
        param_combinations = list(product(*self.param_ranges.values()))
        total_combinations = len(param_combinations)
        
        print(f"Testing {total_combinations} parameter combinations...")
        
        best_result = None
        best_score = -float('inf')
        
        for i, combination in enumerate(param_combinations):
            # Create parameter dict
            params = dict(zip(self.param_ranges.keys(), combination))
            
            # Test this combination
            result = self.test_parameter_combination(params)
            
            if result:
                # Score based on risk-adjusted return and risk-reward ratio
                score = (result['total_return_pct'] * 0.6 + 
                        result['risk_reward_ratio'] * 20 * 0.4)
                
                result['optimization_score'] = score
                self.results.append(result)
                
                if score > best_score:
                    best_score = score
                    best_result = result
            
            # Progress update
            if (i + 1) % 50 == 0 or (i + 1) == total_combinations:
                progress = ((i + 1) / total_combinations) * 100
                print(f"📊 Progress: {progress:.1f}% ({i + 1}/{total_combinations})")
        
        print(f"\n✅ Optimization complete!")
        print(f"📊 Tested {len(self.results)} valid parameter combinations")
        
        return self.analyze_results()
    
    def analyze_results(self):
        """Analyze optimization results"""
        if not self.results:
            print("❌ No valid results found")
            return None
        
        # Sort by optimization score
        sorted_results = sorted(self.results, key=lambda x: x['optimization_score'], reverse=True)
        
        print(f"\n🏆 TOP 5 PARAMETER COMBINATIONS")
        print("=" * 60)
        
        for i, result in enumerate(sorted_results[:5]):
            print(f"\n#{i+1} - Score: {result['optimization_score']:.2f}")
            print(f"Return: {result['total_return_pct']:+.2f}% | Win Rate: {result['win_rate']:.1f}%")
            print(f"Risk-Reward: {result['risk_reward_ratio']:.2f} | Trades: {result['num_trades']}")
            print(f"Time Exits: {result['time_exit_pct']:.1f}% | Take Profits: {result['take_profit_pct']:.1f}%")
            
            params = result['params']
            print(f"Parameters:")
            for param, value in params.items():
                print(f"  {param}: {value}")
        
        # Compare with original performance
        print(f"\n📊 COMPARISON WITH ORIGINAL SYSTEM:")
        print(f"Original: +63.84% return, 1.12 risk-reward, 41.7% time exits")
        best = sorted_results[0]
        print(f"Optimized: {best['total_return_pct']:+.2f}% return, {best['risk_reward_ratio']:.2f} risk-reward, {best['time_exit_pct']:.1f}% time exits")
        
        improvement = best['total_return_pct'] - 63.84
        print(f"Improvement: {improvement:+.2f}% return")
        
        return sorted_results

def main():
    """Main function to run Substep 3.2"""
    print("🎯 SUBSTEP 3.2: Strategy Parameter Optimization")
    print("Optimizing based on Substep 3.1 analysis recommendations")
    print("=" * 60)
    
    # Initialize optimizer
    optimizer = ParameterOptimizer(
        model_path='sequences/btc_lstm_model.h5',
        scaler_path='sequences/scaler.pkl',
        data_path='processed/BTCUSD_4h_cleaned.csv',
        initial_capital=10000
    )
    
    # Run optimization
    results = optimizer.run_optimization()
    
    if results:
        print(f"\n🎯 Substep 3.2 Complete!")
        print("✅ Parameter optimization completed")
        print("✅ Best parameter combinations identified")
        print("✅ Performance improvements quantified")
        print(f"\n🚀 Ready for Substep 3.3: Implement Optimized Strategy")

if __name__ == "__main__":
    main()
