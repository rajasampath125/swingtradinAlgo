import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from itertools import product
import time

class QuickParameterOptimizer:
    def __init__(self, model_path, scaler_path, data_path, initial_capital=10000):
        """Quick parameter optimization with reduced search space"""
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.data_path = data_path
        self.initial_capital = initial_capital
        
        self.load_model_and_data()
        
        # REDUCED parameter ranges - only test most promising combinations
        self.param_ranges = {
            'take_profit_pct': [0.08, 0.10, 0.12],     # 3 values instead of 4
            'stop_loss_pct': [0.04, 0.05, 0.06],       # 3 values instead of 4
            'max_hold_periods': [30, 35],               # 2 values instead of 4
            'min_signal_strength': [1.5, 2.0],         # 2 values instead of 4
            'position_size_pct': [0.15, 0.20]          # 2 values instead of 4
        }
        
        # Total combinations: 3 × 3 × 2 × 2 × 2 = 72 (instead of 1,024)
        total_combinations = np.prod([len(v) for v in self.param_ranges.values()])
        
        print(f"🎯 Quick Parameter Optimizer initialized")
        print(f"💰 Testing {total_combinations} focused parameter combinations")
        print(f"⚡ Estimated time: 5-15 minutes (vs 17+ hours)")
        
        self.results = []
    
    def load_model_and_data(self):
        """Load model, scaler, and REDUCED trading data for speed"""
        try:
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            self.scaler = joblib.load(self.scaler_path)
            
            # Load and prepare data - USE ONLY LAST 2000 PERIODS for speed
            data = pd.read_csv(self.data_path)
            data['datetime'] = pd.to_datetime(data['datetime'])
            data = data.sort_values('datetime').reset_index(drop=True)
            
            split_idx = int(0.8 * len(data))
            full_trading_data = data.iloc[:split_idx].copy()
            
            # SPEED OPTIMIZATION: Use only last 2000 periods instead of full 8757
            if len(full_trading_data) > 2000:
                self.trading_data = full_trading_data.tail(2000).reset_index(drop=True)
                print(f"⚡ Using last {len(self.trading_data)} periods for optimization speed")
            else:
                self.trading_data = full_trading_data
            
            print(f"✅ Model, scaler, and data loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error loading components: {e}")
            return False
    
    def test_parameter_combination(self, params):
        """Test a single parameter combination - SIMPLIFIED VERSION"""
        try:
            current_capital = self.initial_capital
            trades = []
            current_position = None
            position_hold_count = 0
            
            horizon_weights = {'4H': 0.40, '8H': 0.30, '1D': 0.20, '3D': 0.07, '5D': 0.03}
            
            # SPEED OPTIMIZATION: Process every 4th period instead of every period
            for i in range(60, len(self.trading_data), 4):  # Step by 4 for speed
                current_candle = self.trading_data.iloc[i]
                current_price = current_candle['Close']
                current_time = current_candle['datetime']
                
                # Check exit conditions
                if current_position:
                    position_hold_count += 4  # Adjust for step size
                    direction = current_position['direction']
                    
                    # Stop loss
                    if ((direction == 'LONG' and current_price <= current_position['stop_loss']) or 
                        (direction == 'SHORT' and current_price >= current_position['stop_loss'])):
                        trades.append(self.close_position(current_position, current_price, 'STOP_LOSS'))
                        current_capital = trades[-1]['exit_capital']
                        current_position = None
                        position_hold_count = 0
                    
                    # Take profit
                    elif ((direction == 'LONG' and current_price >= current_position['take_profit']) or 
                          (direction == 'SHORT' and current_price <= current_position['take_profit'])):
                        trades.append(self.close_position(current_position, current_price, 'TAKE_PROFIT'))
                        current_capital = trades[-1]['exit_capital']
                        current_position = None
                        position_hold_count = 0
                    
                    # Time exit
                    elif position_hold_count >= params['max_hold_periods']:
                        trades.append(self.close_position(current_position, current_price, 'TIME_EXIT'))
                        current_capital = trades[-1]['exit_capital']
                        current_position = None
                        position_hold_count = 0
                
                # Generate new position
                if not current_position and i >= 60:
                    historical_sequence = self.trading_data.iloc[i-60:i][
                        ['Open', 'High', 'Low', 'Close', 'Volume']
                    ].values
                    
                    predictions = self.generate_predictions(historical_sequence)
                    if predictions:
                        signals, weighted_signal, total_strength = self.calculate_signals(
                            predictions, current_price, horizon_weights)
                        
                        if (total_strength >= params['min_signal_strength'] and
                            abs(weighted_signal) >= params['min_signal_strength']):
                            
                            position_size = current_capital * params['position_size_pct']
                            position_size = min(position_size, current_capital * 0.3)
                            
                            if weighted_signal > 0:  # LONG
                                current_position = self.open_position(
                                    'LONG', current_price, current_time, position_size, params)
                                current_capital -= position_size
                            elif weighted_signal < 0:  # SHORT
                                current_position = self.open_position(
                                    'SHORT', current_price, current_time, position_size, params)
                                current_capital -= position_size
            
            # Close remaining position
            if current_position:
                final_candle = self.trading_data.iloc[-1]
                trades.append(self.close_position(current_position, final_candle['Close'], 'FINAL_EXIT'))
                current_capital = trades[-1]['exit_capital']
            
            if trades:
                return self.calculate_performance_metrics(trades, params)
            else:
                return None
                
        except Exception as e:
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
        max_change = current_price * 0.10
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
        
        return signals, weighted_signal, total_strength
    
    def open_position(self, direction, entry_price, entry_time, position_size, params):
        """Open trading position"""
        shares = position_size / entry_price
        
        if direction == 'LONG':
            stop_loss = entry_price * (1 - params['stop_loss_pct'])
            take_profit = entry_price * (1 + params['take_profit_pct'])
        else:
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
        else:
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
        """Calculate performance metrics"""
        if not trades:
            return None
        
        df_trades = pd.DataFrame(trades)
        
        total_pnl = df_trades['pnl_usd'].sum()
        final_capital = self.initial_capital + total_pnl
        total_return_pct = (total_pnl / self.initial_capital) * 100
        
        num_trades = len(trades)
        winning_trades = len(df_trades[df_trades['pnl_pct'] > 0])
        win_rate = (winning_trades / num_trades) * 100
        
        avg_win = df_trades[df_trades['pnl_pct'] > 0]['pnl_pct'].mean() if winning_trades > 0 else 0
        avg_loss = df_trades[df_trades['pnl_pct'] < 0]['pnl_pct'].mean() if winning_trades < num_trades else 0
        
        risk_reward_ratio = avg_win / abs(avg_loss) if avg_loss < 0 else float('inf')
        
        exit_reasons = df_trades['exit_reason'].value_counts()
        time_exit_pct = (exit_reasons.get('TIME_EXIT', 0) / num_trades) * 100
        take_profit_pct = (exit_reasons.get('TAKE_PROFIT', 0) / num_trades) * 100
        
        return {
            'params': params,
            'final_capital': final_capital,
            'total_return_pct': total_return_pct,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'risk_reward_ratio': risk_reward_ratio,
            'time_exit_pct': time_exit_pct,
            'take_profit_pct': take_profit_pct
        }
    
    def run_optimization(self):
        """Run quick optimization"""
        print(f"\n🚀 Starting Quick Parameter Optimization")
        print("=" * 60)
        
        param_combinations = list(product(*self.param_ranges.values()))
        total_combinations = len(param_combinations)
        
        start_time = time.time()
        
        for i, combination in enumerate(param_combinations):
            params = dict(zip(self.param_ranges.keys(), combination))
            
            result = self.test_parameter_combination(params)
            
            if result:
                # Score combination
                score = (result['total_return_pct'] * 0.6 + 
                        result['risk_reward_ratio'] * 20 * 0.4)
                result['optimization_score'] = score
                self.results.append(result)
            
            # Progress every 10 combinations
            if (i + 1) % 10 == 0 or (i + 1) == total_combinations:
                elapsed = (time.time() - start_time) / 60
                progress = ((i + 1) / total_combinations) * 100
                print(f"📊 Progress: {progress:.1f}% ({i + 1}/{total_combinations}) | Time: {elapsed:.1f}min")
        
        return self.analyze_results()
    
    def analyze_results(self):
        """Analyze optimization results"""
        if not self.results:
            print("❌ No valid results found")
            return None
        
        sorted_results = sorted(self.results, key=lambda x: x['optimization_score'], reverse=True)
        
        print(f"\n🏆 TOP 5 OPTIMIZED PARAMETER COMBINATIONS")
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
        
        return sorted_results

def main():
    print("🎯 QUICK PARAMETER OPTIMIZATION - Substep 3.2")
    print("⚡ Focused search with 72 combinations (5-15 minutes)")
    print("=" * 60)
    
    optimizer = QuickParameterOptimizer(
        model_path='sequences/btc_lstm_model.h5',
        scaler_path='sequences/scaler.pkl',
        data_path='processed/BTCUSD_4h_cleaned.csv',
        initial_capital=10000
    )
    
    results = optimizer.run_optimization()
    
    if results:
        print(f"\n🎯 Quick Optimization Complete!")
        print("✅ Optimal parameters identified in minutes, not hours")
        print("✅ Ready to implement best configuration")

if __name__ == "__main__":
    main()
