import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class LSTMBacktester:
    def __init__(self, model_path, scaler_path, initial_capital=10000):
        """
        Initialize the LSTM backtesting engine
        
        Parameters:
        - model_path: Path to your trained LSTM model
        - scaler_path: Path to your scaler
        - initial_capital: Starting capital in USD
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Trading parameters (we'll optimize these later)
        self.max_change_pct = 10  # Prediction bounds
        self.position_size_pct = 0.1  # 10% of capital per trade
        self.stop_loss_pct = 0.05  # 5% stop loss
        self.take_profit_pct = 0.08  # 8% take profit
        
        # Trade tracking
        self.trades = []
        self.portfolio_values = []
        self.positions = []
        
        print(f"🚀 LSTM Backtester initialized with ${initial_capital:,.0f}")
    
    def load_model_and_scaler(self):
        """Load the trained LSTM model and scaler"""
        try:
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            self.scaler = joblib.load(self.scaler_path)
            print("✅ Model and scaler loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def prepare_backtesting_data(self, data_path):
        """Load and prepare historical data for backtesting"""
        try:
            # Load your processed BTC data
            data = pd.read_csv(data_path)
            data['datetime'] = pd.to_datetime(data['datetime'])
            
            # Sort by date to ensure chronological order
            data = data.sort_values('datetime').reset_index(drop=True)
            
            print(f"📊 Loaded {len(data)} historical data points")
            print(f"📅 Date range: {data['datetime'].min()} to {data['datetime'].max()}")
            
            # Split data: Use 80% for backtesting, keep 20% for validation
            split_idx = int(0.8 * len(data))
            
            self.backtest_data = data.iloc[:split_idx].copy()
            self.validation_data = data.iloc[split_idx:].copy()
            
            print(f"🔄 Backtest period: {len(self.backtest_data)} candles")
            print(f"📈 Validation period: {len(self.validation_data)} candles")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def generate_lstm_prediction(self, historical_sequence):
        """Generate LSTM prediction for a 60-candle sequence"""
        try:
            # Scale the input sequence
            scaled_sequence = self.scaler.transform(historical_sequence)
            X_input = scaled_sequence.reshape(1, 60, 5)
            
            # Generate prediction
            prediction_scaled = self.model.predict(X_input, verbose=0)[0]
            
            # Denormalize predictions
            predictions = {}
            horizons = ['4H', '8H', '1D', '3D', '5D']
            
            for i, horizon in enumerate(horizons):
                # Denormalize prediction
                dummy_array = np.zeros((1, 5))
                dummy_array[0, 3] = prediction_scaled[i]
                raw_prediction = self.scaler.inverse_transform(dummy_array)[0, 3]
                predictions[horizon] = raw_prediction
            
            return predictions
            
        except Exception as e:
            print(f"❌ Error generating prediction: {e}")
            return None
    
    def apply_prediction_bounds(self, predictions, current_price):
        """Apply bounds to prevent extreme predictions"""
        bounded_predictions = {}
        max_change = current_price * (self.max_change_pct / 100)
        
        for horizon, predicted_price in predictions.items():
            bounded_price = np.clip(predicted_price,
                                  current_price - max_change,
                                  current_price + max_change)
            bounded_predictions[horizon] = bounded_price
        
        return bounded_predictions
    
    def calculate_prediction_signals(self, predictions, current_price):
        """Convert LSTM predictions into trading signals"""
        signals = {}
        
        for horizon, predicted_price in predictions.items():
            change_pct = ((predicted_price - current_price) / current_price) * 100
            
            # Define signal thresholds (we'll optimize these later)
            if change_pct > 2.0:  # Bullish if >2% predicted gain
                signals[horizon] = 'BUY'
            elif change_pct < -2.0:  # Bearish if >2% predicted loss
                signals[horizon] = 'SELL'
            else:
                signals[horizon] = 'HOLD'
        
        return signals
    
    def run_backtest_step(self, index):
        """Execute one step of the backtesting process"""
        
        # Need at least 60 candles for LSTM input
        if index < 60:
            return
        
        # Get current market data
        current_candle = self.backtest_data.iloc[index]
        current_price = current_candle['Close']
        current_time = current_candle['datetime']
        
        # Get historical sequence (last 60 candles)
        historical_sequence = self.backtest_data.iloc[index-60:index][
            ['Open', 'High', 'Low', 'Close', 'Volume']
        ].values
        
        # Generate LSTM predictions
        raw_predictions = self.generate_lstm_prediction(historical_sequence)
        if raw_predictions is None:
            return
        
        # Apply bounds and calculate signals
        predictions = self.apply_prediction_bounds(raw_predictions, current_price)
        signals = self.calculate_prediction_signals(predictions, current_price)
        
        # Store prediction data for analysis
        prediction_data = {
            'timestamp': current_time,
            'current_price': current_price,
            'predictions': predictions,
            'signals': signals,
            'portfolio_value': self.current_capital
        }
        
        self.portfolio_values.append(prediction_data)
        
        # Print progress every 100 steps
        if index % 100 == 0:
            progress = (index / len(self.backtest_data)) * 100
            print(f"📊 Backtest progress: {progress:.1f}% - Date: {current_time.strftime('%Y-%m-%d')}")
    
    def run_full_backtest(self):
        """Execute the complete backtesting process"""
        
        print("\n🚀 Starting Full LSTM Backtesting...")
        print("=" * 60)
        
        # Run backtesting for each candle
        for i in range(60, len(self.backtest_data)):
            self.run_backtest_step(i)
        
        print(f"\n✅ Backtesting complete!")
        print(f"📊 Processed {len(self.portfolio_values)} prediction points")
        
        return self.portfolio_values
    
    def analyze_prediction_accuracy(self):
        """Analyze how accurate the LSTM predictions were"""
        
        print("\n📊 LSTM Prediction Accuracy Analysis")
        print("=" * 50)
        
        horizons = ['4H', '8H', '1D', '3D', '5D']
        horizon_periods = {
            '4H': 1, '8H': 2, '1D': 6, '3D': 18, '5D': 30
        }
        
        accuracy_results = {}
        
        for horizon in horizons:
            errors = []
            periods = horizon_periods[horizon]
            
            for i, prediction_data in enumerate(self.portfolio_values):
                if i + periods < len(self.portfolio_values):
                    predicted_price = prediction_data['predictions'][horizon]
                    actual_price = self.portfolio_values[i + periods]['current_price']
                    
                    error_pct = abs((predicted_price - actual_price) / actual_price) * 100
                    errors.append(error_pct)
            
            if errors:
                avg_error = np.mean(errors)
                accuracy_results[horizon] = {
                    'avg_error_pct': avg_error,
                    'accuracy_rate': 100 - avg_error,
                    'sample_size': len(errors)
                }
                
                print(f"{horizon}: {avg_error:.2f}% avg error, {len(errors)} samples")
        
        return accuracy_results

def main():
    """Main function to run the backtesting"""
    
    print("🎯 LSTM Backtesting Framework - Step 1")
    print("=" * 60)
    
    # Initialize backtester
    backtester = LSTMBacktester(
        model_path='sequences/btc_lstm_model.h5',
        scaler_path='sequences/scaler.pkl',
        initial_capital=10000
    )
    
    # Load model and data
    if not backtester.load_model_and_scaler():
        return
    
    if not backtester.prepare_backtesting_data('processed/BTCUSD_4h_cleaned.csv'):
        return
    
    # Run the backtest
    results = backtester.run_full_backtest()
    
    # Analyze prediction accuracy
    accuracy = backtester.analyze_prediction_accuracy()
    
    print("\n🎯 Step 1 Complete!")
    print("✅ Core backtesting engine working")
    print("✅ LSTM predictions generated for historical data")
    print("✅ Prediction accuracy calculated")
    print("\n🚀 Ready for Step 2: Trading Strategy Implementation")

if __name__ == "__main__":
    main()
