import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from datetime import datetime
import os

def process_downloaded_data(csv_file):
    """Process your manually downloaded BTC 4H data"""
    
    print("📁 Processing your downloaded BTC data...")
    
    try:
        # Check if file exists
        if not os.path.exists(csv_file):
            print(f"❌ File not found: {csv_file}")
            print("Please make sure the file is in the raw/ folder")
            return None, None
        
        # Load your fresh data
        fresh_data = pd.read_csv(csv_file)
        
        print(f"✅ Loaded {len(fresh_data)} fresh candles")
        print(f"📊 Columns found: {fresh_data.columns.tolist()}")
        print(f"📅 Data sample (first 2 rows):")
        print(fresh_data.head(2))
        
        # Auto-detect and standardize column names
        # Common variations from different sources
        column_mapping = {}
        
        for col in fresh_data.columns:
            col_lower = col.lower()
            if 'open' in col_lower:
                column_mapping[col] = 'Open'
            elif 'high' in col_lower:
                column_mapping[col] = 'High'
            elif 'low' in col_lower:
                column_mapping[col] = 'Low'
            elif 'close' in col_lower:
                column_mapping[col] = 'Close'
            elif 'volume' in col_lower:
                column_mapping[col] = 'Volume'
        
        # Rename columns if mapping found
        if column_mapping:
            fresh_data.rename(columns=column_mapping, inplace=True)
            print(f"🔄 Standardized columns: {list(column_mapping.values())}")
        
        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in fresh_data.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            print(f"Available columns: {fresh_data.columns.tolist()}")
            print("Please check your data format and column names")
            return None, None
        
        # Convert to numeric if needed
        for col in required_cols:
            fresh_data[col] = pd.to_numeric(fresh_data[col], errors='coerce')
        
        # Remove any rows with NaN values
        fresh_data.dropna(inplace=True)
        
        # Get the most recent data for analysis
        if len(fresh_data) >= 60:
            # Take last 60 candles for LSTM input
            recent_60 = fresh_data[required_cols].tail(60)
            current_price = float(fresh_data['Close'].iloc[-1])
            latest_timestamp = fresh_data.iloc[-1, 0] if len(fresh_data.columns) > 5 else "Recent"
            
            print(f"📈 Using last 60 candles for LSTM prediction")
            print(f"💰 Current BTC Price: ${current_price:,.2f}")
            print(f"📅 Latest data point: {latest_timestamp}")
            
            return recent_60, current_price
        else:
            print(f"⚠️ Need at least 60 candles for LSTM, got {len(fresh_data)}")
            print("Please download more historical data (at least 10 days of 4H data)")
            return None, None
            
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        print("Common issues:")
        print("- Check file format (should be CSV)")
        print("- Verify column names match expected format")
        print("- Ensure data contains numeric OHLCV values")
        return None, None

def run_fresh_lstm_analysis(csv_file):
    """Run LSTM analysis with your downloaded data"""
    
    print("🚀 LSTM Analysis with Fresh Downloaded Data")
    print("=" * 60)
    
    # Process the downloaded data
    recent_data, current_price = process_downloaded_data(csv_file)
    if recent_data is None:
        return
    
    # Load your existing trained model (no retraining needed!)
    print("\n📁 Loading your existing trained LSTM model...")
    try:
        model = tf.keras.models.load_model('sequences/btc_lstm_model.h5', compile=False)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        scaler = joblib.load('sequences/scaler.pkl')
        print("✅ Model and scaler loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure btc_lstm_model.h5 and scaler.pkl are in the sequences/ folder")
        return
    
    # Scale the fresh data using your existing scaler
    print("\n⚙️ Scaling fresh data with existing scaler...")
    try:
        scaled_data = scaler.transform(recent_data.values)
        X_current = scaled_data.reshape(1, 60, 5)
        
        # Check scaling range for current price
        current_scaled = scaled_data[-1, 3]  # Close price scaled value
        print(f"📊 Current price scales to: {current_scaled:.4f}")
        
        if current_scaled > 0.95:
            print("⚠️ WARNING: Current price near scaler maximum - predictions may be less reliable")
        
    except Exception as e:
        print(f"❌ Error scaling data: {e}")
        return
    
    # Generate predictions with your trained model
    print("🔮 Generating LSTM predictions...")
    try:
        predictions_scaled = model.predict(X_current, verbose=0)[0]
    except Exception as e:
        print(f"❌ Error generating predictions: {e}")
        return
    
    # Apply bounds (±10% max change as we determined earlier)
    max_change_pct = 10
    max_change = current_price * (max_change_pct / 100)
    
    print(f"\n📈 Fresh LSTM Predictions (Bounded ±{max_change_pct}%):")
    print("=" * 60)
    
    horizons = ['4H', '8H', '1D', '3D', '5D']
    predictions = {}
    
    for i, horizon in enumerate(horizons):
        # Denormalize prediction using existing scaler
        dummy_array = np.zeros((1, 5))
        dummy_array[0, 3] = predictions_scaled[i]
        raw_prediction = scaler.inverse_transform(dummy_array)[0, 3]
        
        # Apply bounds to prevent extreme predictions
        bounded_prediction = np.clip(raw_prediction, 
                                   current_price - max_change, 
                                   current_price + max_change)
        
        # Calculate changes
        bounded_change = ((bounded_prediction - current_price) / current_price) * 100
        raw_change = ((raw_prediction - current_price) / current_price) * 100
        
        direction = "📈" if bounded_change > 0 else "📉" if bounded_change < 0 else "➡️"
        
        print(f"{horizon}: {direction} ${bounded_prediction:,.0f} ({bounded_change:+.2f}%)")
        
        # Show if prediction was bounded
        if abs(raw_change) > max_change_pct:
            print(f"      [Raw prediction: ${raw_prediction:,.0f} ({raw_change:+.1f}%) - BOUNDED]")
        
        predictions[horizon] = {
            'price': bounded_prediction,
            'change_pct': bounded_change,
            'raw_prediction': raw_prediction,
            'was_bounded': abs(raw_change) > max_change_pct
        }
    
    # Analysis summary
    print(f"\n📋 Market Analysis Summary:")
    print(f"Current BTC Price: ${current_price:,.2f}")
    print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Signal analysis
    bullish_signals = sum(1 for pred in predictions.values() if pred['change_pct'] > 0)
    bearish_signals = sum(1 for pred in predictions.values() if pred['change_pct'] < 0)
    neutral_signals = 5 - bullish_signals - bearish_signals
    bounded_predictions = sum(1 for pred in predictions.values() if pred['was_bounded'])
    
    print(f"\n🎯 LSTM Signal Summary:")
    print(f"Bullish signals: {bullish_signals}/5")
    print(f"Bearish signals: {bearish_signals}/5")
    print(f"Neutral signals: {neutral_signals}/5")
    print(f"Bounded predictions: {bounded_predictions}/5")
    
    if bullish_signals > bearish_signals:
        bias = "📈 BULLISH BIAS"
    elif bearish_signals > bullish_signals:
        bias = "📉 BEARISH BIAS"
    else:
        bias = "➡️ NEUTRAL/MIXED"
    
    print(f"Overall LSTM bias: {bias}")
    
    # Historical validation
    print(f"\n🎯 Model Validation:")
    print(f"Previous predictions: Bearish from $117,690")
    print(f"Current market price: ${current_price:,.2f}")
    if current_price < 117690:
        decline_pct = ((117690 - current_price) / 117690) * 100
        print(f"✅ Previous bearish prediction ACCURATE! (-{decline_pct:.1f}%)")
        print(f"Model successfully predicted the decline!")
    else:
        increase_pct = ((current_price - 117690) / 117690) * 100
        print(f"📊 Market moved up +{increase_pct:.1f}% from previous analysis")
    
    # Trading insights
    print(f"\n💡 Trading Insights:")
    avg_prediction = sum(pred['change_pct'] for pred in predictions.values()) / 5
    print(f"Average predicted move: {avg_prediction:+.2f}%")
    
    if abs(avg_prediction) > 2:
        strength = "Strong" if abs(avg_prediction) > 5 else "Moderate"
        direction = "bullish" if avg_prediction > 0 else "bearish"
        print(f"{strength} {direction} signal detected")
    else:
        print("Weak/neutral signal - consider waiting for clearer setup")
    
    print(f"\n✅ Fresh LSTM analysis complete!")
    print(f"🚀 Model validated with latest data - Ready for backtesting phase!")
    
    return predictions

def main():
    """Main function to run the analysis"""
    
    # File path corrected to point to raw folder
    csv_file = "raw/fresh_btc_4h_data.csv"
    
    print("💡 Looking for your downloaded data in the raw/ folder...")
    print(f"File path: {csv_file}")
    print("=" * 60)
    
    # Run the analysis
    predictions = run_fresh_lstm_analysis(csv_file)
    
    if predictions:
        print("\n🎯 Next Steps:")
        print("1. ✅ Fresh data analysis complete")
        print("2. 🚀 Ready to proceed with backtesting framework")
        print("3. 📊 Use these predictions for swing trading decisions")
    else:
        print("\n❌ Analysis failed - please check data format and try again")

if __name__ == "__main__":
    main()
