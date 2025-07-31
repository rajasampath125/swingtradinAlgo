# Create prediction_bounds.py
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import yfinance as yf

def bounded_real_time_prediction(max_change_pct=10):
    """Generate bounded predictions to prevent extreme forecasts"""
    
    print(f"🚀 BTC Real-Time Prediction (Bounded ±{max_change_pct}%)")
    print("=" * 50)
    
    # Load model and scaler
    model = tf.keras.models.load_model('sequences/btc_lstm_model.h5', compile=False)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    scaler = joblib.load('sequences/scaler.pkl')
    
    # Get recent data (fallback to your collected data if yfinance fails)
    try:
        recent_data = yf.download("BTC-USD", period="15d", interval="4h")
        recent_ohlcv = recent_data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(60)
        current_price = float(recent_data['Close'].iloc[-1])
        print(f"✅ Using live data. Current price: ${current_price:,.0f}")
    except:
        # Fallback to your collected data
        btc_data = pd.read_csv("raw/BTCUSD_Candlestick_4_Hour_BID_10.07.2020-19.07.2025.csv")
        recent_ohlcv = btc_data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(60)
        current_price = float(btc_data['Close'].iloc[-1])
        print(f"⚠️  Using collected data. Latest price: ${current_price:,.0f}")
    
    # Make predictions
    scaled_data = scaler.transform(recent_ohlcv.values)
    X_current = scaled_data.reshape(1, 60, 5)
    predictions_scaled = model.predict(X_current, verbose=0)[0]
    
    # Apply bounds to predictions
    horizons = ['4H', '8H', '1D', '3D', '5D']
    max_change = current_price * (max_change_pct / 100)
    
    print(f"📊 Bounded LSTM Predictions (±{max_change_pct}% max):")
    print("-" * 50)
    
    for i, horizon in enumerate(horizons):
        # Denormalize prediction
        dummy_array = np.zeros((1, 5))
        dummy_array[0, 3] = predictions_scaled[i]
        raw_prediction = scaler.inverse_transform(dummy_array)[0, 3]
        
        # Apply bounds
        bounded_prediction = np.clip(raw_prediction, 
                                   current_price - max_change, 
                                   current_price + max_change)
        
        # Calculate changes
        raw_change = ((raw_prediction - current_price) / current_price) * 100
        bounded_change = ((bounded_prediction - current_price) / current_price) * 100
        
        # Show both raw and bounded
        direction = "📈" if bounded_change > 0 else "📉" if bounded_change < 0 else "➡️"
        
        print(f"{horizon}: {direction} ${bounded_prediction:,.0f} ({bounded_change:+.2f}%)")
        if abs(raw_change) > max_change_pct:
            print(f"      [Raw prediction: ${raw_prediction:,.0f} ({raw_change:+.1f}%) - BOUNDED]")
    
    return bounded_prediction

if __name__ == "__main__":
    bounded_real_time_prediction()
