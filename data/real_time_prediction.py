import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import yfinance as yf
from datetime import datetime, timedelta

def get_recent_btc_data():
    """Fetch the most recent 60 periods of 4H BTC data"""
    
    # Fetch recent data (need 60 4H candles for input sequence)
    btc = yf.download("BTC-USD", period="15d", interval="4h")
    
    if len(btc) < 60:
        print("Warning: Insufficient recent data")
        return None
    
    # Get last 60 candles for prediction
    recent_data = btc[['Open', 'High', 'Low', 'Close', 'Volume']].tail(60)
    return recent_data

def make_real_time_prediction():
    """Generate predictions using your trained model"""
    
    print(f"🚀 BTC Real-Time Prediction Analysis")
    print(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Current BTC Price: $117,690")
    
    # Load your trained model and scaler
    model = tf.keras.models.load_model('sequences/btc_lstm_model.h5', compile=False)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    scaler = joblib.load('sequences/scaler.pkl')
    
    # Get recent data
    recent_data = get_recent_btc_data()
    if recent_data is None:
        return
    
    # Scale the data using your trained scaler
    scaled_data = scaler.transform(recent_data.values)
    
    # Create input sequence (1, 60, 5)
    X_current = scaled_data.reshape(1, 60, 5)
    
    # Generate predictions
    predictions_scaled = model.predict(X_current, verbose=0)[0]
    
    # Convert back to actual prices
    horizons = ['4H', '8H', '1D', '3D', '5D']
    current_price = 117690
    
    print(f"\n📊 LSTM Predictions from $117,690:")
    print("=" * 50)
    
    for i, horizon in enumerate(horizons):
        # Denormalize prediction
        dummy_array = np.zeros((1, 5))
        dummy_array[0, 3] = predictions_scaled[i]  # Close price index
        predicted_price = scaler.inverse_transform(dummy_array)[0, 3]
        
        # Calculate change
        price_change = predicted_price - current_price
        change_pct = (price_change / current_price) * 100
        
        direction = "📈" if price_change > 0 else "📉"
        
        print(f"{horizon}: {direction} ${predicted_price:,.0f} ({change_pct:+.2f}%)")
    
    return predictions_scaled

if __name__ == "__main__":
    make_real_time_prediction()
