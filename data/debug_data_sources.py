import yfinance as yf
import pandas as pd
import numpy as np
import joblib

# Compare data sources
def compare_data_sources():
    # Your historical data (last few rows)
    historical = pd.read_csv("raw/BTCUSD_Candlestick_4_Hour_BID_10.07.2020-19.07.2025.csv")
    print("Historical data (last 5 rows):")
    print(historical[['Local time', 'Close']].tail())
    
    # YFinance recent data
    yf_data = yf.download("BTC-USD", period="5d", interval="4h")
    print("\nYFinance data (last 5 rows):")
    print(yf_data[['Close']].tail())
    
    # Check scaling ranges
    scaler = joblib.load('sequences/scaler.pkl')
    print(f"\nScaler training range:")
    print(f"Min values: {scaler.data_min_}")
    print(f"Max values: {scaler.data_max_}")
    
    # Test scaling of current price
    current_price = 117690
    test_array = np.array([[current_price, current_price, current_price, current_price, 1000]])
    scaled_test = scaler.transform(test_array)
    print(f"\nCurrent BTC ${current_price} scales to: {scaled_test[0][3]:.4f}")

compare_data_sources()
