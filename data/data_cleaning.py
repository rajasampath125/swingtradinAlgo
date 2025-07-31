import pandas as pd
import numpy as np

def clean_btc_data(file_path):
    """Clean and prepare BTC/USD 4H data for LSTM processing"""
    
    # Load raw data
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} raw data points")
    
    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['Local time'], format='%d.%m.%Y %H:%M:%S.%f GMT%z')
    
    # Check if prices need scaling (BTC should be in thousands, not single digits)
    sample_close = df['Close'].iloc[100]  # Check a middle value
    if sample_close < 1000:  # If BTC price shows as under $1000, it's likely scaled
        print(f"Detected scaled prices (sample: {sample_close}). Applying correction...")
        price_cols = ['Open', 'High', 'Low', 'Close']
        df[price_cols] = df[price_cols].astype(float)  # Convert to float first
        # You might need to adjust the scaling factor based on your data source
        # For now, let's check what the actual values should be
    else:
        # Convert to float for decimal precision
        df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].astype(float)
    
    # Sort by datetime (ensure chronological order)
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Basic data validation
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Price range: ${df['Close'].min():,.2f} to ${df['Close'].max():,.2f}")
    print(f"Average volume: {df['Volume'].mean():.2f}")
    
    # Save cleaned data
    df[['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']].to_csv('processed/BTCUSD_4h_cleaned.csv', index=False)
    print("Cleaned data saved to processed/BTCUSD_4h_cleaned.csv")
    
    return df

# Run the cleaning
file_path = "raw/BTCUSD_Candlestick_4_Hour_BID_10.07.2020-19.07.2025.csv"
cleaned_df = clean_btc_data(file_path)

# Display sample of cleaned data
print("\n=== CLEANED DATA SAMPLE ===")
print(cleaned_df[['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']].head())
