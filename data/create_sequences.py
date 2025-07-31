import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_lstm_sequences(df, sequence_length=60, horizons=[1, 2, 6, 18, 30]):
    """
    Create LSTM sequences with multi-horizon targets
    
    Parameters:
    - sequence_length: Number of past 4H bars to use as input (60 = 10 days)
    - horizons: Prediction horizons in 4H periods
      [1, 2, 6, 18, 30] = [4h, 8h, 1d, 3d, 5d] ahead
    """
    
    print(f"Creating sequences with {sequence_length} input bars")
    print(f"Prediction horizons: {horizons} periods ahead")
    
    # Focus on OHLCV features
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = df[feature_cols].values
    
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences
    X, y = [], []
    max_horizon = max(horizons)
    
    for i in range(sequence_length, len(scaled_data) - max_horizon):
        # Input sequence (past 60 bars of OHLCV)
        X.append(scaled_data[i-sequence_length:i])
        
        # Multi-horizon targets (future close prices)
        targets = []
        for h in horizons:
            future_close = scaled_data[i + h - 1][3]  # Close price is index 3
            targets.append(future_close)
        y.append(targets)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Created {len(X)} sequences")
    print(f"Input shape: {X.shape}")  # Should be (samples, sequence_length, features)
    print(f"Output shape: {y.shape}")  # Should be (samples, num_horizons)
    
    return X, y, scaler

# Load cleaned data
df = pd.read_csv('processed/BTCUSD_4h_cleaned.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

print(f"Loaded {len(df)} cleaned data points")
print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

# Create sequences
X, y, scaler = create_lstm_sequences(df)

# Split data chronologically (80% train, 20% test)
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\nData splits:")
print(f"Training: {len(X_train)} sequences")
print(f"Testing: {len(X_test)} sequences")

# Save sequences for LSTM training
np.save('sequences/X_train.npy', X_train)
np.save('sequences/X_test.npy', X_test)
np.save('sequences/y_train.npy', y_train)
np.save('sequences/y_test.npy', y_test)

# Save scaler for future price denormalization
import joblib
joblib.dump(scaler, 'sequences/scaler.pkl')

print("\nSequences saved to sequences/ folder")
print("Ready for LSTM model training!")
