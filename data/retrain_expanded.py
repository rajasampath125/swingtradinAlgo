# Create retrain_expanded.py
def retrain_with_expanded_range():
    """Retrain model with artificially expanded price range"""
    
    # Load your existing data
    df = pd.read_csv('processed/BTCUSD_4h_cleaned.csv')
    
    # Create artificial high-price samples (data augmentation)
    high_price_samples = df.tail(1000).copy()  # Last 1000 samples
    
    # Scale prices up by 10-20% to expand training range
    price_cols = ['Open', 'High', 'Low', 'Close']
    high_price_samples[price_cols] = high_price_samples[price_cols] * 1.15
    
    # Combine with original data
    expanded_df = pd.concat([df, high_price_samples], ignore_index=True)
    
    # Retrain with expanded range
    # ... (use your existing training pipeline)
    
    print("Model retrained with expanded price range")
