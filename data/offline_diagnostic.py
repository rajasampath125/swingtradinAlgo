import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

def analyze_your_collected_data():
    """Analyze your collected data without external APIs"""
    print("🔍 DIAGNOSTIC: Your Collected Data Analysis")
    print("=" * 50)
    
    try:
        # Load your BTC 4H data
        btc_data = pd.read_csv("raw/BTCUSD_Candlestick_4_Hour_BID_10.07.2020-19.07.2025.csv")
        print(f"✅ Loaded your BTC data: {len(btc_data)} candles")
        
        # Analyze recent patterns (last 60 candles = 10 days)
        recent_data = btc_data.tail(60)
        recent_prices = recent_data['Close'].values
        
        print(f"Data date range: {btc_data.iloc[0, 0]} to {btc_data.iloc[-1, 0]}")
        print(f"Recent price range: ${recent_prices.min():,.0f} - ${recent_prices.max():,.0f}")
        print(f"Latest price in your data: ${recent_prices[-1]:,.0f}")
        
        # Calculate volatility
        recent_volatility = np.std(recent_prices) / np.mean(recent_prices) * 100
        print(f"Recent volatility: {recent_volatility:.2f}%")
        
        return btc_data, recent_data, recent_prices
        
    except Exception as e:
        print(f"❌ Error loading your data: {e}")
        return None, None, None

def check_scaler_compatibility():
    """Check if your scaler ranges make sense"""
    print("\n🔍 DIAGNOSTIC: Scaler Analysis")
    print("=" * 50)
    
    try:
        scaler = joblib.load('sequences/scaler.pkl')
        
        feature_names = ['Open', 'High', 'Low', 'Close', 'Volume']
        print("Scaler training ranges:")
        for i, name in enumerate(feature_names):
            print(f"  {name}: ${scaler.data_min_[i]:,.0f} - ${scaler.data_max_[i]:,.0f}")
        
        # Test current BTC price scaling
        current_market_price = 117690  # Current market price you mentioned
        test_array = np.array([[current_market_price] * 4 + [1000]])  # OHLC + Volume
        scaled_test = scaler.transform(test_array)
        
        print(f"\nCurrent market BTC ${current_market_price:,.0f}:")
        print(f"  Scales to: {scaled_test[0][3]:.4f}")
        
        # Check if we're in dangerous territory
        if scaled_test[0][3] > 0.95:
            print("🚨 CRITICAL: Current price is at scaler's upper limit!")
            print("   This explains the extreme predictions.")
        elif scaled_test[0][3] > 0.85:
            print("⚠️  WARNING: Current price is very high in scaler range.")
        else:
            print("✅ Current price is in safe scaling range.")
            
        return scaler, scaled_test[0][3]
        
    except Exception as e:
        print(f"❌ Error loading scaler: {e}")
        return None, None

def simulate_prediction_at_different_prices():
    """Test what happens when we simulate predictions at various price levels"""
    print("\n🔍 DIAGNOSTIC: Prediction Sensitivity Analysis")
    print("=" * 50)
    
    try:
        scaler = joblib.load('sequences/scaler.pkl')
        
        # Test how scaling affects different price levels
        test_prices = [50000, 75000, 100000, 115000, 117690, 120000, 125000]
        
        print("Price scaling sensitivity:")
        for price in test_prices:
            test_array = np.array([[price] * 4 + [1000]])
            scaled_value = scaler.transform(test_array)[0][3]
            
            if scaled_value > 1.0:
                status = "🚨 BEYOND TRAINING RANGE"
            elif scaled_value > 0.95:
                status = "⚠️  DANGEROUS ZONE"
            elif scaled_value > 0.8:
                status = "📊 HIGH RANGE"
            else:
                status = "✅ SAFE RANGE"
                
            print(f"  ${price:,}: {scaled_value:.4f} - {status}")
            
    except Exception as e:
        print(f"❌ Error in sensitivity analysis: {e}")

def analyze_prediction_extremes():
    """Analyze why predictions are so extreme"""
    print("\n🔍 DIAGNOSTIC: Extreme Prediction Analysis")
    print("=" * 50)
    
    current_price = 117690
    predictions = [96262, 92464, 101435, 85755, 84610]
    horizons = ['4H', '8H', '1D', '3D', '5D']
    
    print("Your model's predictions analysis:")
    for i, (horizon, pred) in enumerate(zip(horizons, predictions)):
        change = pred - current_price
        change_pct = (change / current_price) * 100
        
        print(f"  {horizon}: ${pred:,} ({change_pct:+.1f}%)")
    
    # Historical context
    print(f"\n📊 Historical BTC crash analysis:")
    print(f"  • COVID crash (Mar 2020): -50% in days")
    print(f"  • FTX collapse (Nov 2022): -25% in week")
    print(f"  • Your predictions: -18% to -28%")
    print(f"  • Normal swing moves: ±2-5%")

def main_diagnostic():
    """Run offline diagnostic without external data"""
    print("🚀 OFFLINE LSTM DIAGNOSTIC (No External APIs)")
    print("=" * 60)
    
    # Analyze your collected data
    btc_data, recent_data, recent_prices = analyze_your_collected_data()
    
    # Check scaler compatibility
    scaler, current_scaled = check_scaler_compatibility()
    
    # Test prediction sensitivity
    simulate_prediction_at_different_prices()
    
    # Analyze the extreme predictions
    analyze_prediction_extremes()
    
    # Conclusions and recommendations
    print("\n🎯 DIAGNOSTIC CONCLUSIONS")
    print("=" * 60)
    
    if current_scaled and current_scaled > 0.95:
        print("🚨 ROOT CAUSE IDENTIFIED:")
        print("   Current BTC price ($117,690) is at the upper limit of your scaler's")
        print("   training range. This causes extreme predictions.")
        print("")
        print("💡 SOLUTIONS:")
        print("   1. IMMEDIATE: Use prediction bounds (±10% max change)")
        print("   2. SHORT-TERM: Retrain with expanded price range")
        print("   3. LONG-TERM: Implement rolling window training")
        
    elif current_scaled and current_scaled > 0.8:
        print("⚠️  LIKELY CAUSE:")
        print("   Current price is in upper 20% of training range.")
        print("   Model may be extrapolating beyond reliable patterns.")
        
    else:
        print("❓ UNCLEAR CAUSE:")
        print("   Technical scaling seems OK. Predictions might indicate:")
        print("   • Genuine bearish pattern detection")
        print("   • Model overfitting to specific patterns")
        print("   • Hidden preprocessing issues")

if __name__ == "__main__":
    main_diagnostic()
