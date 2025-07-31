import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

def manual_prediction_check():
    """Check if recent patterns match historical training patterns"""
    print("🔍 DIAGNOSTIC: Recent Market Data Analysis")
    print("=" * 50)
    
    # Load recent actual BTC data
    recent_data = yf.download("BTC-USD", period="10d", interval="4h")
    
    # Check if recent patterns match historical training patterns
    recent_prices = recent_data['Close'].values
    recent_volatility = np.std(recent_prices) / np.mean(recent_prices) * 100
    
    print(f"Recent BTC price range: ${recent_prices.min():,.0f} - ${recent_prices.max():,.0f}")
    print(f"Recent volatility: {recent_volatility:.2f}%")
    print(f"Recent average price: ${np.mean(recent_prices):,.0f}")
    
    return recent_data, recent_prices, recent_volatility

def compare_data_sources():
    """Compare your training data with yfinance data"""
    print("\n🔍 DIAGNOSTIC: Data Source Comparison")
    print("=" * 50)
    
    # Your historical data (last few rows)
    try:
        historical = pd.read_csv("raw/BTCUSD_Candlestick_4_Hour_BID_10.07.2020-19.07.2025.csv")
        print("✅ Historical data loaded successfully")
        print("Historical data (last 5 rows):")
        print(historical[['Local time', 'Close']].tail())
        
        historical_range = f"${historical['Close'].min():,.0f} - ${historical['Close'].max():,.0f}"
        print(f"Historical price range: {historical_range}")
        
    except Exception as e:
        print(f"❌ Error loading historical data: {e}")
        return None
    
    # YFinance recent data
    yf_data = yf.download("BTC-USD", period="5d", interval="4h")
    print(f"\n✅ YFinance data loaded successfully")
    print("YFinance data (last 5 rows):")
    print(yf_data[['Close']].tail())
    
    return historical, yf_data

def check_scaler_ranges():
    """Check if current prices are within your scaler's training range"""
    print("\n🔍 DIAGNOSTIC: Scaler Range Analysis")
    print("=" * 50)
    
    try:
        scaler = joblib.load('sequences/scaler.pkl')
        print("✅ Scaler loaded successfully")
        
        print(f"Scaler training ranges:")
        feature_names = ['Open', 'High', 'Low', 'Close', 'Volume']
        for i, name in enumerate(feature_names):
            print(f"  {name}: ${scaler.data_min_[i]:,.2f} - ${scaler.data_max_[i]:,.2f}")
        
        # Test scaling of current price
        current_price = 117690
        test_array = np.array([[current_price, current_price, current_price, current_price, 1000]])
        scaled_test = scaler.transform(test_array)
        
        print(f"\n📊 Current BTC ${current_price:,.0f} scales to: {scaled_test[0][3]:.4f}")
        
        # Check if we're near the upper bound (dangerous for predictions)
        if scaled_test[0][3] > 0.95:
            print("⚠️  WARNING: Current price is very close to scaler's maximum!")
            print("   This can cause unreliable predictions.")
        elif scaled_test[0][3] > 0.8:
            print("⚠️  CAUTION: Current price is in upper 20% of training range.")
        else:
            print("✅ Current price is within safe scaling range.")
            
        return scaler, scaled_test[0][3]
        
    except Exception as e:
        print(f"❌ Error loading scaler: {e}")
        return None, None

def analyze_prediction_reasonableness():
    """Analyze if the extreme predictions make sense"""
    print("\n🔍 DIAGNOSTIC: Prediction Reasonableness Check")
    print("=" * 50)
    
    current_price = 117690
    predictions = {
        '4H': 96262,
        '8H': 92464, 
        '1D': 101435,
        '3D': 85755,
        '5D': 84610
    }
    
    print("Analyzing your model's predictions:")
    for horizon, pred_price in predictions.items():
        change_pct = ((pred_price - current_price) / current_price) * 100
        
        # Historical context for such moves
        if abs(change_pct) > 20:
            severity = "🚨 EXTREME"
        elif abs(change_pct) > 10:
            severity = "⚠️  LARGE"
        elif abs(change_pct) > 5:
            severity = "📊 MODERATE"
        else:
            severity = "✅ NORMAL"
            
        print(f"  {horizon}: {change_pct:+.1f}% - {severity}")
    
    print(f"\n📈 Historical context:")
    print(f"  • Normal BTC 4H moves: ±2-5%")
    print(f"  • Large BTC daily moves: ±5-10%") 
    print(f"  • Crash scenarios: >15% drops")
    print(f"  • Your predictions: 18-28% drops")

def run_complete_diagnostic():
    """Run all diagnostic checks"""
    print("🚀 LSTM PREDICTION DIAGNOSTIC SUITE")
    print("=" * 60)
    
    # Check 1: Recent market data
    recent_data, recent_prices, volatility = manual_prediction_check()
    
    # Check 2: Data source comparison  
    historical, yf_data = compare_data_sources()
    
    # Check 3: Scaler analysis
    scaler, current_scaled_value = check_scaler_ranges()
    
    # Check 4: Prediction reasonableness
    analyze_prediction_reasonableness()
    
    # Summary and recommendations
    print("\n🎯 DIAGNOSTIC SUMMARY & RECOMMENDATIONS")
    print("=" * 60)
    
    issues_found = []
    
    if current_scaled_value and current_scaled_value > 0.9:
        issues_found.append("Current price near scaler maximum")
    
    if volatility > 8:
        issues_found.append(f"High recent volatility ({volatility:.1f}%)")
    
    if len(issues_found) > 0:
        print("❌ Issues identified:")
        for issue in issues_found:
            print(f"   • {issue}")
            
        print(f"\n💡 RECOMMENDED ACTIONS:")
        if "scaler maximum" in str(issues_found):
            print("   1. Retrain model with recent data to expand scaler range")
            print("   2. Apply prediction bounds (±15% max change)")
            print("   3. Use rolling window training (last 2-3 years)")
        
        if "volatility" in str(issues_found):
            print("   4. Wait for market to stabilize before using predictions")
            print("   5. Reduce position sizes during high volatility")
            
    else:
        print("✅ No obvious technical issues found")
        print("💭 Extreme predictions might indicate:")
        print("   • Genuine bearish technical pattern detected")
        print("   • Model overfitting to specific historical patterns")
        print("   • Hidden data preprocessing issues")

if __name__ == "__main__":
    run_complete_diagnostic()
