import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt

def load_model_and_data():
    """Load trained model and test data with compatibility fix"""
    
    # Load model without compilation to avoid deserialization issues
    model = tf.keras.models.load_model('sequences/btc_lstm_model.h5', compile=False)
    
    # Recompile the model with the same settings used during training
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # Load test data and scaler
    X_test = np.load('sequences/X_test.npy')
    y_test = np.load('sequences/y_test.npy')
    scaler = joblib.load('sequences/scaler.pkl')
    
    print("✅ Model loaded and recompiled successfully")
    
    return model, X_test, y_test, scaler

# Keep the rest of the functions exactly the same...
def test_recent_predictions(model, X_test, y_test, scaler, num_samples=10):
    """Test model predictions on recent data"""
    
    # Get predictions for last few samples (most recent data)
    recent_X = X_test[-num_samples:]
    recent_y_true = y_test[-num_samples:]
    
    # Make predictions
    predictions = model.predict(recent_X, verbose=0)  # Added verbose=0 to reduce output
    
    print("=== RECENT PREDICTION ANALYSIS ===")
    print(f"Testing on {num_samples} most recent sequences")
    
    # Convert back to actual prices
    horizons = ['4H', '8H', '1D', '3D', '5D']
    
    for i in range(min(5, num_samples)):  # Show first 5 samples
        print(f"\n--- Sample {i+1} (Recent BTC Data) ---")
        
        for h, horizon in enumerate(horizons):
            # Denormalize predictions and actual values
            pred_scaled = predictions[i][h]
            true_scaled = recent_y_true[i][h]
            
            # Create dummy array for inverse transform
            dummy_pred = np.zeros((1, 5))
            dummy_true = np.zeros((1, 5))
            dummy_pred[0, 3] = pred_scaled  # Close price is index 3
            dummy_true[0, 3] = true_scaled
            
            pred_price = scaler.inverse_transform(dummy_pred)[0, 3]
            true_price = scaler.inverse_transform(dummy_true)[0, 3]
            
            error = abs(pred_price - true_price)
            error_pct = (error / true_price) * 100
            
            print(f"  {horizon}: Predicted ${pred_price:,.0f} | Actual ${true_price:,.0f} | Error: {error_pct:.2f}%")
    
    return predictions, recent_y_true

def plot_prediction_comparison(predictions, y_true, scaler):
    """Plot prediction vs actual for all horizons"""
    
    horizons = ['4H', '8H', '1D', '3D', '5D']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for h in range(5):
        # Denormalize data for plotting
        pred_prices = []
        true_prices = []
        
        for i in range(len(predictions)):
            dummy_pred = np.zeros((1, 5))
            dummy_true = np.zeros((1, 5))
            dummy_pred[0, 3] = predictions[i][h]
            dummy_true[0, 3] = y_true[i][h]
            
            pred_price = scaler.inverse_transform(dummy_pred)[0, 3]
            true_price = scaler.inverse_transform(dummy_true)[0, 3]
            
            pred_prices.append(pred_price)
            true_prices.append(true_price)
        
        # Plot
        axes[h].scatter(true_prices, pred_prices, alpha=0.6)
        axes[h].plot([min(true_prices), max(true_prices)], 
                    [min(true_prices), max(true_prices)], 'r--', label='Perfect Prediction')
        axes[h].set_xlabel('Actual BTC Price ($)')
        axes[h].set_ylabel('Predicted BTC Price ($)')
        axes[h].set_title(f'{horizons[h]} Ahead Predictions')
        axes[h].legend()
        axes[h].grid(True, alpha=0.3)
    
    # Remove empty subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig('sequences/prediction_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Prediction analysis plot saved as 'sequences/prediction_analysis.png'")

if __name__ == "__main__":
    # Load everything
    model, X_test, y_test, scaler = load_model_and_data()
    
    print(f"Model loaded successfully")
    print(f"Test data shape: {X_test.shape}")
    
    # Test recent predictions
    predictions, y_true = test_recent_predictions(model, X_test, y_test, scaler)
    
    # Plot comparison
    plot_prediction_comparison(predictions, y_true, scaler)
    
    print("\n🎯 Prediction testing complete!")
    print("Review the actual vs predicted prices above")
    print("Next: Implement backtesting for trading strategy evaluation")
