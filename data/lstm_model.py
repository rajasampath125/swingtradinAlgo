import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import joblib

def build_lstm_model(input_shape, num_horizons=5):
    """
    Build LSTM model for multi-horizon BTC price prediction
    
    Parameters:
    - input_shape: (timesteps, features) = (60, 5)
    - num_horizons: Number of prediction horizons = 5
    """
    
    model = Sequential([
        # First LSTM layer with return_sequences=True to stack layers
        LSTM(64, return_sequences=True, input_shape=input_shape, name='lstm_1'),
        Dropout(0.2, name='dropout_1'),
        
        # Second LSTM layer
        LSTM(32, return_sequences=False, name='lstm_2'),
        Dropout(0.2, name='dropout_2'),
        
        # Dense layers for final processing
        Dense(16, activation='relu', name='dense_1'),
        Dropout(0.1, name='dropout_3'),
        
        # Output layer - 5 neurons for 5 prediction horizons
        Dense(num_horizons, activation='linear', name='predictions')
    ])
    
    # Compile with Adam optimizer
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',  # Mean Squared Error for regression
        metrics=['mae']  # Mean Absolute Error for monitoring
    )
    
    return model

def train_model():
    """Train the LSTM model with your BTC data"""
    
    print("Loading sequences...")
    X_train = np.load('sequences/X_train.npy')
    X_test = np.load('sequences/X_test.npy')
    y_train = np.load('sequences/y_train.npy')
    y_test = np.load('sequences/y_test.npy')
    
    print(f"Training data: {X_train.shape}")
    print(f"Training targets: {y_train.shape}")
    print(f"Test data: {X_test.shape}")
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (60, 5)
    model = build_lstm_model(input_shape)
    
    # Display model architecture
    print("\n=== MODEL ARCHITECTURE ===")
    model.summary()
    
    # Set up callbacks for better training
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001,
            verbose=1
        )
    ]
    
    print("\n=== STARTING TRAINING ===")
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=50,  # Will stop early if no improvement
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the trained model
    model.save('sequences/btc_lstm_model.h5')
    print("\nModel saved as 'sequences/btc_lstm_model.h5'")
    
    # Evaluate on test set
    print("\n=== MODEL EVALUATION ===")
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss (MSE): {test_loss:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    
    return model, history

def plot_training_history(history):
    """Plot training metrics"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot MAE
    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_title('Model MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('sequences/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Training history plot saved as 'sequences/training_history.png'")

if __name__ == "__main__":
    # Train the model
    model, history = train_model()
    
    # Plot training results
    plot_training_history(history)
    
    print("\n🎉 LSTM Model Training Complete!")
    print("Next step: Test predictions and backtesting")
