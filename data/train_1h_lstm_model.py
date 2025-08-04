import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class HourlyLSTMTrainer:
    def __init__(self, data_path, model_save_path='models/btc_1h_lstm_model.h5', scaler_save_path='models/btc_1h_scaler.pkl'):
        """
        Train 1H LSTM model to enhance your successful 4H system
        Multi-horizon predictions for precise entry/exit timing
        """
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.scaler_save_path = scaler_save_path
        
        # Model parameters (optimized for 1H timeframe)
        self.sequence_length = 60  # 60 hours lookback
        self.batch_size = 32
        self.epochs = 100
        self.validation_split = 0.2
        
        # Multi-horizon prediction targets (aligned with your 4H system)
        self.horizons = {
            '1H': 1,    # 1 hour ahead
            '4H': 4,    # 4 hours ahead (align with your 4H system)
            '8H': 8,    # 8 hours ahead  
            '1D': 24,   # 1 day ahead
            '5D': 120   # 5 days ahead
        }
        
        print(f"🕐 1H LSTM Trainer Initialized")
        print(f"📊 Multi-horizon predictions: {list(self.horizons.keys())}")
        print(f"🎯 Building on your successful 82.57% return 4H system")
    
    def load_and_prepare_data(self):
        """Load processed 1H data and prepare for LSTM training"""
        try:
            # Load your processed 1H dataset
            print(f"\n📊 Loading processed 1H data...")
            data = pd.read_csv(self.data_path)
            data['datetime'] = pd.to_datetime(data['datetime'])
            
            print(f"✅ Loaded {len(data):,} 1H candles")
            print(f"📅 Date range: {data['datetime'].min()} to {data['datetime'].max()}")
            
            # Feature columns (matching your successful 4H system approach)
            self.feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Verify all required columns exist
            missing_cols = [col for col in self.feature_columns if col not in data.columns]
            if missing_cols:
                print(f"❌ Missing columns: {missing_cols}")
                return None
            
            self.raw_data = data.copy()
            print(f"✅ Data structure validated")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def create_sequences_and_targets(self):
        """Create LSTM sequences with multi-horizon targets"""
        print(f"\n🔧 Creating LSTM sequences and multi-horizon targets...")
        
        try:
            # Prepare feature data
            feature_data = self.raw_data[self.feature_columns].values
            
            # Normalize features using MinMaxScaler (same as your 4H system)
            self.scaler = MinMaxScaler()
            feature_data_scaled = self.scaler.fit_transform(feature_data)
            
            # Create sequences and targets
            X, y = [], []
            max_horizon = max(self.horizons.values())
            
            for i in range(self.sequence_length, len(feature_data_scaled) - max_horizon):
                # Input sequence (60 hours of OHLCV data)
                sequence = feature_data_scaled[i-self.sequence_length:i]
                X.append(sequence)
                
                # Multi-horizon targets (predict future close prices)
                current_close = feature_data_scaled[i, 3]  # Current close price (normalized)
                
                horizon_targets = []
                for horizon_name, horizon_periods in self.horizons.items():
                    future_close = feature_data_scaled[i + horizon_periods - 1, 3]
                    horizon_targets.append(future_close)
                
                y.append(horizon_targets)
            
            self.X = np.array(X)
            self.y = np.array(y)
            
            print(f"✅ Created {len(self.X):,} training sequences")
            print(f"📊 Input shape: {self.X.shape}")
            print(f"📊 Target shape: {self.y.shape}")
            print(f"🎯 Predicting {len(self.horizons)} horizons: {list(self.horizons.keys())}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating sequences: {e}")
            return False
    
    def split_train_test(self):
        """Split data chronologically (same approach as your 4H system)"""
        print(f"\n📊 Splitting data chronologically...")
        
        try:
            # Use 80% for training (same as your successful 4H system)
            split_idx = int(0.8 * len(self.X))
            
            self.X_train = self.X[:split_idx]
            self.X_test = self.X[split_idx:]
            self.y_train = self.y[:split_idx]
            self.y_test = self.y[split_idx:]
            
            print(f"✅ Training set: {len(self.X_train):,} sequences")
            print(f"✅ Test set: {len(self.X_test):,} sequences")
            print(f"📊 Train/Test split: {len(self.X_train)/len(self.X)*100:.1f}% / {len(self.X_test)/len(self.X)*100:.1f}%")
            
            return True
            
        except Exception as e:
            print(f"❌ Error splitting data: {e}")
            return False
    
    def build_model(self):
        """Build LSTM model adapted from your successful 4H architecture"""
        print(f"\n🏗️ Building 1H LSTM model (enhanced from your 4H architecture)...")
        
        try:
            self.model = tf.keras.Sequential([
                # First LSTM layer with return sequences
                tf.keras.layers.LSTM(64, return_sequences=True, 
                                   input_shape=(self.sequence_length, len(self.feature_columns)),
                                   name='lstm_layer_1'),
                tf.keras.layers.Dropout(0.2, name='dropout_1'),
                
                # Second LSTM layer
                tf.keras.layers.LSTM(32, return_sequences=False, name='lstm_layer_2'),
                tf.keras.layers.Dropout(0.2, name='dropout_2'),
                
                # Dense layers for processing
                tf.keras.layers.Dense(16, activation='relu', name='dense_1'),
                tf.keras.layers.Dropout(0.1, name='dropout_3'),
                
                # Output layer for multi-horizon predictions
                tf.keras.layers.Dense(len(self.horizons), name='multi_horizon_output')
            ])
            
            # Compile with same optimizer as your successful 4H model
            self.model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            print(f"✅ Model architecture built successfully")
            print(f"📊 Model summary:")
            self.model.summary()
            
            return True
            
        except Exception as e:
            print(f"❌ Error building model: {e}")
            return False
    
    def train_model(self):
        """Train the 1H LSTM model with monitoring"""
        print(f"\n🚀 Training 1H LSTM model...")
        print(f"⚙️ Epochs: {self.epochs}, Batch size: {self.batch_size}")
        
        try:
            # Callbacks for training optimization
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=7,
                    min_lr=1e-6,
                    verbose=1
                )
            ]
            
            # Train the model
            start_time = datetime.now()
            
            self.history = self.model.fit(
                self.X_train, self.y_train,
                validation_data=(self.X_test, self.y_test),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            training_time = datetime.now() - start_time
            print(f"✅ Training completed in {training_time}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during training: {e}")
            return False
    
    def evaluate_model(self):
        """Evaluate model performance across all horizons"""
        print(f"\n📊 Evaluating 1H LSTM model performance...")
        
        try:
            # Overall model evaluation
            test_loss, test_mae = self.model.evaluate(self.X_test, self.y_test, verbose=0)
            print(f"✅ Test Loss: {test_loss:.6f}")
            print(f"✅ Test MAE: {test_mae:.6f}")
            
            # Generate predictions for detailed analysis
            predictions = self.model.predict(self.X_test, verbose=0)
            
            # Evaluate each horizon individually
            print(f"\n🎯 Individual Horizon Performance:")
            for i, (horizon_name, _) in enumerate(self.horizons.items()):
                y_true = self.y_test[:, i]
                y_pred = predictions[:, i]
                
                mae = np.mean(np.abs(y_true - y_pred))
                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                
                print(f"  {horizon_name:>3}: MAE = {mae:.6f}, RMSE = {rmse:.6f}")
            
            # Store predictions for later analysis
            self.test_predictions = predictions
            
            return True
            
        except Exception as e:
            print(f"❌ Error evaluating model: {e}")
            return False
    
    def save_model_and_scaler(self):
        """Save trained model and scaler for ensemble integration"""
        print(f"\n💾 Saving 1H LSTM model and scaler...")
        
        try:
            # Save model
            self.model.save(self.model_save_path)
            print(f"✅ Model saved to: {self.model_save_path}")
            
            # Save scaler
            joblib.dump(self.scaler, self.scaler_save_path)
            print(f"✅ Scaler saved to: {self.scaler_save_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def plot_training_history(self):
        """Plot training history for analysis"""
        print(f"\n📈 Plotting training history...")
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot loss
            ax1.plot(self.history.history['loss'], label='Training Loss')
            ax1.plot(self.history.history['val_loss'], label='Validation Loss')
            ax1.set_title('1H LSTM Model Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Plot MAE
            ax2.plot(self.history.history['mae'], label='Training MAE')
            ax2.plot(self.history.history['val_mae'], label='Validation MAE')
            ax2.set_title('1H LSTM Model MAE')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig('1h_lstm_training_history.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"✅ Training history plot saved")
            
        except Exception as e:
            print(f"⚠️ Could not plot training history: {e}")
    
    def run_full_training(self):
        """Execute complete 1H LSTM training pipeline"""
        print(f"\n🚀 Starting Complete 1H LSTM Training Pipeline")
        print("=" * 70)
        
        # Step 1: Load and prepare data
        if not self.load_and_prepare_data():
            return False
        
        # Step 2: Create sequences and targets
        if not self.create_sequences_and_targets():
            return False
        
        # Step 3: Split train/test
        if not self.split_train_test():
            return False
        
        # Step 4: Build model
        if not self.build_model():
            return False
        
        # Step 5: Train model
        if not self.train_model():
            return False
        
        # Step 6: Evaluate model
        if not self.evaluate_model():
            return False
        
        # Step 7: Save model and scaler
        if not self.save_model_and_scaler():
            return False
        
        # Step 8: Plot training history
        self.plot_training_history()
        
        print(f"\n🎉 1H LSTM Training Complete!")
        print(f"✅ Model ready for multi-timeframe ensemble integration")
        print(f"✅ Enhanced precision timing for your 82.57% return system")
        print(f"🚀 Ready for Substep 4.3: Create Multi-Timeframe Ensemble")
        
        return True

def main():
    """Main function to train 1H LSTM model"""
    print("🎯 Phase 2 - Substep 4.2: 1H LSTM Model Training")
    print("Enhancing your successful 82.57% return system with 1H precision")
    print("=" * 70)
    
    # Initialize trainer
    trainer = HourlyLSTMTrainer(
        data_path='processed/BTCUSD_1h_cleaned.csv',
        model_save_path='models/btc_1h_lstm_model.h5',
        scaler_save_path='models/btc_1h_scaler.pkl'
    )
    
    # Run complete training pipeline
    success = trainer.run_full_training()
    
    if success:
        print(f"\n🎯 Substep 4.2 Complete!")
        print("✅ 1H LSTM model trained and validated")
        print("✅ Multi-horizon predictions optimized")
        print("✅ Model saved for ensemble integration")
        print(f"\n🚀 Next: Substep 4.3 - Multi-Timeframe Ensemble System")
    else:
        print(f"\n❌ Training failed - check logs for details")

if __name__ == "__main__":
    main()
