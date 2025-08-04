import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class HourlyDataProcessor:
    def __init__(self, input_file, output_file):
        """
        Process 1H BTC data for LSTM training
        Maintains same quality standards as your successful 4H system
        """
        self.input_file = input_file
        self.output_file = output_file
        
        print(f"🕐 1H Data Processor Initialized")
        print(f"📁 Input: {input_file}")
        print(f"💾 Output: {output_file}")
    
    def load_and_clean_data(self):
        """Load and perform initial cleaning of 1H data"""
        try:
            # Load the CSV file
            print(f"\n📊 Loading 1H BTC data...")
            df = pd.read_csv(self.input_file)
            
            print(f"✅ Loaded {len(df):,} raw 1H candles")
            print(f"📅 Columns: {df.columns.tolist()}")
            
            # Display sample of raw data
            print(f"\n📋 Raw Data Sample:")
            print(df.head(3))
            
            # Standardize column names to match your 4H system
            column_mapping = {
                'Local time': 'datetime',
                'Open': 'Open',
                'High': 'High', 
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            }
            
            # Apply column mapping if needed
            if 'Local time' in df.columns:
                df = df.rename(columns=column_mapping)
                print(f"✅ Standardized column names")
            
            # Ensure we have required columns
            required_columns = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"❌ Missing required columns: {missing_columns}")
                return None
            
            self.raw_data = df[required_columns].copy()
            print(f"✅ Data structure validated")
            
            return self.raw_data
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def clean_datetime_format(self):
        """Clean and standardize datetime format"""
        print(f"\n🕐 Processing datetime column...")
        
        try:
            # Convert datetime string to pandas datetime with European format (FIXED)
            self.raw_data['datetime'] = pd.to_datetime(self.raw_data['datetime'], dayfirst=True)
            
            # Sort by datetime to ensure chronological order
            self.raw_data = self.raw_data.sort_values('datetime').reset_index(drop=True)
            
            # Remove any duplicate timestamps
            initial_count = len(self.raw_data)
            self.raw_data = self.raw_data.drop_duplicates(subset=['datetime']).reset_index(drop=True)
            duplicates_removed = initial_count - len(self.raw_data)
            
            if duplicates_removed > 0:
                print(f"⚠️ Removed {duplicates_removed} duplicate timestamps")
            
            print(f"✅ Datetime processing complete")
            print(f"📅 Date range: {self.raw_data['datetime'].min()} to {self.raw_data['datetime'].max()}")
            print(f"🕐 Total timespan: {(self.raw_data['datetime'].max() - self.raw_data['datetime'].min()).days} days")
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing datetime: {e}")
            return False
    
    def validate_and_clean_ohlcv(self):
        """Validate and clean OHLCV data with same standards as 4H system"""
        print(f"\n💰 Validating and cleaning OHLCV data...")
        
        try:
            # Convert OHLCV columns to numeric
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            for col in numeric_columns:
                self.raw_data[col] = pd.to_numeric(self.raw_data[col], errors='coerce')
                
                # Check for NaN values
                nan_count = self.raw_data[col].isna().sum()
                if nan_count > 0:
                    print(f"⚠️ Found {nan_count} NaN values in {col}")
            
            # Remove rows with any NaN values
            initial_count = len(self.raw_data)
            self.raw_data = self.raw_data.dropna().reset_index(drop=True)
            nan_removed = initial_count - len(self.raw_data)
            
            if nan_removed > 0:
                print(f"⚠️ Removed {nan_removed} rows with NaN values")
            
            # Validate OHLC relationships (High >= Open,Close,Low and Low <= Open,Close,High)
            invalid_ohlc = (
                (self.raw_data['High'] < self.raw_data['Open']) |
                (self.raw_data['High'] < self.raw_data['Close']) |
                (self.raw_data['High'] < self.raw_data['Low']) |
                (self.raw_data['Low'] > self.raw_data['Open']) |
                (self.raw_data['Low'] > self.raw_data['Close']) |
                (self.raw_data['Low'] > self.raw_data['High'])
            )
            
            invalid_count = invalid_ohlc.sum()
            if invalid_count > 0:
                print(f"⚠️ Found {invalid_count} rows with invalid OHLC relationships")
                # Remove invalid OHLC rows
                self.raw_data = self.raw_data[~invalid_ohlc].reset_index(drop=True)
                print(f"✅ Removed invalid OHLC rows")
            
            # Check for unrealistic price movements (>50% in 1 hour)
            self.raw_data['price_change'] = self.raw_data['Close'].pct_change()
            extreme_moves = abs(self.raw_data['price_change']) > 0.5
            extreme_count = extreme_moves.sum()
            
            if extreme_count > 0:
                print(f"⚠️ Found {extreme_count} extreme price movements (>50% in 1H)")
                # Flag but don't remove - could be legitimate in crypto
                print(f"ℹ️ Keeping extreme moves - common in cryptocurrency markets")
            
            # Validate volume (should be positive)
            negative_volume = self.raw_data['Volume'] < 0
            neg_vol_count = negative_volume.sum()
            
            if neg_vol_count > 0:
                print(f"⚠️ Found {neg_vol_count} negative volume values")
                self.raw_data = self.raw_data[~negative_volume].reset_index(drop=True)
                print(f"✅ Removed negative volume rows")
            
            print(f"✅ OHLCV validation complete")
            print(f"📊 Final clean dataset: {len(self.raw_data):,} 1H candles")
            
            return True
            
        except Exception as e:
            print(f"❌ Error validating OHLCV data: {e}")
            return False
    
    def detect_and_fill_gaps(self):
        """Detect and handle missing hourly timestamps"""
        print(f"\n🔍 Checking for missing hourly timestamps...")
        
        try:
            # Create complete hourly range
            start_time = self.raw_data['datetime'].min()
            end_time = self.raw_data['datetime'].max()
            
            complete_range = pd.date_range(start=start_time, end=end_time, freq='H')
            existing_times = set(self.raw_data['datetime'])
            
            missing_times = [t for t in complete_range if t not in existing_times]
            
            if missing_times:
                print(f"⚠️ Found {len(missing_times)} missing hourly timestamps")
                print(f"📊 Data completeness: {(len(existing_times)/len(complete_range))*100:.2f}%")
                
                # For 1H data, we'll forward-fill missing values (more conservative than interpolation)
                print(f"🔧 Forward-filling missing timestamps...")
                
                # Create DataFrame with complete time range
                complete_df = pd.DataFrame({'datetime': complete_range})
                
                # Merge with existing data
                merged_df = pd.merge(complete_df, self.raw_data, on='datetime', how='left')
                
                # Forward fill missing values
                merged_df = merged_df.fillna(method='ffill')
                
                # Update raw_data
                self.raw_data = merged_df.dropna().reset_index(drop=True)
                
                print(f"✅ Filled {len(missing_times)} missing timestamps")
                print(f"📊 Updated dataset: {len(self.raw_data):,} 1H candles")
            else:
                print(f"✅ No missing timestamps found - complete hourly dataset")
            
            return True
            
        except Exception as e:
            print(f"❌ Error handling timestamp gaps: {e}")
            return False
    
    def calculate_technical_features(self):
        """Calculate additional technical features for enhanced 1H analysis"""
        print(f"\n📈 Calculating technical features for 1H analysis...")
        
        try:
            # Price-based features
            self.raw_data['price_range'] = self.raw_data['High'] - self.raw_data['Low']
            self.raw_data['price_change_pct'] = self.raw_data['Close'].pct_change() * 100
            
            # Volume-based features
            self.raw_data['volume_ma_24h'] = self.raw_data['Volume'].rolling(window=24).mean()
            self.raw_data['volume_ratio'] = self.raw_data['Volume'] / self.raw_data['volume_ma_24h']
            
            # Volatility features (important for 1H timeframe)
            self.raw_data['volatility_6h'] = self.raw_data['price_change_pct'].rolling(window=6).std()
            self.raw_data['volatility_24h'] = self.raw_data['price_change_pct'].rolling(window=24).std()
            
            # Price momentum (useful for 1H predictions)
            self.raw_data['momentum_6h'] = self.raw_data['Close'] / self.raw_data['Close'].shift(6)
            self.raw_data['momentum_24h'] = self.raw_data['Close'] / self.raw_data['Close'].shift(24)
            
            # FIXED: Only remove first 24 rows due to rolling window requirements
            initial_count = len(self.raw_data)
            self.raw_data = self.raw_data.iloc[24:].reset_index(drop=True)
            feature_removed = initial_count - len(self.raw_data)
            
            print(f"ℹ️ Removed {feature_removed} initial rows due to rolling window requirements")
            print(f"✅ Technical features calculated")
            print(f"📊 Remaining clean data: {len(self.raw_data):,} 1H candles")
            print(f"📊 Features added: price_range, price_change_pct, volume_ratio, volatility metrics, momentum metrics")
            
            return True
            
        except Exception as e:
            print(f"❌ Error calculating technical features: {e}")
            return False
    
    def align_with_4h_timeframe(self):
        """Ensure 1H data aligns properly with your existing 4H system"""
        print(f"\n🔄 Aligning 1H data with 4H timeframe compatibility...")
        
        try:
            # FIXED: Ensure datetime column is in proper datetime format
            self.raw_data['datetime'] = pd.to_datetime(self.raw_data['datetime'], errors='coerce')
            
            # Remove timezone info if present to avoid .dt accessor issues
            if hasattr(self.raw_data['datetime'].dtype, 'tz') and self.raw_data['datetime'].dtype.tz is not None:
                self.raw_data['datetime'] = self.raw_data['datetime'].dt.tz_localize(None)
            
            # Add 4H period identifier for alignment
            self.raw_data['4h_period'] = self.raw_data['datetime'].dt.floor('4H')
            
            # Add hour of day feature (important for 1H patterns)
            self.raw_data['hour_of_day'] = self.raw_data['datetime'].dt.hour
            
            # Add day of week feature
            self.raw_data['day_of_week'] = self.raw_data['datetime'].dt.dayofweek
            
            # Verify alignment with 4H boundaries
            four_hour_marks = [0, 4, 8, 12, 16, 20]  # Standard 4H candle start hours
            alignment_check = self.raw_data['hour_of_day'].isin(four_hour_marks)
            aligned_count = alignment_check.sum()
            
            print(f"✅ 4H alignment features added")
            print(f"📊 {aligned_count:,} candles align with 4H boundaries ({aligned_count/len(self.raw_data)*100:.1f}%)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error aligning with 4H timeframe: {e}")
            return False
    
    def generate_final_dataset(self):
        """Generate final clean dataset for LSTM training"""
        print(f"\n💾 Generating final 1H dataset...")
        
        try:
            # Select core columns for LSTM (matching your 4H system)
            lstm_columns = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            
            # Add selected technical features
            enhanced_columns = lstm_columns + [
                'price_range', 'volume_ratio', 'volatility_24h', 
                'momentum_6h', 'hour_of_day'
            ]
            
            # Create final dataset
            available_columns = [col for col in enhanced_columns if col in self.raw_data.columns]
            self.final_data = self.raw_data[available_columns].copy()
            
            # Final data validation
            print(f"📊 Final Dataset Summary:")
            print(f"   • Total 1H candles: {len(self.final_data):,}")
            print(f"   • Date range: {self.final_data['datetime'].min()} to {self.final_data['datetime'].max()}")
            print(f"   • Columns: {len(self.final_data.columns)}")
            print(f"   • Memory usage: {self.final_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Display sample of final data
            print(f"\n📋 Final Data Sample:")
            print(self.final_data.head())
            
            # Save to CSV
            self.final_data.to_csv(self.output_file, index=False)
            print(f"\n💾 Final 1H dataset saved to: {self.output_file}")
            
            return self.final_data
            
        except Exception as e:
            print(f"❌ Error generating final dataset: {e}")
            return None
    
    def run_full_processing(self):
        """Execute complete 1H data processing pipeline"""
        print(f"\n🚀 Starting Complete 1H Data Processing Pipeline")
        print("=" * 70)
        
        # Step 1: Load and clean data
        if self.load_and_clean_data() is None:
            return None
        
        # Step 2: Process datetime
        if not self.clean_datetime_format():
            return None
        
        # Step 3: Validate OHLCV
        if not self.validate_and_clean_ohlcv():
            return None
        
        # Step 4: Handle missing timestamps
        if not self.detect_and_fill_gaps():
            return None
        
        # Step 5: Calculate features
        if not self.calculate_technical_features():
            return None
        
        # Step 6: Align with 4H system
        if not self.align_with_4h_timeframe():
            return None
        
        # Step 7: Generate final dataset
        final_data = self.generate_final_dataset()
        
        if final_data is not None:
            print(f"\n🎉 1H Data Processing Complete!")
            print(f"✅ High-quality 1H dataset ready for LSTM training")
            print(f"📊 {len(final_data):,} clean 1H candles processed")
            print(f"🚀 Ready for Substep 4.2: 1H LSTM Model Training")
            
            return final_data
        else:
            print(f"\n❌ 1H Data Processing Failed")
            return None

def main():
    """Main function to process 1H BTC data"""
    print("🎯 Phase 2 - Substep 4.1: 1H Data Processing")
    print("Building on your successful 82.57% return 4H system")
    print("=" * 70)
    
    # Initialize processor with your actual file path
    processor = HourlyDataProcessor(
        input_file='raw/BTCUSD_Candlestick_1_Hour_BID_01.07.2020-26.07.2025.csv',
        output_file='processed/BTCUSD_1h_cleaned.csv'
    )
    
    # Run complete processing
    final_data = processor.run_full_processing()
    
    if final_data is not None:
        print(f"\n🎯 Substep 4.1 Complete!")
        print("✅ 1H data processed and ready")
        print("✅ Quality matches your successful 4H system")
        print("✅ Enhanced with 1H-specific technical features")
        print(f"\n🚀 Next: Substep 4.2 - Train 1H LSTM Model")

if __name__ == "__main__":
    main()
