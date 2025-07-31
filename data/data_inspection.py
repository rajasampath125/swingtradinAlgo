import pandas as pd
import numpy as np

# Load the BTC/USD 4H data
file_path = "raw/BTCUSD_Candlestick_4_Hour_BID_10.07.2020-19.07.2025.csv"
df = pd.read_csv(file_path)

print("=== BASIC DATA INFO ===")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Data types:\n{df.dtypes}")

print("\n=== FIRST 5 ROWS ===")
print(df.head())

print("\n=== LAST 5 ROWS ===")
print(df.tail())

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

print("\n=== DATE RANGE CHECK ===")
print(f"First timestamp: {df.iloc[0, 0]}")
print(f"Last timestamp: {df.iloc[-1, 0]}")
print(f"Total data points: {len(df)}")
