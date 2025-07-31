try:
    from sklearn.preprocessing import MinMaxScaler
    import joblib
    import tensorflow as tf
    print("✅ All required packages installed successfully!")
    print(f"Scikit-learn: Available")
    print(f"Joblib: Available") 
    print(f"TensorFlow: {tf.__version__}")
except ImportError as e:
    print(f"❌ Missing package: {e}")
