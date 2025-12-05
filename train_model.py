import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle
import os

def load_and_prepare_data():
    """
    Load electricity consumption data
    Dataset: Individual household electric power consumption
    Source: UCI Machine Learning Repository / Kaggle
    """
    print("Loading data...")
    
    # Read the dataset
    # Download from: https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set
    df = pd.read_csv('data/household_power_consumption.txt', 
                     sep=';', 
                     low_memory=False,
                     parse_dates={'datetime': ['Date', 'Time']},
                     infer_datetime_format=True)
    
    # Replace '?' with NaN
    df = df.replace('?', np.nan)
    
    # Convert to numeric
    for col in df.columns:
        if col != 'datetime':
            df[col] = pd.to_numeric(df[col])
    
    # Drop rows with missing values
    df = df.dropna()
    
    # Sample smaller dataset for faster processing (optional)
    df = df.sample(n=50000, random_state=42)
    
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return df

def engineer_features(df):
    """Create features for anomaly detection"""
    print("Engineering features...")
    
    # Extract time features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    
    # Select relevant features
    features = [
        'Global_active_power',
        'Global_reactive_power',
        'Voltage',
        'Global_intensity',
        'Sub_metering_1',
        'Sub_metering_2',
        'Sub_metering_3',
        'hour',
        'day_of_week',
        'month'
    ]
    
    X = df[features].copy()
    
    return X, df

def train_anomaly_model(X):
    """Train Isolation Forest model"""
    print("Training anomaly detection model...")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Isolation Forest
    model = IsolationForest(
        contamination=0.05,  # 5% anomalies expected
        random_state=42,
        n_estimators=100
    )
    
    model.fit(X_scaled)
    
    # Get predictions
    predictions = model.predict(X_scaled)
    anomaly_count = (predictions == -1).sum()
    
    print(f"Training complete!")
    print(f"Total samples: {len(predictions)}")
    print(f"Anomalies detected: {anomaly_count} ({anomaly_count/len(predictions)*100:.2f}%)")
    
    return model, scaler

def save_model(model, scaler):
    """Save trained model and scaler"""
    print("Saving model...")
    
    # Create model directory
    os.makedirs('model', exist_ok=True)
    
    # Save model and scaler
    with open('model/anomaly_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)
    
    print("Model saved to model/anomaly_model.pkl")

def main():
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Engineer features
    X, df_full = engineer_features(df)
    
    # Train model
    model, scaler = train_anomaly_model(X)
    
    # Save model
    save_model(model, scaler)
    
    print("\n✓ Training complete! Ready for deployment.")

if __name__ == "__main__":
    main()