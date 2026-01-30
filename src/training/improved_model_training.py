import pandas as pd
import numpy as np
import joblib
import json
import os
from pymongo import MongoClient
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Import centralized connection class
from src.utils.mongodb_feature_store import feature_store

def train_and_evaluate():
    print("🚀 STARTING OPTIMIZED TRAINING: Tuning for Better Accuracy")
    print("="*60)

    # 1. Load Feature data from MongoDB Atlas Cloud
    # Feature store handles Local vs Cloud connection
    db = feature_store.db
    
    data = list(db.model_features.find())
    if not data:
        print("❌ No data found in model_features! Run feature engineering first.")
        return
        
    df = pd.DataFrame(data).drop('_id', axis=1)
    
    # 2. Data Preparation
    target = 'next_day_aqi'
    X_raw = df.select_dtypes(include=[np.number])
    feature_cols = [c for c in X_raw.columns if c != target]
    
    X = X_raw[feature_cols].fillna(0)
    y = X_raw[target].fillna(X_raw[target].mean())

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Optimized Models Definition (Tuned Hyperparameters)
    models = {
        # XGBoost tuned for sharper patterns
        "XGBoost": xgb.XGBRegressor(
            n_estimators=300,      # Increased from 100
            learning_rate=0.03,    # Slower learning for better precision
            max_depth=10,          # Deeper trees to catch AQI spikes
            subsample=0.8,
            colsample_bytree=0.8
        ),
        # Random Forest tuned to reduce smoothing
        "RandomForest": RandomForestRegressor(
            n_estimators=200,      # More trees for stability
            max_depth=15,          # Deeper trees to follow Actual line
            min_samples_split=2,   # Sensitivity to small data changes
            random_state=42
        ),
        "LinearRegression": LinearRegression()
    }

    results = {}
    trained_objects = {}

    # 4. Training Loop
    for name, model in models.items():
        print(f"--- Training {name} ---")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        results[name] = rmse
        trained_objects[name] = model
        print(f"📉 {name} RMSE: {rmse:.2f}")

    # 5. Best Model Selection
    best_model_name = min(results, key=results.get)
    best_model = trained_objects[best_model_name]
    
    print("="*60)
    print(f"🏆 WINNER: {best_model_name} with RMSE {results[best_model_name]:.2f}")

    # 6. Save Model and Features
    os.makedirs('models/production', exist_ok=True)
    joblib.dump(best_model, 'models/production/best_model.joblib')
    
    # Save feature list for the dashboard
    with open('models/production/features.json', 'w') as f:
        json.dump(feature_cols, f)
        
    print(f"✅ Optimized model saved to models/production/best_model.joblib")

if __name__ == "__main__":
    train_and_evaluate()