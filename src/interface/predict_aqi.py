import joblib
import pandas as pd
import numpy as np
from src.utils.mongodb_feature_store import feature_store

def get_prediction():
    print("🔮 Fetching latest data for prediction...")
    
    # 1. Load Model and Scaler
    try:
        model = joblib.load('models/production/best_model.joblib')
        scaler = joblib.load('models/production/scaler.joblib')
    except:
        return "❌ Models not found! Train the model first."

    # 2. Get the latest feature record from MongoDB
    latest_data = list(feature_store.db.model_features.find().sort('date', -1).limit(1))
    
    if not latest_data:
        return "⚠️ No features found in Database."

    df_latest = pd.DataFrame(latest_data)
    
    # Pre-processing: Remove non-feature columns
    X_latest = df_latest.drop(['_id', 'date', 'city', 'next_day_aqi'], axis=1, errors='ignore')
    
    # 3. Scaling & Prediction
    X_scaled = scaler.transform(X_latest)
    prediction = model.predict(X_scaled)[0]
    
    # AQI negative nahi ho sakta
    prediction = max(0, round(prediction, 2))
    
    return {
        "current_time": df_latest['date'].values[0],
        "current_aqi": df_latest['aqi_value'].values[0],
        "predicted_aqi_24h": prediction,
        "status": "Success"
    }

if __name__ == "__main__":
    result = get_prediction()
    print(f"\n📍 Location: Karachi")
    print(f"🕒 Last Update: {result['current_time']}")
    print(f"😷 Current AQI: {result['current_aqi']}")
    print(f"🚀 Predicted AQI (Next 24h): {result['predicted_aqi_24h']}")