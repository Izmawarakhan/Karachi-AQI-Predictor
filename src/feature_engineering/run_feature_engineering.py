import pandas as pd
import numpy as np
from src.utils.mongodb_feature_store import feature_store

def run_master_engineering():
    print("🧪 Starting Multi-Model Feature Engineering...")
    
    # Raw data fetch karna
    data = list(feature_store.db.aqi_raw.find())
    if not data:
        print("❌ No raw data found!"); return
        
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').drop_duplicates(subset=['date'])

    # Target: Aglay 24 ghante ka AQI
    df['next_day_aqi'] = df['aqi_value'].shift(-24)

    # Common Features: Lags, Rolling, and Time
    for lag in [1, 6, 24]:
        df[f'aqi_lag_{lag}h'] = df['aqi_value'].shift(lag)
    
    for window in [6, 12, 24]:
        df[f'aqi_mean_{window}h'] = df['aqi_value'].rolling(window=window).mean()
        df[f'aqi_std_{window}h'] = df['aqi_value'].rolling(window=window).std()

    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Cleaning
    df_final = df.dropna().copy()
    df_final['date'] = df_final['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    feature_store.db.model_features.delete_many({}) 
    feature_store.db.model_features.insert_many(df_final.to_dict('records'))
    print(f"✅ SUCCESS: {len(df_final)} samples saved for multi-model training.")

if __name__ == "__main__":
    run_master_engineering()