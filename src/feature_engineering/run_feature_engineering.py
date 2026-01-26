import pandas as pd
import numpy as np
from src.utils.mongodb_feature_store import feature_store

def run_engineering():
    print("🧪 Starting Feature Engineering...")
    
    # 1. MongoDB se raw data uthaein
    data = list(feature_store.db.aqi_raw.find())
    if not data:
        print("❌ No raw data found in MongoDB!")
        return
    
    df = pd.DataFrame(data)
    # _id nikal dein agar maujood ho kyunki naye insert mein naya _id banega
    if '_id' in df.columns:
        df = df.drop('_id', axis=1)
    
    # Date processing
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').drop_duplicates(subset=['date'])
    
    print(f"📊 Processing {len(df)} raw records...")

    # 2. TARGET VARIABLE
    df['next_day_aqi'] = df['aqi_value'].shift(-24)
    
    # 3. LAG FEATURES
    df['aqi_lag_1h'] = df['aqi_value'].shift(1)
    df['aqi_lag_24h'] = df['aqi_value'].shift(24)
    
    # 4. ROLLING FEATURES
    df['aqi_rolling_avg_24h'] = df['aqi_value'].rolling(window=24, min_periods=1).mean()
    
    # 5. TIME FEATURES
    df['hour'] = df['date'].dt.hour
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # 6. CLEANING (HATAO NaN/NaT)
    # Sirf woh rows rakhein jinka target maujood hai
    df_final = df.dropna(subset=['next_day_aqi']).copy()
    
    # --- 🔥 THE ULTIMATE FIX FOR NaT/NaN ERROR ---
    # 1. Dates ko string banayein
    df_final['date'] = df_final['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # 2. Tamam NaN aur NaT ko None (Null) mein badal dein jo MongoDB accept karta hai
    df_final = df_final.replace({np.nan: None})
    # ---------------------------------------------

    # 7. FINAL CHECK & SAVE
    if not df_final.empty:
        try:
            # Table/Collection update
            feature_store.db.model_features.delete_many({}) 
            records = df_final.to_dict('records')
            
            # MongoDB Insert
            feature_store.db.model_features.insert_many(records)
            print(f"✅ Feature Engineering Complete! {len(records)} samples saved to 'model_features'.")
        except Exception as e:
            print(f"❌ MongoDB Insert Error: {e}")
    else:
        print("❌ Error: No data left after cleaning.")

if __name__ == "__main__":
    run_engineering()