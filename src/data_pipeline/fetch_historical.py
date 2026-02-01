import requests
import pandas as pd
from datetime import datetime, timedelta
from src.utils.mongodb_feature_store import feature_store

def fetch_historical_year():
    print("📜 Fetching Fresh 1-Year Historical Data for Karachi...")
    
    # 1. Dates (Last 365 Days)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    aqi_url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    # Added surface_pressure for 90% accuracy goal
    params_w = {
        "latitude": 24.8608,
        "longitude": 67.0011,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure",
        "timezone": "auto"
    }

    params_aqi = {
        "latitude": 24.8608,
        "longitude": 67.0011,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "pm2_5",
        "timezone": "auto"
    }

    try:
        print(f"📅 Range: {start_date} to {end_date}")
        
        w_res = requests.get(weather_url, params=params_w).json()
        aqi_res = requests.get(aqi_url, params=params_aqi).json()

        # 3. DataFrames
        df_w = pd.DataFrame({
            'date': pd.to_datetime(w_res['hourly']['time']),
            'temperature': w_res['hourly']['temperature_2m'],
            'humidity': w_res['hourly']['relative_humidity_2m'],
            'wind_speed': w_res['hourly']['wind_speed_10m'],
            'pressure': w_res['hourly']['surface_pressure']  # New Feature
        })

        df_aqi = pd.DataFrame({
            'date': pd.to_datetime(aqi_res['hourly']['time']),
            'aqi_value': aqi_res['hourly']['pm2_5']
        })

        # 4. Merge
        df_final = pd.merge(df_aqi, df_w, on='date', how='inner')
        df_final['city'] = 'Karachi'

        if not df_final.empty:
            # CLEAN SLATE: Purana data delete karein
            print("🗑️ Clearing old data from MongoDB...")
            feature_store.db.aqi_raw.delete_many({}) 
            
            records = df_final.to_dict('records')
            feature_store.db.aqi_raw.insert_many(records)
            print(f"✅ Successfully saved {len(records)} hours of fresh data to MongoDB!")
        else:
            print("⚠️ No data found.")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    fetch_historical_year()