import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from src.utils.mongodb_feature_store import feature_store

def fetch_historical_year():
    print("📜 Fetching 1 Year Historical Data for Karachi...")
    
    # 1. Dates Set Karein (Last 365 Days)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # 2. API URLs (Archive API use karni hogi puray saal ke liye)
    # Note: Archive API weather aur air quality dono ke liye alag call karni parti hai
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    aqi_url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    params_w = {
        "latitude": 24.8608,
        "longitude": 67.0011,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
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
        
        # Data Fetch Karein
        w_res = requests.get(weather_url, params=params_w).json()
        aqi_res = requests.get(aqi_url, params=params_aqi).json()

        # 3. DataFrames Banayein
        df_w = pd.DataFrame({
            'date': pd.to_datetime(w_res['hourly']['time']),
            'temperature': w_res['hourly']['temperature_2m'],
            'humidity': w_res['hourly']['relative_humidity_2m'],
            'wind_speed': w_res['hourly']['wind_speed_10m']
        })

        df_aqi = pd.DataFrame({
            'date': pd.to_datetime(aqi_res['hourly']['time']),
            'aqi_value': aqi_res['hourly']['pm2_5']
        })

        # 4. Merge Karein
        df_final = pd.merge(df_aqi, df_w, on='date', how='inner')
        df_final['city'] = 'Karachi'

        # 5. MongoDB mein Save Karein
        if not df_final.empty:
            # Purana raw data clear karna hai toh niche wali line uncomment karein
            # feature_store.db.aqi_raw.delete_many({}) 
            
            records = df_final.to_dict('records')
            feature_store.db.aqi_raw.insert_many(records)
            print(f"✅ Successfully saved {len(records)} hours of historical data to MongoDB!")
        else:
            print("⚠️ No data found for the specified range.")

    except Exception as e:
        print(f"❌ Error fetching historical data: {e}")

if __name__ == "__main__":
    fetch_historical_year()