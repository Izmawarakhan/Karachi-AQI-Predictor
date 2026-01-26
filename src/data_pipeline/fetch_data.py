import requests
import pandas as pd
from src.utils.mongodb_feature_store import feature_store

def fetch_live():
    print("🛰️ Fetching Live Data...")
    # Air Quality & Weather API calls
    aqi_url = "https://air-quality-api.open-meteo.com/v1/air-quality?latitude=24.8608&longitude=67.0011&hourly=pm2_5&timezone=auto"
    w_url = "https://api.open-meteo.com/v1/forecast?latitude=24.8608&longitude=67.0011&hourly=temperature_2m,relative_humidity_2m&timezone=auto"
    
    try:
        aqi_res = requests.get(aqi_url).json()
        w_res = requests.get(w_url).json()
        
        # 1. AQI Dataframe (pm2_5)
        df_aqi = pd.DataFrame({
            'date': pd.to_datetime(aqi_res['hourly']['time']),
            'aqi_value': aqi_res['hourly']['pm2_5']
        })
        
        # 2. Weather Dataframe (temp, humidity)
        df_weather = pd.DataFrame({
            'date': pd.to_datetime(w_res['hourly']['time']),
            'temperature': w_res['hourly']['temperature_2m'],
            'humidity': w_res['hourly']['relative_humidity_2m']
        })
        
        # 3. MERGE (The Fix): Dono ko 'date' column par match karein
        # Isse length ka masla hal ho jayega (sirf matching hours rahenge)
        df = pd.merge(df_aqi, df_weather, on='date', how='inner')
        df['city'] = 'Karachi'
        
        if not df.empty:
            records = df.to_dict('records')
            feature_store.db.aqi_raw.insert_many(records)
            print(f"✅ Successfully merged and saved {len(records)} records to MongoDB.")
        else:
            print("⚠️ No matching data found between APIs.")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    fetch_live()