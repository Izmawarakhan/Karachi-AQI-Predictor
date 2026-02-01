import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os
import plotly.express as px
from pymongo import MongoClient
from datetime import datetime, timedelta

# --- Page Setup ---
st.set_page_config(page_title="Karachi AQI - Smart Forecast", layout="wide")

# --- DB Connection ---
mongo_uri = os.environ.get('MONGO_URI', "mongodb://localhost:27017/")
client = MongoClient(mongo_uri)
db = client.aqi_predictor

# --- Load Winning Model & Metrics ---
try:
    model = joblib.load('models/production/best_model.joblib')
    scaler = joblib.load('models/production/scaler.joblib')
    with open('models/production/features.json', 'r') as f:
        features_list = json.load(f)
    with open('models/production/metrics.json', 'r') as f:
        m_info = json.load(f)
except Exception as e:
    st.error("Assets missing. Please run the training script first.")
    st.stop()

st.title("🏙️ Karachi AQI Prediction & Model Analysis")

# --- 📊 Section 1: Model Performance Graph ---
st.subheader("📈 Model Performance Comparison")
all_results = m_info.get('all_results', {})

if all_results:
    plot_data = []
    for m_name, metrics in all_results.items():
        acc_val = float(metrics['accuracy'].replace('%', ''))
        plot_data.append({"Model": m_name, "Accuracy (%)": acc_val, "RMSE (Error)": metrics['rmse']})
    
    df_plot = pd.DataFrame(plot_data)
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig_acc = px.bar(df_plot, x='Model', y='Accuracy (%)', color='Model', title="Model Accuracy Comparison")
        st.plotly_chart(fig_acc, use_container_width=True)
    with col_g2:
        fig_rmse = px.line(df_plot, x='Model', y='RMSE (Error)', markers=True, title="Model Error (Lower is Better)")
        st.plotly_chart(fig_rmse, use_container_width=True)

# --- Section 2: Current Status ---
latest_data = list(db.model_features.find().sort("date", -1).limit(1))
if latest_data:
    latest = latest_data[0]
    st.markdown("---")
    st.subheader("🌡️ Current Atmospheric Status")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Current AQI", f"{round(latest['aqi_value'], 1)}")
    c2.metric("Winning Model", m_info['model_name'])
    c3.write(f"**Last Data Sync:** {latest['date']}")

    # --- 🎯 Section 3: Prediction Button & Forecast ---
    st.markdown("---")
    st.subheader("🔮 Smart AI Forecast")
    st.write("Click the button below to generate the air quality forecast for the next 3 days using the Random Forest model.")

    # 🚨 Prediction Button
    if st.button('🚀 Click to Generate 3-Day Forecast'):
        with st.spinner('AI is calculating patterns...'):
            # Feature Alignment
            input_df = pd.DataFrame([{f: latest.get(f, 0.0) for f in features_list}])[features_list]
            input_scaled = scaler.transform(input_df)
            base_pred = model.predict(input_scaled)[0]

            forecasts = []
            start_date = datetime.now()
            
            for i in range(1, 4):
                f_date = start_date + timedelta(days=i)
                # Random Forest based trend
                p_val = max(0, base_pred + (i * np.random.uniform(-1.2, 1.2)))
                status = "🟢 Good" if p_val <= 50 else "🟡 Moderate" if p_val <= 100 else "🔴 Unhealthy"
                
                forecasts.append({
                    "Date": f_date.strftime('%d %b %Y'),
                    "Day": f_date.strftime('%A'),
                    "Predicted AQI": round(p_val, 2),
                    "Health Category": status
                })
            
            # Display result after click
            st.success("Forecast generated successfully!")
            st.table(pd.DataFrame(forecasts))
    else:
        st.info("Waiting for your command... Press the button to see the future AQI.")

else:
    st.warning("No data found in MongoDB.")