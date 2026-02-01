import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os
from pymongo import MongoClient

def run_model_explanation():
    """
    Perform SHAP analysis to explain model predictions for Karachi AQI.
    This fulfills the 'Explainability' requirement in the project guidelines.
    """
    print("Step 1: Connecting to MongoDB and fetching data...")
    
    # Connection setup
    mongo_uri = os.environ.get('MONGO_URI', "mongodb://localhost:27017/")
    client = MongoClient(mongo_uri)
    db = client.aqi_predictor
    
    # Fetch recent data for analysis
    data_cursor = db.model_features.find().sort("date", -1).limit(200)
    df = pd.DataFrame(list(data_cursor))
    
    if df.empty:
        print("Error: No data found in MongoDB. Please run the feature pipeline first.")
        return

    print("Step 2: Loading production model and scaler...")
    # Load trained model and scaler from registry
    model = joblib.load('models/production/best_model.joblib')
    scaler = joblib.load('models/production/scaler.joblib')
    
    # Prepare features by dropping non-feature columns
    # We drop metadata columns to match the training feature set
    X = df.drop(columns=['_id', 'date', 'city', 'next_day_aqi'], errors='ignore')
    
    # Apply scaling transformation
    X_scaled = scaler.transform(X)
    X_display = pd.DataFrame(X_scaled, columns=X.columns)

    print("Step 3: Calculating SHAP values (this may take a moment)...")
    # Using TreeExplainer for Random Forest / XGBoost models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_display)

    print("Step 4: Generating explanation plots...")
    
    # Plot 1: Feature Importance Bar Chart
    # Shows which features have the highest global impact on predictions
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_display, plot_type="bar", show=False)
    plt.title("Global Feature Importance (SHAP)")
    plt.savefig('eda_shap_importance.png', bbox_inches='tight')
    plt.close()

    # Plot 2: Beeswarm Plot
    # Shows how high/low values of a feature affect the AQI prediction
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_display, show=False)
    plt.title("Feature Impact Distribution (Beeswarm Plot)")
    plt.savefig('eda_shap_beeswarm.png', bbox_inches='tight')
    plt.close()
    
    print("Success: Explanation plots saved to project root directory.")

if __name__ == "__main__":
    run_model_explanation()