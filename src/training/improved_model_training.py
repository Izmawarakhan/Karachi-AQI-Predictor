import pandas as pd
import numpy as np
import joblib, json, os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from src.utils.mongodb_feature_store import feature_store

def train_all_models():
    print("🚀 Training Models with Accuracy Tracking...")
    
    data = list(feature_store.db.model_features.find())
    if not data:
        print("❌ No data found!"); return
        
    df = pd.DataFrame(data)
    cols_to_drop = ['_id', 'date', 'city', 'next_day_aqi']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).select_dtypes(include=[np.number])
    y = df['next_day_aqi']

    # Features list save karna
    features = list(X.columns)
    os.makedirs('models/production', exist_ok=True)
    with open('models/production/features.json', 'w') as f:
        json.dump(features, f)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'models/production/scaler.joblib')

    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Ridge": Ridge(alpha=1.0),
        "Linear Regression": LinearRegression()
    }

    results = {}
    best_rmse = float('inf')
    best_model = None
    best_name = ""
    best_acc = 0

    for name, m in models.items():
        m.fit(X_train_scaled, y_train)
        preds = m.predict(X_test_scaled)
        
        rmse = root_mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        # Accuracy Percentage (R2 based)
        acc_pct = max(0, r2 * 100) 
        
        results[name] = {
            "rmse": round(float(rmse), 2), 
            "accuracy": f"{round(acc_pct, 2)}%"
        }
        print(f"📊 {name} -> RMSE: {rmse:.2f}, Accuracy: {round(acc_pct, 2)}%")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = m
            best_name = name
            best_acc = acc_pct

    # Final Winner Save
    joblib.dump(best_model, 'models/production/best_model.joblib')
    with open('models/production/metrics.json', 'w') as f:
        json.dump({
            "model_name": best_name, 
            "rmse": round(float(best_rmse), 2), 
            "accuracy": f"{round(best_acc, 2)}%",
            "all_results": results
        }, f)
        
    print(f"🏆 Winner: {best_name} | Accuracy: {round(best_acc, 2)}%")

if __name__ == "__main__":
    train_all_models()