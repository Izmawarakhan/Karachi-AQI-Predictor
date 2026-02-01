import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils.mongodb_feature_store import feature_store

def run_eda():
    print("📊 Starting Exploratory Data Analysis (EDA)...")
    
    # Data fetch karna
    data = list(feature_store.db.model_features.find())
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    
    # 1. AQI over time graph
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='date', y='aqi_value')
    plt.title('AQI Trend over Time (Karachi)')
    plt.savefig('eda_trend.png')
    print("✅ Trend graph saved as eda_trend.png")

    # 2. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.savefig('eda_correlation.png')
    print("✅ Heatmap saved as eda_correlation.png")

if __name__ == "__main__":
    import numpy as np
    run_eda()
    