# Karachi-AQI-Predictor
# 10pearlsAQI – Karachi Air Quality Index Prediction System

**Live Application:**(https://karachi-aqi-predictor-zec4fardmlwwrfmxcnkqqo.streamlit.app/)
**Author:** Izma Wara Khan
**Organization:** 10Pearls

---

## 1. Introduction

Air Quality Index (AQI) is a standardized indicator used to measure air pollution levels and their impact on human health. Due to high traffic density, industrial emissions, and seasonal weather variations, Karachi frequently experiences unhealthy AQI levels. This project presents an end-to-end AQI prediction system designed to forecast air quality conditions for the next three days using historical AQI data and weather parameters.

---

## 2. Project Overview

**10pearlsAQI** is an end-to-end machine learning–based AQI prediction system developed for Karachi, Pakistan. The system forecasts AQI for the next **3 days** by analyzing atmospheric pollutants (PM2.5/AQI) along with meteorological variables such as temperature, humidity, wind speed, and pressure.

### Objectives Achieved

* **Automated Data Pipeline:** Real-time hourly data ingestion from Open-Meteo and OpenWeather APIs.
* **Advanced Feature Engineering:** Creation of 24+ engineered features including lag variables and rolling statistics to capture AQI trends.
* **Model Benchmarking & Selection:** Multiple regression models were evaluated. Random Forest Regressor outperformed others based on RMSE and R² metrics and was selected for deployment.
* **Model Explainability:** SHAP (SHapley Additive exPlanations) was integrated to interpret model predictions and feature importance.
* **Interactive Dashboard:** A Streamlit-based dashboard was developed for real-time AQI monitoring and forecasting.
* **CI/CD Automation:** Fully automated pipelines using GitHub Actions for scheduled data collection and model training.

---

## 3. System Architecture

The system follows a decoupled, data-driven architecture consisting of three main layers:

* **Data Layer:** MongoDB Atlas is used as the centralized feature store.
* **Pipeline Layer:** GitHub Actions triggers two automated pipelines:

  * Hourly Feature Engineering Pipeline
  * Daily Model Training Pipeline
* **Service Layer:** A Streamlit dashboard consumes the trained model for real-time inference and visualization.

---

## 4. Technology Stack

| Category             | Tools / Technologies    |
| -------------------- | ----------------------- |
| Programming Language | Python                  |
| Dashboard            | Streamlit               |
| Machine Learning     | Random Forest Regressor |
| Data Processing      | Pandas                  |
| Visualization        | Plotly                  |
| Automation           | GitHub Actions          |
| Feature Store        | MongoDB Atlas           |

---

## 5. Data Pipeline & Feature Engineering

The system collects real-time hourly data from Open-Meteo and OpenWeather APIs. After preprocessing and cleaning, extensive feature engineering is applied to capture historical behavior and temporal patterns.

### Feature Engineering Techniques

* **Lag Features:** AQI(t-1), AQI(t-6), AQI(t-24) to model persistence in air quality.
* **Rolling Statistics:** Mean and standard deviation over 6-hour, 12-hour, and 24-hour windows.
* **Temporal Features:** Hour-of-day and day-of-week encoding to capture traffic and activity patterns.

---

## 6. Machine Learning Models & Performance

Multiple regression models were benchmarked using RMSE and R² metrics. The Random Forest model achieved the best overall performance and was deployed in production.

| Model             | R² (%)  | RMSE  | Status   |
| ----------------- | ------- | ----- | -------- |
| Random Forest     | ~69.18% | 7.22  | Deployed |
| Ridge Regression  | ~40.5%  | 10.12 | Baseline |
| Linear Regression | ~40.2%  | 10.15 | Baseline |

---

## 7. Model Interpretability (SHAP Analysis)

SHAP (SHapley Additive exPlanations) was used to interpret model predictions and explain the contribution of each feature.

### Key Insights

* Current AQI value is the most influential predictor of future AQI.
* Temperature and AQI lag-24h are among the top contributing features.

---

## 8. Dashboard

An interactive Streamlit dashboard serves as the main interface for real-time AQI prediction and visualization. The dashboard connects directly to the MongoDB feature store to display the latest data for Karachi.

### Core Functionalities

* **Model Performance Analytics:** Interactive Plotly charts comparing Accuracy (%) and RMSE of Random Forest with baseline models.
* **Live Atmospheric Status:** Displays current AQI and weather parameters based on the latest hourly data.
* **3-Day AQI Forecast:** Users can generate a 3-day AQI forecast using the deployed Random Forest model through a single action.

---

## 9. Troubleshooting & Challenges

### API Data Fetching

Challenges included missing values, inconsistent responses, varying time zones, and occasional API failures. These were resolved using timestamp normalization, retries, strict merging logic, null handling, and logging mechanisms.

### Feature Store Challenges

Hopsworks Feature Store was initially tested but faced authentication and free-tier limitations. MongoDB Atlas was finalized for improved stability. All credentials are securely managed using GitHub Secrets.

### Model Training Challenges

Frequent AQI spikes in Karachi reduced the effectiveness of linear models. Random Forest was selected for its ability to model non-linear patterns and improved robustness.

### CI/CD & Deployment Issues

GitHub Actions pipelines occasionally failed due to API timeouts and environment variable issues. Enhanced error handling, logging, and secrets management improved pipeline reliability.

---

## 10. Conclusion

10pearlsAQI successfully delivers a fully automated AQI forecasting solution for Karachi. By combining advanced feature engineering, Random Forest modeling, SHAP-based explainability, and CI/CD automation, the system provides reliable short-term AQI predictions and supports data-driven health awareness and de
