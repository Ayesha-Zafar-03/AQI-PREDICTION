import os
import sys
import joblib
import numpy as np
import pandas as pd

# --- Ensure src is importable when running directly ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def naive_forecast(current_pm25, historical_data=None):
    if historical_data and len(historical_data) >= 7:
        return sum(historical_data[-7:]) / 7  # 7-day moving average
    return current_pm25  # Persistence

def load_model_local(path="models/aqi_model/random_forest.pkl"):
    """Load a trained model from local storage."""
    if os.path.exists(path):
        try:
            print(f"Loading model from {path}...")
            return joblib.load(path)
        except Exception as e:
            print(f"Failed to load model: {e}")
            return None
    else:
        print(f"Model file not found at {path}")
        return None


def predict_from_model(model, recent_feature_row):
    """Predict AQI using the trained model."""
    if model is None:
        return None

    # ✅ Ensure we have exactly the same features as training
    expected_features = [
        'pm2_5', 'pm10', 'no2', 'o3', 'so2', 'co', 'aqi',
        'dayofweek', 'day', 'month',
        'pm2_5_lag1', 'pm2_5_lag2', 'pm2_5_lag3',
        'pm2_5_roll7', 'pm2_5_roll14'
    ]

    # Reorder and fill missing columns with 0 (or NaN if your model handles it)
    for col in expected_features:
        if col not in recent_feature_row.columns:
            recent_feature_row[col] = 0

    X = recent_feature_row[expected_features]

    preds = model.predict(X)
    return [int(round(float(x))) for x in np.array(preds).flatten()]


def predict_next_3_days(model, current_data, history_df):
    """Predict AQI for the next 3 days using the model."""
    preds = []
    updated_history = history_df.copy()
    current_input = current_data.copy()

    # Get recent averages for other pollutants from history (last 7 days)
    if not updated_history.empty:
        recent = updated_history.tail(7)
        avg_values = {
            'pm10': recent['pm10'].mean(),
            'no2': recent['no2'].mean(),
            'o3': recent['o3'].mean(),
            'so2': recent['so2'].mean(),
            'co': recent['co'].mean(),
            # Use 'openweather_aqi' if available, otherwise fallback to 'aqi'
            'openweather_aqi': recent['openweather_aqi'].mean() if 'openweather_aqi' in recent.columns
                                else recent['aqi'].mean() if 'aqi' in recent.columns else 0
        }
    else:
        avg_values = {k: 0 for k in ['pm10', 'no2', 'o3', 'so2', 'co', 'openweather_aqi']}

    for i in range(3):
        # Prepare feature row for the next day
        features_df, updated_history = prepare_feature_row(current_input, updated_history)

        # Predict PM2.5
        pred_list = predict_from_model(model, features_df)
        if pred_list is None:  # Model missing → stop loop
            break

        pred = pred_list[0]
        preds.append(pred)

        # Prepare the next day's input
        current_input = {
            "datetime": (
                pd.to_datetime(current_input["datetime"]) + pd.Timedelta(days=1)
            ).strftime("%Y-%m-%d %H:%M:%S"),
            "pm2_5": pred,
            "pm10": avg_values['pm10'],  # Use average instead of 0
            "no2": avg_values['no2'],
            "o3": avg_values['o3'],
            "so2": avg_values['so2'],
            "co": avg_values['co'],
            "openweather_aqi": avg_values['openweather_aqi'],
        }

    return preds
