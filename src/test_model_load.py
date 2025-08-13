# test_model_load.py
import joblib
import os

model_path = "C:/Users/PMLS/Documents/AQI/models/aqi_model/rf_model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print("Model loaded successfully:", model)
else:
    print("Model file not found:", model_path)