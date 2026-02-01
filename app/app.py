import sys, os
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
import joblib
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import shap
import requests
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils_stub import pm25_to_aqi

load_dotenv()

st.set_page_config(layout="wide", page_title="AQI Predictor - Multan")

# ------------------ Cute dashboard CSS ------------------
background_image = "https://tse3.mm.bing.net/th/id/OIP._DsO70BYi98bsuY-UmV9EgHaHa?rs=1&pid=ImgDetMain&o=7&rm=3"
st.markdown(
    f"""
    <style>
    /* Full-page background */
    .stApp {{
        background-image: url("{background_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    /* Header */
    .header {{
        background: #fdf5df;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-family: 'Arial', sans-serif;
        font-weight: bold;
        font-size: 28px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }}

    /* Main cards */
    .card {{
        background: linear-gradient(#a0aecd, #6bb77b, #F2BFA4);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        color: white;
        font-family: 'Arial', sans-serif;
        transition: transform 0.2s ease;
    }}
    .card:hover {{
        transform: scale(1.03);
    }}

    /* Forecast cards */
    .pred {{
        background: linear-gradient(#394f8a, #6bb77b, #F5E7DE);
        text-align: center;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        color: white;
        font-family: 'Arial', sans-serif;
        transition: transform 0.2s ease;
    }}
    .pred:hover {{
        transform: scale(1.05);
    }}

    /* AQI numbers */
    .aqi-num {{
        font-size: 64px;
        font-weight: 800;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }}

    /* Muted text */
    .small-muted {{
        color: white;
        font-weight: 500;
    }}

    h3 {{
        font-family: 'Arial', sans-serif;
        font-weight: bold;
        color: #F5E7DE;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ Helper functions ------------------
def category_and_style(aqi_value):
    if aqi_value <= 50: return "GOOD", "😊", "#009966"
    elif aqi_value <= 100: return "MODERATE", "😐", "#FFDE33"
    elif aqi_value <= 150: return "UNHEALTHY (SG)", "😷", "#FF9933"
    elif aqi_value <= 200: return "UNHEALTHY", "🤒", "#CC0033"
    elif aqi_value <= 300: return "VERY UNHEALTHY", "☠️", "#660099"
    else: return "HAZARDOUS", "☠️", "#7E0023"

def calculate_fallback_prediction(lag1, lag2, lag3, roll7, roll14, weather, day_index):
    """Improved fallback prediction algorithm - DETERMINISTIC"""
    # Weighted moving average with emphasis on recent data
    weights = [0.50, 0.30, 0.20]
    base_pred = (lag1 * weights[0] + lag2 * weights[1] + lag3 * weights[2])
    
    # Blend with rolling averages (prevents drift)
    base_pred = base_pred * 0.7 + roll7 * 0.3
    
    # Trend analysis
    if roll7 > roll14:
        trend_factor = 1.02  # Slight increase
    elif roll7 < roll14:
        trend_factor = 0.98  # Slight decrease
    else:
        trend_factor = 1.0
    
    base_pred *= trend_factor
    
    # Weather adjustments (conservative and deterministic)
    if weather['wind_speed'] < 2:
        base_pred *= 1.05  # Low wind
    elif weather['wind_speed'] > 5:
        base_pred *= 0.95  # High wind
    
    # Dampening for future days (prevents error amplification)
    dampening = 1.0 - (day_index * 0.03)
    base_pred *= dampening
    
    return base_pred

# ------------------ Fetch fresh PM2.5 data ------------------
@st.cache_data(ttl=3600)
def fetch_current_pm25():
    """Fetch current PM2.5 from OpenWeather API with fallback to CSV"""
    api_key = os.getenv('OPENWEATHER_API_KEY')
    lat = float(os.getenv('LAT', '30.1575'))
    lon = float(os.getenv('LON', '71.5249'))
    
    try:
        url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        current_pm25 = data['list'][0]['components']['pm2_5']
        timestamp = datetime.utcnow()
        
        return current_pm25, timestamp, "API"
        
    except Exception as e:
        RAW_FILE = "data/raw_aqi_data.csv"
        if os.path.exists(RAW_FILE):
            raw_df = pd.read_csv(RAW_FILE)
            return raw_df['pm2_5'].iloc[-1], None, "CSV"
        return 100, None, "DEFAULT"

# ------------------ Fetch weather forecast ------------------
@st.cache_data(ttl=3600)
def fetch_weather_forecast():
    """Fetch 3-day weather forecast from OpenWeather API"""
    api_key = os.getenv('OPENWEATHER_API_KEY')
    lat = float(os.getenv('LAT', '30.1575'))
    lon = float(os.getenv('LON', '71.5249'))
    
    try:
        url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        forecasts = []
        for i in range(0, min(24, len(data['list'])), 8):
            day_data = data['list'][i]
            forecasts.append({
                'temp': day_data['main']['temp'],
                'humidity': day_data['main']['humidity'],
                'wind_speed': day_data['wind']['speed'],
                'pressure': day_data['main']['pressure']
            })
        return forecasts[:3]
        
    except Exception as e:
        # Default weather values
        return [
            {'temp': 25, 'humidity': 60, 'wind_speed': 3, 'pressure': 1013},
            {'temp': 26, 'humidity': 58, 'wind_speed': 3.5, 'pressure': 1012},
            {'temp': 25, 'humidity': 62, 'wind_speed': 2.8, 'pressure': 1014}
        ]

# ------------------ Load model ------------------
@st.cache_resource
def load_model():
    """Load trained models with caching"""
    MODEL_DIR = "model/aqi_model"
    MODEL_FILE = "rf_model.pkl"
    path = os.path.join(MODEL_DIR, MODEL_FILE)
    
    if os.path.exists(path):
        try:
            model = joblib.load(path)
            return model, True
        except Exception as e:
            st.warning(f"⚠️ Model loading error: {e}")
            return None, False
    else:
        return None, False

# ------------------ Header ------------------
st.markdown("<div class='header'><h2>AQI Predictor — Multan</h2></div>", unsafe_allow_html=True)

# Fetch current data
last_pm25, fetch_time, data_source = fetch_current_pm25()

if data_source == "API":
    st.success(f"✅ Live data fetched at {fetch_time.strftime('%H:%M:%S UTC')}")
   

# Load historical data
RAW_FILE = "data/raw_aqi_data.csv"
if os.path.exists(RAW_FILE):
    raw_df = pd.read_csv(RAW_FILE)
    raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])
else:
    # Create minimal historical data if file missing
    raw_df = pd.DataFrame({
        'timestamp': [datetime.utcnow() - timedelta(days=i) for i in range(30, 0, -1)],
        'pm2_5': [last_pm25 + (i % 20 - 10) for i in range(30)]
    })

col1, col2 = st.columns([1,2])

# ------------------ Left column: current PM2.5 ------------------
with col1:
    st.markdown(
        f"<div class='card' style='text-align:center'>"
        f"<div style='font-size:20px'>PM2.5</div>"
        f"<div class='aqi-num'>{last_pm25:.1f}</div>"
        "<div class='small-muted'>μg/m³</div></div>",
        unsafe_allow_html=True
    )

# ------------------ Right column: current AQI ------------------
today_aqi = pm25_to_aqi(last_pm25)
cat_today, emo_today, color_today = category_and_style(today_aqi)

with col2:
    st.markdown(
        f"<div class='card'><h3>TODAY — {datetime.utcnow().strftime('%a %d %b %Y')}</h3>"
        f"<div style='display:flex; gap:20px; align-items:center'>"
        f"<div style='flex:1'><div class='aqi-num' style='color:{color_today}'>{today_aqi}</div>"
        f"<div class='small-muted' style='color:{color_today}'>{cat_today} {emo_today}</div></div>"
        f"<div style='flex:2'><p>Dominant pollutant: PM2.5</p></div>"
        f"</div></div>",
        unsafe_allow_html=True
    )

# ------------------ Load model ------------------
model, model_loaded = load_model()

if model_loaded:
    st.success(f"✅ Model loaded successfully")


# ------------------ Fetch weather forecast ------------------
weather_forecasts = fetch_weather_forecast()

# ------------------ IMPROVED 3-day forecast (FIXED ALGORITHM) ------------------
pm25_history = raw_df['pm2_5'].tolist() if len(raw_df) > 0 else [last_pm25]

# Update with current value if from API
if data_source == "API" and len(pm25_history) > 0:
    pm25_history[-1] = last_pm25

preds_pm25 = []
preds_aqi = []

# Create a copy of history for iteration (don't pollute it with predictions)
historical_pm25 = pm25_history.copy()

for i in range(3):
    # CRITICAL FIX: Use ACTUAL historical data for lags, not predictions!
    # This prevents error amplification across days
    
    if i == 0:
        # Day 1: Use only actual history
        lag1 = historical_pm25[-1]
        lag2 = historical_pm25[-2] if len(historical_pm25) >= 2 else lag1
        lag3 = historical_pm25[-3] if len(historical_pm25) >= 3 else lag1
    elif i == 1:
        # Day 2: Use 1 prediction + 2 historical
        lag1 = preds_pm25[0]  # Day 1 prediction
        lag2 = historical_pm25[-1]  # Actual last day
        lag3 = historical_pm25[-2] if len(historical_pm25) >= 2 else historical_pm25[-1]
    else:  # i == 2
        # Day 3: Use 2 predictions + 1 historical
        lag1 = preds_pm25[1]  # Day 2 prediction
        lag2 = preds_pm25[0]  # Day 1 prediction
        lag3 = historical_pm25[-1]  # Actual last day
    
    # Calculate rolling averages from ACTUAL history only
    recent_7 = historical_pm25[-7:] if len(historical_pm25) >= 7 else historical_pm25
    recent_14 = historical_pm25[-14:] if len(historical_pm25) >= 14 else historical_pm25
    roll7 = sum(recent_7) / len(recent_7) if recent_7 else lag1
    roll14 = sum(recent_14) / len(recent_14) if recent_14 else lag1

    # Get weather forecast
    weather = weather_forecasts[i] if i < len(weather_forecasts) else weather_forecasts[-1]
    
    # Build feature row
    future_date = datetime.utcnow() + timedelta(days=i + 1)
    row = pd.DataFrame({
        "pm2_5_lag1": [lag1],
        "pm2_5_lag2": [lag2],
        "pm2_5_lag3": [lag3],
        "pm2_5_roll7": [roll7],
        "pm2_5_roll14": [roll14],
        "dayofweek": [future_date.weekday()],
        "day": [future_date.day],
        "month": [future_date.month],
    })

    if model:
        try:
            pred_pm25_raw = model.predict(row.values)[0]
            
            # Apply dampening for future days to prevent error amplification
            dampening = 1.0 - (i * 0.05)  # Day 1: 100%, Day 2: 95%, Day 3: 90%
            pred_pm25 = pred_pm25_raw * dampening
            
            # Constrain predictions to be within reasonable range of current
            max_change = 30 + (i * 10)  # Day 1: ±30, Day 2: ±40, Day 3: ±50
            pred_pm25 = np.clip(pred_pm25, last_pm25 - max_change, last_pm25 + max_change)
            
        except Exception as e:
            # Fallback with improved algorithm
            pred_pm25 = calculate_fallback_prediction(lag1, lag2, lag3, roll7, roll14, weather, i)
    else:
        # Use deterministic fallback
        pred_pm25 = calculate_fallback_prediction(lag1, lag2, lag3, roll7, roll14, weather, i)
    
    # Ensure realistic bounds
    pred_pm25 = max(10, min(400, pred_pm25))

    preds_pm25.append(pred_pm25)
    preds_aqi.append(pm25_to_aqi(pred_pm25))

# ------------------ Forecast cards ------------------
st.markdown("<h3 style='margin-top:20px'>Forecast (next 3 days)</h3>", unsafe_allow_html=True)
cols = st.columns(3)
dates = [(datetime.utcnow() + timedelta(days=i + 1)).strftime("%a %d %b") for i in range(3)]

for c, d, pm, aqi, weather in zip(cols, dates, preds_pm25, preds_aqi, weather_forecasts):
    cat, emo, color = category_and_style(aqi)
    c.markdown(
        f"<div class='pred card'>"
        f"<h4>{d}</h4>"
        f"<div>PM2.5: {pm:.1f} μg/m³</div>"
        f"<div style='font-size:36px;font-weight:700;color:{color}'>{aqi}</div>"
        f"<div style='color:{color}'>{cat} {emo}</div>"
        f"<div style='font-size:12px;margin-top:10px'>"
        f"🌡️ {weather['temp']:.1f}°C | 💨 {weather['wind_speed']:.1f} m/s"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True
    )

# ------------------ Trend chart ------------------
fig, ax1 = plt.subplots(figsize=(8,4), facecolor='none')
ax1.plot(dates, preds_pm25, color='#F2BFA4', marker='o', linewidth=3, label='PM2.5')
ax1.set_xlabel('Date', fontsize=12, fontweight='bold', color='white')
ax1.set_ylabel('PM2.5 (µg/m³)', color='#F2BFA4', fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor='#F2BFA4', labelsize=10)
ax1.tick_params(axis='x', labelcolor='white', labelsize=10)

ax2 = ax1.twinx()
ax2.plot(dates, preds_aqi, color='#F5E7DE', marker='s', linewidth=3, label='AQI')
ax2.set_ylabel('AQI', color='#F5E7DE', fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor='#F5E7DE', labelsize=10)

ax1.grid(alpha=0.3)
fig.tight_layout()
st.markdown("<h3 style='margin-top:20px'>PM2.5 & AQI Trend</h3>", unsafe_allow_html=True)
st.pyplot(fig)
plt.close(fig)

# ------------------ SHAP explanations ------------------
if model:
    try:
        explainer = shap.TreeExplainer(model)
        shap.initjs()
        
        last_row = pd.DataFrame({
            "pm2_5_lag1": [historical_pm25[-1]],
            "pm2_5_lag2": [historical_pm25[-2] if len(historical_pm25) >= 2 else historical_pm25[-1]],
            "pm2_5_lag3": [historical_pm25[-3] if len(historical_pm25) >= 3 else historical_pm25[-1]],
            "pm2_5_roll7": [sum(historical_pm25[-7:])/min(7, len(historical_pm25))],
            "pm2_5_roll14": [sum(historical_pm25[-14:])/min(14, len(historical_pm25))],
            "dayofweek": [datetime.utcnow().weekday()],
            "day": [datetime.utcnow().day],
            "month": [datetime.utcnow().month],
        })
        shap_values = explainer.shap_values(last_row)

        st.markdown("<h3 style='margin-top:20px'>Global Feature Importance</h3>", unsafe_allow_html=True)
        fig_shap, ax = plt.subplots(figsize=(6,4), facecolor='none')
        shap.summary_plot(shap_values, last_row, plot_type="bar", show=False)
        st.pyplot(fig_shap)
        plt.close(fig_shap)

        st.markdown("<h3 style='margin-top:20px'>Feature Contribution per Forecast Day</h3>", unsafe_allow_html=True)
        for i in range(3):
            future_date = datetime.utcnow() + timedelta(days=i + 1)
            
            # Build same lag structure as prediction
            if i == 0:
                lag1 = historical_pm25[-1]
                lag2 = historical_pm25[-2] if len(historical_pm25) >= 2 else lag1
                lag3 = historical_pm25[-3] if len(historical_pm25) >= 3 else lag1
            elif i == 1:
                lag1 = preds_pm25[0]
                lag2 = historical_pm25[-1]
                lag3 = historical_pm25[-2] if len(historical_pm25) >= 2 else historical_pm25[-1]
            else:
                lag1 = preds_pm25[1]
                lag2 = preds_pm25[0]
                lag3 = historical_pm25[-1]
            
            row_day = pd.DataFrame({
                "pm2_5_lag1": [lag1],
                "pm2_5_lag2": [lag2],
                "pm2_5_lag3": [lag3],
                "pm2_5_roll7": [sum(historical_pm25[-7:])/min(7, len(historical_pm25))],
                "pm2_5_roll14": [sum(historical_pm25[-14:])/min(14, len(historical_pm25))],
                "dayofweek": [future_date.weekday()],
                "day": [future_date.day],
                "month": [future_date.month],
            })
            shap_values_day = explainer.shap_values(row_day)
            st.markdown(f"**{future_date.strftime('%a %d %b')} Forecast**")
            fig_wf, ax = plt.subplots(figsize=(6,4), facecolor='none')
            shap.plots._waterfall.waterfall_legacy(
                explainer.expected_value[0] if hasattr(explainer.expected_value, '__len__') else explainer.expected_value,
                shap_values_day[0],
                row_day.iloc[0],
                show=False
            )
            st.pyplot(fig_wf)
            plt.close(fig_wf)
    except Exception as e:
        st.warning(f"⚠️ SHAP explanations unavailable: {e}")

# ------------------ Footer info ------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; color:white; font-size:12px'>
    <p>💡Powered By Ayesha Zafar </p>
    </div>
    """,
    unsafe_allow_html=True
)
