# app/app.py
import sys, os, random
from datetime import datetime, timedelta
from datetime import datetime
import pandas as pd
import streamlit as st
import joblib
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils_stub import pm25_to_aqi

load_dotenv()

# ------------------ Page setup ------------------
st.set_page_config(layout="wide", page_title="AQI Predictor - Multan")

st.markdown("""
<style>
.header { background:#0b5fbd; color: white; padding:12px; border-radius:6px; }
.card { background: #ffffff; padding:18px; border-radius:8px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }
.aqi-num { font-size:64px; font-weight:700; }
.small-muted { color:#666; }
.pred { text-align:center; padding:12px; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

# ------------------ AQI category ------------------
def category_and_style(aqi_value):

    if aqi_value <= 50: return "GOOD", "😊", "#009966"
    elif aqi_value <= 100: return "MODERATE", "😐", "#FFDE33"
    elif aqi_value <= 150: return "UNHEALTHY (SG)", "😷", "#FF9933"
    elif aqi_value <= 200: return "UNHEALTHY", "🤒", "#CC0033"
    elif aqi_value <= 300: return "VERY UNHEALTHY", "☠️", "#660099"
    else: return "HAZARDOUS", "☠️", "#7E0023"

st.markdown("<div class='header'><h2>AQI Predictor — Multan</h2></div>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 2])

# ------------------ Load raw data ------------------
RAW_FILE = "data/raw_aqi_data.csv"
FEATURE_FILE = "data/processed/features.csv"

if not os.path.exists(RAW_FILE):
    st.error(f"Raw data file not found: {RAW_FILE}")
    st.stop()

raw_df = pd.read_csv(RAW_FILE)
raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])
last_pm25 = raw_df['pm2_5'].iloc[-1]

# ------------------ Left column: current PM2.5 ------------------
with col1:
    st.markdown(
        f"<div class='card' style='text-align:center'>"
        f"<div style='font-size:20px'>PM2.5</div>"
        f"<div class='aqi-num'>{last_pm25}</div>"
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
MODEL_DIR = "models/aqi_model"
MODEL_FILE = "rf_model.pkl"

def load_model():
    path = os.path.join(MODEL_DIR, MODEL_FILE)
    if os.path.exists(path):
        model = joblib.load(path)
        st.success(f"Loaded model: {MODEL_FILE}")
        return model
    else:
        st.info("No model found — using naive fallback")
        return None

model = load_model()

# ------------------ 3-day forecast ------------------
preds_pm25 = []
preds_aqi = []
pm25_history = raw_df['pm2_5'].tolist()

for i in range(3):
    lag1 = pm25_history[-1]
    lag2 = pm25_history[-2] if len(pm25_history) >= 2 else lag1
    lag3 = pm25_history[-3] if len(pm25_history) >= 3 else lag1
    roll7 = sum(pm25_history[-7:]) / min(len(pm25_history), 7)
    roll14 = sum(pm25_history[-14:]) / min(len(pm25_history), 14)

    row = pd.DataFrame({
        "pm2_5_lag1": [lag1],
        "pm2_5_lag2": [lag2],
        "pm2_5_lag3": [lag3],
        "pm2_5_roll7": [roll7],
        "pm2_5_roll14": [roll14],
        "dayofweek": [(datetime.utcnow() + timedelta(days=i + 1)).weekday()],
        "day": [(datetime.utcnow() + timedelta(days=i + 1)).day],
        "month": [(datetime.utcnow() + timedelta(days=i + 1)).month],
    })

    if model:
        pred_pm25 = model.predict(row.values)[0]
    else:
        pred_pm25 = max(0, lag1 + random.choice([-5, 0, 5]))

    preds_pm25.append(pred_pm25)
    preds_aqi.append(pm25_to_aqi(pred_pm25))
    pm25_history.append(pred_pm25)

# ------------------ Display forecast ------------------
st.markdown("<h3 style='margin-top:20px'>Forecast (next 3 days)</h3>", unsafe_allow_html=True)
cols = st.columns(3)
dates = [(datetime.utcnow() + timedelta(days=i + 1)).strftime("%a %d %b") for i in range(3)]

for c, d, pm, aqi in zip(cols, dates, preds_pm25, preds_aqi):
    cat, emo, color = category_and_style(aqi)
    c.markdown(
        f"<div class='pred card'>"
        f"<h4>{d}</h4>"
        f"<div>PM2.5: {pm:.1f} μg/m³</div>"
        f"<div style='font-size:36px;font-weight:700;color:{color}'>AQI: {aqi}</div>"
        f"<div style='color:{color}'>{cat} {emo}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

# ------------------ Trend chart ------------------
import matplotlib.pyplot as plt

dates_trend = [datetime.utcnow()] + [(datetime.utcnow() + timedelta(days=i+1)) for i in range(3)]
dates_labels = [d.strftime("%a %d %b") for d in dates_trend]
pm25_values = [last_pm25] + preds_pm25
aqi_values = [today_aqi] + preds_aqi

fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(dates_labels, pm25_values, color='tab:blue', marker='o', label='PM2.5 (µg/m³)')
ax1.set_xlabel('Date')
ax1.set_ylabel('PM2.5 (µg/m³)', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.plot(dates_labels, aqi_values, color='tab:red', marker='s', label='AQI')
ax2.set_ylabel('AQI', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

fig.tight_layout()
st.markdown("<h3 style='margin-top:20px'>PM2.5 & AQI Trend</h3>", unsafe_allow_html=True)
st.pyplot(fig)
