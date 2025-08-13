# app/app.py
import sys, os, random
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
import joblib
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import shap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils_stub import pm25_to_aqi

load_dotenv()

st.set_page_config(layout="wide", page_title="AQI Predictor - Multan")

# ------------------ Cute dashboard CSS ------------------
background_image = "https://tse3.mm.bing.net/th/id/OIP._DsO70BYi98bsuY-UmV9EgHaHa?rs=1&pid=ImgDetMain&o=7&rm=3"  # Your blue background image
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
        background: #fdf5df;  /* semi-transparent blue */
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

# ------------------ Header ------------------
st.markdown("<div class='header'><h2>AQI Predictor — Multan</h2></div>", unsafe_allow_html=True)
col1, col2 = st.columns([1,2])

# ------------------ Load raw data ------------------
RAW_FILE = "data/raw_aqi_data.csv"
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

# ------------------ Forecast cards ------------------
st.markdown("<h3 style='margin-top:20px'>Forecast (next 3 days)</h3>", unsafe_allow_html=True)
cols = st.columns(3)
dates = [(datetime.utcnow() + timedelta(days=i + 1)).strftime("%a %d %b") for i in range(3)]
for c, d, pm, aqi in zip(cols, dates, preds_pm25, preds_aqi):
    cat, emo, color = category_and_style(aqi)
    c.markdown(
        f"<div class='pred card'>"
        f"<h4>{d}</h4>"
        f"<div>PM2.5: {pm:.1f} μg/m³</div>"
        f"<div style='font-size:36px;font-weight:700;color:{color}'>{aqi}</div>"
        f"<div style='color:{color}'>{cat} {emo}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

# ------------------ Trend chart ------------------
fig, ax1 = plt.subplots(figsize=(8,4), facecolor='none')  # transparent
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
    explainer = shap.TreeExplainer(model)
    shap.initjs()
    last_row = pd.DataFrame({
        "pm2_5_lag1": [pm25_history[-1]],
        "pm2_5_lag2": [pm25_history[-2]],
        "pm2_5_lag3": [pm25_history[-3]],
        "pm2_5_roll7": [sum(pm25_history[-7:])/7],
        "pm2_5_roll14": [sum(pm25_history[-14:])/14],
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
        row_day = pd.DataFrame({
            "pm2_5_lag1": [pm25_history[-1]],
            "pm2_5_lag2": [pm25_history[-2]],
            "pm2_5_lag3": [pm25_history[-3]],
            "pm2_5_roll7": [sum(pm25_history[-7:])/7],
            "pm2_5_roll14": [sum(pm25_history[-14:])/14],
            "dayofweek": [(datetime.utcnow() + timedelta(days=i + 1)).weekday()],
            "day": [(datetime.utcnow() + timedelta(days=i + 1)).day],
            "month": [(datetime.utcnow() + timedelta(days=i + 1)).month],
        })
        shap_values_day = explainer.shap_values(row_day)
        st.markdown(f"**{(datetime.utcnow() + timedelta(days=i+1)).strftime('%a %d %b')} Forecast**")
        fig_wf, ax = plt.subplots(figsize=(6,4), facecolor='none')
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value[0],
            shap_values_day[0],
            row_day.iloc[0],
            show=False
        )
        st.pyplot(fig_wf)
        plt.close(fig_wf)
