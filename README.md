# 🌫️ AQI-PREDICTION — Multan Air Quality Forecast

> A Machine Learning-powered Air Quality Index (AQI) predictor for Multan, providing live updates, 3-day forecasts, and interactive visualizations using Streamlit.

---

## 🚀 Project Overview

AQI-PREDICTION is a predictive system that forecasts air quality for Multan using **historical PM2.5 data**, weather forecasts, and a trained machine learning model. The project features:

- Real-time AQI updates via OpenWeather API.
- 3-day forecast for PM2.5 and AQI.
- Interactive visualizations of trends and predictions.
- Feature importance explanations using **SHAP**.
- Containerized deployment with **Docker/Podman**.

---


## 🔗 Live Demo

Try the live deployed app here:

👉 https://aqi‑prediction‑hmnwsdviqmzbdcsdvrbpnt.streamlit.app/

---


## 🗂️ Project Structure

```

AQI-PREDICTION/
├─ app/
│  ├─ app.py                 # Main Streamlit application
│  └─ daily_updater.py       # Script to update daily AQI data
├─ data/
│  └─ raw_aqi_data.csv       # Historical PM2.5 data
├─ src/
│  ├─ **init**.py
│  └─ utils_stub.py          # Helper functions (e.g., PM2.5 → AQI conversion)
├─ static/                   # Optional static assets (images, CSS)
├─ models/
│  └─ aqi_model/
│     └─ rf_model.pkl        # Trained Random Forest model
├─ Dockerfile
├─ run_podman.ps1
├─ requirements.txt
└─ README.md

````

---

## 🧠 How It Works

1. **Data Collection**
   - Historical PM2.5 values are stored in `data/raw_aqi_data.csv`.
   - Current PM2.5 and 3-day weather forecast are fetched via OpenWeather API.

2. **Feature Engineering**
   - Lag values (1–3 days) and rolling averages (7 & 14 days) are computed.
   - Calendar features like day, month, and day-of-week are added.

3. **Prediction**
   - The Random Forest model predicts PM2.5 for the next 3 days.
   - PM2.5 values are converted to AQI categories (Good, Moderate, Unhealthy, etc.).
   - Fallback deterministic algorithm ensures robust predictions if the model fails.

4. **Visualization**
   - Streamlit dashboard displays:
     - Current PM2.5 & AQI
     - 3-day forecast cards with weather info
     - PM2.5 & AQI trend chart
     - SHAP feature importance and contribution plots

---

## 🛠️ Installation

### Clone the repository

```bash
git clone https://github.com/Ayesha-Zafar-03/AQI-PREDICTION.git
cd AQI-PREDICTION
````

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🐍 Run the App

```bash
streamlit run app/app.py
```

* The app opens in your browser.
* Current PM2.5 and AQI are displayed.
* Interactive 3-day forecast and trend charts are shown.
* SHAP explains feature contributions.

---

## 🐳 Containerized Deployment

Build the Docker/Podman image:

```bash
podman build -t aqi-prediction .
```

Run:

```bash
podman run --rm -p 8501:8501 aqi-prediction
```

---

## 🔑 Environment Variables

Create a `.env` file with:

```
OPENWEATHER_API_KEY=your_api_key_here
LAT=30.1575
LON=71.5249
```

* `OPENWEATHER_API_KEY` – API key for OpenWeather.
* `LAT`, `LON` – Coordinates for Multan.

---

## 📊 Screenshots

![Dashboard Preview](./f3955c23-bcee-446b-ae11-188e478807e8.png)

---

## 💡 Contributing

Contributions are welcome! You can:

* Improve the ML model accuracy.
* Add more visualization features.
* Extend to other cities.

---

## 📄 License

This project is open-source. See `LICENSE` for details.

---

## ❤️ Acknowledgements

* OpenWeather API for real-time weather & air quality data.
* SHAP library for model explainability.
* Streamlit for building the interactive dashboard.

