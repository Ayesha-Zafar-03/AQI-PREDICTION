# AQI-PREDICTION-MULTAN

![Python](https://img.shields.io/badge/python-3.10+-blue)
![Docker](https://img.shields.io/badge/docker-GHCR-brightgreen)
![CI/CD](https://github.com/Ayesha-Zafar-03/AQI-PREDICTION/actions/workflows/update_aqi_data.yml/badge.svg)

Air Quality Index (AQI) prediction for Multan using **Machine Learning**, **Docker**, and **CI/CD**.  
Fetches live pollutant data, predicts future AQI, and updates CSV data automatically.

---

## 📌 Project Overview

- Predicts **AQI** using historical pollutants:
  - PM2.5, PM10, NO2, SO2, O3, CO
- Trained ML model stored at `models/aqi-model/model.pkl`
- Docker image built & pushed to **GitHub Container Registry (GHCR)**
- CI/CD workflow updates `data/raw-aqi-data.csv` every 6 hours
- Streamlit app for **interactive visualization** & predictions

---

## 🗂 Repository Structure

```

AQI-PREDICTION/
├─ app/
│  ├─ app.py                # Streamlit interface + inference
│  └─ daily_updater.py      # Fetches latest AQI data
├─ data/
│  └─ raw-aqi-data.csv      # Historical and updated AQI data
├─ models/
│  └─ aqi-model/
│     └─ model.pkl          # Trained ML model
├─ src/
│  ├─ **init**.py
│  └─ utils_stub.py         # Utility functions
├─ Dockerfile               # Build Docker image
├─ requirements.txt
├─ train_model.py           # Train and save ML model
├─ .github/workflows/
│  └─ update_aqi_data.yml  # CI/CD workflow
├─ .dockerignore
├─ .gitignore
└─ README.md

````

---

## 🐳 Docker & GHCR

Docker image stored in **GHCR**.

### Build Docker image locally:

```bash
docker build -t ghcr.io/<your-username>/aqi-updater:latest .
````

### Run Docker container:

```bash
docker run -p 8501:8501 ghcr.io/<your-username>/aqi-updater:latest
```

Streamlit app will be available at `http://localhost:8501`.

---

## ⚙️ CI/CD Pipeline

* Workflow: `.github/workflows/update_aqi_data.yml`
* Runs every **6 hours** (`cron: '0 */6 * * *'`)
* Fetches latest pollutant data via API
* Updates `data/raw-aqi-data.csv` automatically
* Ensures ML predictions always use **up-to-date data**

---

## 🧠 Model Training

Train the ML model:

```bash
python train_model.py
```

* Saves trained model at `models/aqi-model/model.pkl`
* Model is used by Streamlit app for predictions

---

## 🏃‍♂️ Run Locally

1. Clone repository:

```bash
git clone https://github.com/Ayesha-Zafar-03/AQI-PREDICTION.git
cd AQI-PREDICTION
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run Streamlit app:

```bash
streamlit run app/app.py
```

---

## 📊 Data & Features

* Pollutants: **PM2.5, PM10, NO2, SO2, O3, CO**
* Timestamped data every **6 hours**
* Features include lag values, rolling averages, and weather parameters
* Target: **AQI**

---

## 🔮 Future Improvements

* Multi-city AQI prediction
* Real-time backend API deployment
* Additional pollutant and meteorological features
* Model explainability (SHAP, feature importance)

---


