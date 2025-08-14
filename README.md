# Project readme
# AQI-PREDICTION — Multan

> **Forecast Multan’s Air Quality (next 3 days) with ML + Streamlit (serverless)**
>
> * Data: past 30 days via **Pandas** + future via **OpenWeather API**
> * Models: **Random Forest**, **Linear Regression**, **Stacked Ensemble** (saved as `.pkl`)
> * Explainability: **EDA** + **SHAP**
> * Infra: **Podman** containers, images stored on **GHCR**, deployed on **Streamlit Cloud**
> * CI/CD: **GitHub Actions** build & deploy pipeline

---

## ✨ Features

* 📊 **3‑day AQI forecast** for Multan with PM2.5 → AQI conversion
* 🔎 **EDA dashboard**: trends, distributions, correlations
* 🧠 **Explainability with SHAP**: global importance + per‑day contribution
* 🧱 **Reproducible containers** via Podman; images hosted on GHCR
* ☁️ **Serverless UI** on Streamlit Cloud
* 🔁 **CI/CD** with GitHub Actions

---

## 🗂️ Project structure (suggested)

```
AQI-PREDICTION/
├─ app/
│  └─ app.py                 # Streamlit app (UI + inference + SHAP)
├─ data/
│  ├─ raw_aqi_data.csv       # 30-day PM2.5 history for Multan
│  └─ processed/
│     └─ features.csv        # Engineered features (lags, rolling stats, dates)
├─ models/
│  └─ aqi_model/
│     └─ rf_model.pkl        # Trained model (RF or stacked)
├─ src/
│  ├─ feature_pipeline.py    # Build features.csv
│  ├─ train.py               # Train RF/LR + stacked model; save .pkl
│  └─ utils_stub.py          # pm25_to_aqi and helpers
├─ .github/workflows/
│  └─ ci-cd.yml              # Build, test, and deploy
├─ Dockerfile (or Containerfile)
├─ requirements.txt
├─ .env.example
└─ README.md
```

---

## 🧰 Requirements

* Python 3.10+
* Streamlit
* pandas, numpy, scikit-learn, matplotlib, shap, python-dotenv
* Podman (or Docker)

Install Python deps:

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment variables

Copy `.env.example` → `.env` and fill in values:

```
OPENWEATHER_API_KEY=your_api_key_here
CITY=Multan
LAT=30.1575
LON=71.5249
```

> Keep your API key secret. Don’t commit `.env`.

---

## 📥 Data acquisition

**Historical (past 30 days):**

* Load/clean PM2.5 & weather into `data/raw_aqi_data.csv` using Pandas

**Future (next 3 days):**

* Query OpenWeather endpoints for weather/pollution forecast

**Feature engineering:**

* Build `data/processed/features.csv` with:

  * Lags: `pm2_5_lag1..3`
  * Rolling means: `pm2_5_roll7`, `pm2_5_roll14`
  * Calendar: `dayofweek`, `day`, `month`

Example (pseudo):

```bash
python -m src.feature_pipeline --lat $LAT --lon $LON --days 30 \
  --out data/processed/features.csv
```

---

## 🧪 EDA (what to look at)

* **Distributions**: PM2.5/AQI histograms → typical ranges & outliers
* **Trends**: 30‑day PM2.5 moving averages; weekend vs weekday effects
* **Correlations**: heatmap across PM2.5, temp, humidity, wind → drivers
* **Conditioned plots**: AQI vs wind speed/dir; high temp + low wind → spikes

**Typical insights (Multan)**

* PM2.5 is the dominant driver of AQI
* Low wind & thermal inversions often precede higher AQI
* Humidity shows mixed correlation; context dependent

> Save plots to `reports/eda/` and show key ones in the Streamlit app.

---

## 🤖 Modeling

* **Algorithms**: RandomForestRegressor, LinearRegression, **Stacked**(RF+LR)
* **Targets**: next‑day PM2.5 → converted to AQI via `pm25_to_aqi`
* **Metrics**: MAE, RMSE, R² on held‑out recent days
* **Artifacts**: `.pkl` models under `models/aqi_model/`

Train & save:

```bash
python -m src.train \
  --features data/processed/features.csv \
  --out models/aqi_model/rf_model.pkl \
  --stacked true
```

> Want **no joblib**? Use Python `pickle` to load `.pkl` or retrain on startup.

---

## 🧩 Explainability with SHAP

* **Global importance**: which features generally push predictions up/down
* **Per‑prediction waterf all**: why today/tomorrow changed (lags vs weather)
* **Expected patterns**:

  * `pm2_5_lag1` and `pm2_5_roll7/14` → strongest for **T+1**
  * Weather/forecast features gain weight for **T+2/T+3**


## 📈 3‑day forecasting (how it works)

1. Read last PM2.5 values from `data/raw_aqi_data.csv`
2. Build features for T+1..T+3 (lags/rolling + date features + API forecasts)
3. Predict PM2.5 per day → convert with `pm25_to_aqi`
4. Render **cards** + **trend chart** in Streamlit

> Categories: GOOD ≤50, MODERATE ≤100, USG ≤150, UNHEALTHY ≤200, VERY UNHEALTHY ≤300, HAZARDOUS >300.

---

## 🖥️ Run locally

```bash
streamlit run app/app.py
```

* Ensure `data/raw_aqi_data.csv` exists
* Ensure `models/aqi_model/rf_model.pkl` exists (or app will use a naive fallback)

**Load model without joblib** (option):

```python
import pickle
with open("models/aqi_model/rf_model.pkl", "rb") as f:
    model = pickle.load(f)
```

**Or train on startup** (no serialization):

```python
from sklearn.ensemble import RandomForestRegressor
# fit model using features.csv each run
```

---

## 📦 Containerization with Podman

Build image:

```bash
podman build -t aqi-dev:latest .
```

Run locally:

```bash
podman run --rm -p 8501:8501 \
  --env-file .env \
  -v "$PWD/data:/app/data" \
  aqi-dev:latest
```

### Push to GHCR

Login:

```bash
export CR_PAT=YOUR_GITHUB_PAT
podman login ghcr.io -u <github-username> -p $CR_PAT
```

Tag & push:

```bash
podman tag aqi-dev:latest ghcr.io/<github-username>/aqi-dev:latest
podman push ghcr.io/<github-username>/aqi-dev:latest
```

> Use the image in Streamlit Cloud or other runners if desired.

---

## ☁️ Deploy on Streamlit Cloud (serverless)

1. Push repo to GitHub
2. Create new Streamlit app → point to `app/app.py`
3. Add secrets: `OPENWEATHER_API_KEY`, etc.
4. Set Python version & `requirements.txt`

> The app auto‑starts on commit; see CI/CD below for automation.

---

## 🚀 CI/CD with GitHub Actions

`.github/workflows/ci-cd.yml` (minimal example):

```yaml
name: CI/CD
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
jobs:
  build-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: python -m pytest || echo "no tests yet"
  container:
    needs: build-test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: |
          echo ${{ secrets.GHCR_PAT }} | podman login ghcr.io -u ${{ github.actor }} --password-stdin
          podman build -t ghcr.io/${{ github.repository_owner }}/aqi-dev:latest .
          podman push ghcr.io/${{ github.repository_owner }}/aqi-dev:latest
```

> Optionally add a job to ping Streamlit Cloud or use Streamlit’s deploy integration.

---

## 🐛 Troubleshooting

* **ModuleNotFoundError: joblib** → either install `joblib` or switch to `pickle`
* **Model missing** → ensure `models/aqi_model/rf_model.pkl` exists or enable train‑on‑startup
* **OpenWeather quota** → cache API responses; add retry/backoff
* **Timezones** → store timestamps in UTC; convert only for display

---

## 📚 References

* US EPA PM2.5 → AQI formula (used in `utils_stub.pm25_to_aqi`)
* OpenWeather API docs for weather & air pollution endpoints

---

## 📝 License

Choose a license (e.g., MIT) and place it in `LICENSE`.

---

## 🙌 Acknowledgements

* Inspiration & feedback from the data science community
* Open‑source maintainers of Streamlit, scikit‑learn, SHAP
