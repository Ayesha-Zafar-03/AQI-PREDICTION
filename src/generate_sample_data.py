import pandas as pd
import numpy as np
import os

os.makedirs("data/raw", exist_ok=True)

date_rng = pd.date_range(start='2025-08-01', periods=30, freq='3H')

pollutant_data = {
    "timestamp": date_rng,
    "pm2_5": np.random.randint(10, 100, size=30),
    "pm10": np.random.randint(20, 150, size=30),
    "no2": np.random.randint(5, 50, size=30),
    "o3": np.random.randint(3, 40, size=30),
    "so2": np.random.randint(1, 20, size=30),
    "co": np.random.uniform(0.1, 1.0, size=30).round(2),
}

weather_data = {
    "datetime": date_rng,
    "aqi": np.random.randint(50, 200, size=30),
}

pd.DataFrame(pollutant_data).to_csv("data/raw/multan_pollutants.csv", index=False)
pd.DataFrame(weather_data).to_csv("data/raw/multan_weather.csv", index=False)

print("Generated 30-row sample pollutant and weather CSVs.")
