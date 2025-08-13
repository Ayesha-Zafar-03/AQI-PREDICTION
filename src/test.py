import requests
import pandas as pd
from datetime import datetime, timedelta

LAT = 30.1575
LON = 71.5249

def fetch_openaq_data(lat, lon, days=30):
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    params = {
        "coordinates": f"{lat},{lon}",
        "radius": 20000,  # 20 km radius
        "date_from": start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "date_to": end_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": 10000,
        "order_by": "datetime",
        "sort": "asc",
        "parameter": ["pm25", "pm10", "no2", "o3", "so2", "co"]
    }

    response = requests.get("https://api.openaq.org/v2/measurements", params=params)
    data = response.json()

    records = []
    for item in data.get("results", []):
        records.append({
            "timestamp": item["date"]["utc"],
            "parameter": item["parameter"],
            "value": item["value"]
        })

    if not records:
        print("No data found.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df_pivot = df.pivot_table(index="timestamp", columns="parameter", values="value", aggfunc="mean").reset_index()

    # Convert timestamp to datetime
    df_pivot["timestamp"] = pd.to_datetime(df_pivot["timestamp"])

    # Fill missing pollutant columns with 0
    for col in ["pm25", "pm10", "no2", "o3", "so2", "co"]:
        if col not in df_pivot.columns:
            df_pivot[col] = 0
        else:
            df_pivot[col] = df_pivot[col].fillna(0)

    # Rename pm25 -> pm2_5 for your feature pipeline compatibility
    df_pivot = df_pivot.rename(columns={"pm25": "pm2_5"})

    # Save raw data CSV
    df_pivot.to_csv("data/raw/raw_aqi_data.csv", index=False)
    print("Saved raw AQI data to data/raw_aqi_data.csv")

    return df_pivot

if __name__ == "__main__":
    fetch_openaq_data(LAT, LON, days=30)
