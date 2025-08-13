import requests
import pandas as pd
from datetime import datetime, timedelta
import os

# ------------------- Constants -------------------
LAT = 30.1575
LON = 71.5249
OPENWEATHER_API_KEY = "c14419919ea7db36dfbed4831d2d4a5c"  # Replace with your real key
RAW_AQI_FILE = "data/raw_aqi_data.csv"  # Path to your 30-day AQI CSV file

# ------------------- Functions -------------------


def get_historical_openweather(days=14):
    import requests
    api_key = os.getenv("OPENWEATHER_API_KEY")
    historical_data = []
    for i in range(days, 0, -1):
        timestamp = int((datetime.utcnow() - timedelta(days=i)).timestamp())
        url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat=30.1575&lon=71.5249&start={timestamp}&end={timestamp}&appid={api_key}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            pm25 = data.get("list", [{}])[0].get("components", {}).get("pm2_5")
            if pm25 is not None and pm25 >= 0:
                historical_data.append(pm25)
        except Exception as e:
            print(f"Error fetching historical data for day -{i}: {e}")
    return historical_data
def load_historical_aqi(file_path):
    """
    Load last 30 days AQI data from CSV with 'timestamp' column.
    """
    try:
        df = pd.read_csv(file_path, parse_dates=["timestamp"])
        df.rename(columns={"timestamp": "datetime"}, inplace=True)  # standardize name
        print(f"Loaded {len(df)} rows of historical AQI data from {file_path}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found at: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading historical AQI data: {e}")

def get_current_openweather():
    """
    Fetch current AQI and pollutant data from OpenWeather.
    """
    url = "https://api.openweathermap.org/data/2.5/air_pollution"
    params = {"lat": LAT, "lon": LON, "appid": OPENWEATHER_API_KEY}
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()

    if "list" not in data or len(data["list"]) == 0:
        raise Exception("No air pollution data found")

    current = data["list"][0]
    components = current.get("components", {})
    aqi = current.get("main", {}).get("aqi")

    return {
        "datetime": datetime.now(),
        "pm2_5": components.get("pm2_5"),
        "pm10": components.get("pm10"),
        "no2": components.get("no2"),
        "o3": components.get("o3"),
        "so2": components.get("so2"),
        "co": components.get("co"),
        "openweather_aqi": aqi,
    }

def fetch_weather_forecast(lat, lon, api_key):
    """
    Fetch weather forecast data (temperature, humidity, wind, etc) from OpenWeather.
    """
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "metric"
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    records = []
    for item in data.get("list", []):
        main = item.get("main", {})
        wind = item.get("wind", {})
        records.append({
            "datetime": item.get("dt_txt"),
            "temp": main.get("temp"),
            "temp_min": main.get("temp_min"),
            "temp_max": main.get("temp_max"),
            "humidity": main.get("humidity"),
            "pressure": main.get("pressure"),
            "wind_speed": wind.get("speed"),
            "wind_deg": wind.get("deg")
        })

    return pd.DataFrame(records)

# ------------------- Main -------------------
if __name__ == "__main__":
    # 1️⃣ Load last 30 days AQI data
    historical_df = load_historical_aqi(RAW_AQI_FILE)
    print(historical_df.head(), "\n")

    # 2️⃣ Get current AQI
    current_data = get_current_openweather()
    print("Current AQI and pollutant data:", current_data, "\n")

    # 3️⃣ Fetch weather forecast
    df_weather = fetch_weather_forecast(LAT, LON, OPENWEATHER_API_KEY)
    print(f"Rows fetched from forecast: {len(df_weather)}")
    print(df_weather.head())
