import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")
LAT = float(os.getenv('LAT', '30.1575'))  # Multan
LON = float(os.getenv('LON', '71.5249'))
CSV_FILE = "data/raw_aqi_data.csv"

def fetch_latest_aqi_data():
    """Fetch the most recent AQI data from OpenWeather API"""
    try:
        url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        item = data['list'][0]
        c = item['components']
        
        record = {
            "timestamp": datetime.utcfromtimestamp(item['dt']),
            "pm2_5": c.get("pm2_5"),
            "pm10": c.get("pm10"),
            "no2": c.get("no2"),
            "o3": c.get("o3"),
            "so2": c.get("so2"),
            "co": c.get("co"),
            "aqi": item.get("main", {}).get("aqi")
        }
        
        return record
        
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None

def update_csv_with_new_data():
    """Append new data to CSV file, avoiding duplicates"""
    
    # Fetch latest data
    new_record = fetch_latest_aqi_data()
    if not new_record:
        print("⚠️ Failed to fetch new data")
        return False
    
    # Load existing CSV
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        print("⚠️ CSV file not found, creating new one")
        df = pd.DataFrame()
    
    new_timestamp = new_record['timestamp']
    
    # Check if this timestamp already exists (avoid duplicates)
    if len(df) > 0:
        # Check if we already have data from this hour
        df['hour'] = pd.to_datetime(df['timestamp']).dt.floor('H')
        new_hour = pd.to_datetime(new_timestamp).floor('H')
        
        if new_hour in df['hour'].values:
            print(f"ℹ️ Data for {new_hour} already exists, skipping...")
            return False
        
        df = df.drop('hour', axis=1)
    
    # Append new record
    new_df = pd.DataFrame([new_record])
    df = pd.concat([df, new_df], ignore_index=True)
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Keep only last 30 days of data (optional)
    cutoff_date = datetime.utcnow() - timedelta(days=30)
    df = df[pd.to_datetime(df['timestamp']) >= cutoff_date]
    
    # Save back to CSV
    df.to_csv(CSV_FILE, index=False)
    
    print(f"✅ Added new record: {new_timestamp}")
    print(f"📊 Total records in CSV: {len(df)}")
    
    return True

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "once":
        # Run once and exit
        print("📥 Fetching single update...")
        update_csv_with_new_data()
    else:
        print("📥 Running single update (use 'once' argument)")
        update_csv_with_new_data()
