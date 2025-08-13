import pandas as pd
import os
import numpy as np

# --------------------
# CONFIG
# --------------------
RAW_DATA_PATH = "data/raw_aqi_data.csv"  # Your input data
FEATURES_PATH = "data/processed/features.csv"  # Output file
DATE_COLUMN = "timestamp"  # Date column name


def ensure_data_directories():
    """Create data directories if they don't exist."""
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)


def load_raw_data():
    """Load and process raw data only when needed."""
    try:
        if not os.path.exists(RAW_DATA_PATH):
            print(f"Warning: {RAW_DATA_PATH} not found. Creating empty dataset.")
            # Create empty dataset with expected columns
            empty_df = pd.DataFrame(columns=[
                'timestamp', 'pm2_5', 'pm10', 'no2', 'o3', 'so2', 'co',
                'openweather_aqi', 'aqi'
            ])
            ensure_data_directories()
            empty_df.to_csv(RAW_DATA_PATH, index=False)
            return empty_df

        # Load existing data
        df = pd.read_csv(RAW_DATA_PATH)

        # Ensure timestamp is datetime
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors='coerce')

        # Sort by date
        df = df.sort_values(DATE_COLUMN)

        return df

    except Exception as e:
        print(f"Error loading raw data: {e}")
        return pd.DataFrame(columns=[
            'timestamp', 'pm2_5', 'pm10', 'no2', 'o3', 'so2', 'co',
            'openweather_aqi', 'aqi'
        ])


def process_features():
    """Process features from raw data."""
    try:
        df = load_raw_data()

        if df.empty:
            print("No data to process.")
            return df

        # --------------------
        # KEEP LAST 30 DAYS
        # --------------------
        last_date = df[DATE_COLUMN].max()
        cutoff_date = last_date - pd.Timedelta(days=30)
        df = df[df[DATE_COLUMN] > cutoff_date].copy()

        # --------------------
        # BASIC DATE FEATURES
        # --------------------
        df["dayofweek"] = df[DATE_COLUMN].dt.dayofweek
        df["day"] = df[DATE_COLUMN].dt.day
        df["month"] = df[DATE_COLUMN].dt.month

        # --------------------
        # LAG FEATURES
        # --------------------
        df["pm2_5_lag1"] = df["pm2_5"].shift(1)
        df["pm2_5_lag2"] = df["pm2_5"].shift(2)
        df["pm2_5_lag3"] = df["pm2_5"].shift(3)

        # --------------------
        # ROLLING MEAN FEATURES
        # --------------------
        df["pm2_5_roll7"] = df["pm2_5"].rolling(window=7).mean()
        df["pm2_5_roll14"] = df["pm2_5"].rolling(window=14).mean()

        # --------------------
        # FUTURE SHIFT TARGETS
        # --------------------
        df["pm2_5_next_1"] = df["pm2_5"].shift(-1)
        df["pm2_5_next_2"] = df["pm2_5"].shift(-2)
        df["pm2_5_next_3"] = df["pm2_5"].shift(-3)

        # --------------------
        # SAVE FEATURES
        # --------------------
        ensure_data_directories()
        df.to_csv(FEATURES_PATH, index=False)
        print(f"✅ Features saved to {FEATURES_PATH}")

        return df

    except Exception as e:
        print(f"Error processing features: {e}")
        return pd.DataFrame()


def prepare_feature_row(current_input, history_df):
    """Prepare feature row from current input and history."""
    try:
        import pandas as pd

        # Convert the incoming dictionary to a DataFrame
        df = pd.DataFrame([current_input])

        # Fill missing columns (in case some aren't provided)
        expected_cols = ["datetime", "pm2_5", "pm10", "no2", "o3", "so2", "co", "openweather_aqi"]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0

        # Handle missing values
        df = df.fillna(0)

        # Add computed AQI column
        def compute_aqi(pm25, pm10):
            try:
                # PM2.5 AQI calculation (simplified)
                if pm25 <= 12:
                    aqi_pm25 = (50 / 12) * pm25
                elif pm25 <= 35.4:
                    aqi_pm25 = ((100 - 51) / (35.4 - 12.1)) * (pm25 - 12.1) + 51
                else:
                    aqi_pm25 = 150  # Simplified

                # PM10 AQI calculation (simplified)
                if pm10 <= 54:
                    aqi_pm10 = (50 / 54) * pm10
                elif pm10 <= 154:
                    aqi_pm10 = ((100 - 51) / (154 - 55)) * (pm10 - 55) + 51
                else:
                    aqi_pm10 = 150  # Simplified

                return max(aqi_pm25, aqi_pm10)
            except:
                return 50  # Safe fallback

        df["aqi"] = df.apply(lambda row: compute_aqi(row["pm2_5"], row["pm10"]), axis=1)

        # Update history_df with the new row
        if history_df is not None and not history_df.empty:
            history_df = pd.concat([history_df, df], ignore_index=True)
        else:
            history_df = df.copy()

        return df, history_df

    except Exception as e:
        print(f"Error preparing feature row: {e}")
        # Return minimal safe values
        safe_df = pd.DataFrame([{
            'datetime': current_input.get('datetime', pd.Timestamp.now()),
            'pm2_5': current_input.get('pm2_5', 0),
            'pm10': current_input.get('pm10', 0),
            'aqi': 50
        }])
        return safe_df, safe_df


def load_history(path=FEATURES_PATH):
    """Load historical features from CSV."""
    try:
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Ensure timestamp column is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            return df
        else:
            print(f"History file {path} not found. Returning empty DataFrame.")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error loading history from {path}: {e}")
        return pd.DataFrame()


def save_history(df, path=FEATURES_PATH):
    """Save historical features to CSV."""
    try:
        ensure_data_directories()
        df.to_csv(path, index=False)
        print(f"History saved to {path}")
    except Exception as e:
        print(f"Error saving history to {path}: {e}")


def initialize_data_files():
    """Initialize data files if they don't exist."""
    try:
        ensure_data_directories()

        # Create empty raw data file if it doesn't exist
        if not os.path.exists(RAW_DATA_PATH):
            empty_raw = pd.DataFrame(columns=[
                'timestamp', 'pm2_5', 'pm10', 'no2', 'o3', 'so2', 'co',
                'openweather_aqi', 'aqi'
            ])
            empty_raw.to_csv(RAW_DATA_PATH, index=False)
            print(f"Created empty raw data file: {RAW_DATA_PATH}")

        # Create empty features file if it doesn't exist
        if not os.path.exists(FEATURES_PATH):
            empty_features = pd.DataFrame()
            empty_features.to_csv(FEATURES_PATH, index=False)
            print(f"Created empty features file: {FEATURES_PATH}")

    except Exception as e:
        print(f"Error initializing data files: {e}")


# Only run feature processing if this file is executed directly
if __name__ == "__main__":
    print("Processing features...")
    initialize_data_files()
    process_features()