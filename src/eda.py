import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def perform_eda(input_file):
    # Read CSV
    df = pd.read_csv(input_file)
    print("Columns in dataset:", df.columns.tolist())

    # Show basic stats
    print("\n📊 Summary statistics:")
    print(df.describe())

    # Convert timestamp to datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
        df = df.sort_values("timestamp")

    # --- Plot AQI over time ---
    if "aqi" in df.columns and "timestamp" in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(df["timestamp"], df["aqi"], marker='o', color='blue')
        plt.title("AQI Trend")
        plt.xlabel("Date")
        plt.ylabel("AQI")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("eda_aqi_trend.png")
        plt.show()
    else:
        print("\n⚠️ 'timestamp' or 'aqi' column missing — skipping trend plot.")

    # --- Histogram of AQI values ---
    if "aqi" in df.columns:
        plt.figure(figsize=(7, 5))
        plt.hist(df["aqi"], bins=20, color='green', edgecolor='black')
        plt.title("AQI Distribution")
        plt.xlabel("AQI")
        plt.ylabel("Frequency")
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig("eda_aqi_hist.png")
        plt.show()
    else:
        print("\n⚠️ 'aqi' column missing — skipping histogram.")


if __name__ == "__main__":
    default_file_path = os.path.join("data", "processed", "features.csv")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=default_file_path,
        help="Path to processed AQI features CSV"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"CSV file not found at {args.input}")

    perform_eda(args.input)
