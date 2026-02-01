import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Load historical data
df = pd.read_csv("data/raw_aqi_data.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Create features
df['pm2_5_lag1'] = df['pm2_5'].shift(1)
df['pm2_5_lag2'] = df['pm2_5'].shift(2)
df['pm2_5_lag3'] = df['pm2_5'].shift(3)
df['pm2_5_roll7'] = df['pm2_5'].rolling(7).mean()
df['pm2_5_roll14'] = df['pm2_5'].rolling(14).mean()
df['dayofweek'] = df['timestamp'].dt.weekday
df['day'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month

df = df.dropna()

X = df[['pm2_5_lag1','pm2_5_lag2','pm2_5_lag3',
        'pm2_5_roll7','pm2_5_roll14',
        'dayofweek','day','month']]
y = df['pm2_5']

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X, y)

# Save model
os.makedirs("model/aqi_model", exist_ok=True)
joblib.dump(model, "model/aqi_model/rf_model.pkl")

print("✅ rf_model.pkl saved in models/aqi_model/")
