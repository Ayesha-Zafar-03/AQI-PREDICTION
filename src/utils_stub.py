# src/utils_stub.py
def pm25_to_aqi(pm):
    if pm is None:
        return None
    if pm < 0:
        return 0  # Or raise ValueError("PM2.5 cannot be negative")
    bp = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for c_low, c_high, i_low, i_high in bp:
        if c_low <= pm <= c_high:
            aqi = ((i_high - i_low) / (c_high - c_low)) * (pm - c_low) + i_low
            return int(aqi)  # Truncate for EPA compliance
    print(f"Warning: PM2.5 value {pm} exceeds 500.4 µg/m³, capping AQI at 500")
    return 500

def pm10_to_aqi(pm):
    if pm is None:
        return None
    bp = [
        (0.0, 54.0, 0, 50),
        (55.0, 154.0, 51, 100),
        (155.0, 254.0, 101, 150),
        (255.0, 354.0, 151, 200),
        (355.0, 424.0, 201, 300),
        (425.0, 504.0, 301, 400),
        (505.0, 604.0, 401, 500),
    ]
    for c_low, c_high, i_low, i_high in bp:
        if c_low <= pm <= c_high:
            aqi = ((i_high - i_low) / (c_high - c_low)) * (pm - c_low) + i_low
            return int(round(aqi))
    return 500

def overall_aqi(data):
    aqis = [
        pm25_to_aqi(data.get('pm2_5')),
        pm10_to_aqi(data.get('pm10')),
        # Add no2_to_aqi, o3_to_aqi, etc.
    ]
    aqis = [a for a in aqis if a is not None]
    return max(aqis) if aqis else None