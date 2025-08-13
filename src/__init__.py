# src/__init__.py
"""
AQI Predictor - src package

This package contains:
- data acquisition functions
- feature engineering pipeline
- model training and evaluation
- inference utilities
- helper functions

By structuring the project as a package, you can import functions
from anywhere inside the project like:

    from src import get_current_openweather, pm25_to_aqi
"""

# Expose commonly used functions for direct import from src
from .data_acquisition import get_current_openweather
from .utils_stub import pm25_to_aqi
from .inference import (
    load_model_local,
    naive_forecast,
    predict_from_model
)

__all__ = [
    "get_current_openweather",
    "pm25_to_aqi",
    "load_model_local",
    "naive_forecast",
    "predict_from_model"
]