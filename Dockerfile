# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY app ./app

# Optional environment variables
ENV LAT=30.1575
ENV LON=71.5249

# Default command to run your updater
CMD ["python", "app/daily_updater.py", "once"]
