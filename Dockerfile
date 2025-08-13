FROM python:3.12-slim

WORKDIR /app

# Install requirements if you have them
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# Install watchdog for file change detection
RUN pip install watchdog

CMD ["python", "dev_autoreload.py"]
