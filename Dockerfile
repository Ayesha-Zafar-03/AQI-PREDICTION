# Use a lightweight Python base image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy only requirements first (better caching)
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Expose a port if your app uses a web server (optional)
# EXPOSE 8501  # Uncomment if using Streamlit or Flask

# Set environment variable to prevent Python buffering issues
ENV PYTHONUNBUFFERED=1

# Default command to run your app
CMD ["python", "main.py"]  # Replace main.py with your main script
