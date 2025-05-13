FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Force reinstall opencv-python-headless instead of opencv-python
RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-python && \
    pip install --no-cache-dir opencv-python-headless

# Copy application code
COPY . .

# Create directories
RUN mkdir -p static saved_images

# Expose port
EXPOSE 8080

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"] 