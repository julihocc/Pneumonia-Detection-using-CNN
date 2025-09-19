# Base image with Python and ML libraries
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/

# Install the package
COPY setup.py .
RUN pip install -e .

# Create directories for models, logs, and outputs
RUN mkdir -p models logs outputs data

# Set default environment variables
ENV MODEL_PATH="/app/models/best_model.h5"
ENV CONFIG_PATH="/app/configs/default.yaml"

# Expose port for API
EXPOSE 8000

# Default command (can be overridden)
CMD ["python", "-m", "pneumonia_detector.api"]