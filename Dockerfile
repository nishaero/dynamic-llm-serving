FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies as root
RUN apt-get update && apt-get install -y \
    git curl unzip build-essential \
    python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Create non-root user and group
RUN useradd --create-home --home-dir /appuser --shell /bin/bash appuser

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code and set ownership
COPY app/ /app/app
RUN chown -R appuser:appuser /app

WORKDIR /app

# Switch to non-root user
USER appuser

# Expose FastAPI port
EXPOSE 8080

# Run FastAPI with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]