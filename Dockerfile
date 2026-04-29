# ──────────────────────────────────────────
# CardioSense AI — Production Docker Image
# ──────────────────────────────────────────
FROM python:3.12-slim

LABEL maintainer="Arwin Dominic, Dinesh Kumar, Chethan K Chavan, Amaresh"
LABEL project="CardioSense AI"
LABEL version="1.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/          ./src/
COPY api/          ./api/
COPY models/       ./models/
COPY data/         ./data/
COPY run_pipeline.py .

# MLflow uses SQLite so the DB persists inside the container
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db
ENV BUNDLE_PATH=/app/models/production_bundle.pkl
ENV PORT=5000
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Non-root user for security
RUN useradd -m -u 1000 cardiosense && chown -R cardiosense /app
USER cardiosense

# Expose API port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Start the FastAPI server with uvicorn
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "5000"]
