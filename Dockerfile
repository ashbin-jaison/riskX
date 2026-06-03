# ── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# System dependencies needed by geopandas / osmnx / shapely
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    gdal-bin \
    libgeos-dev \
    libproj-dev \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY api/requirements.txt ./api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

# Copy application code and data
COPY api/      ./api/
COPY data/     ./data/
COPY cache/    ./cache/

# Expose port
EXPOSE 8001

# Start the FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]
