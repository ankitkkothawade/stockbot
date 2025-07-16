# syntax=docker/dockerfile:1

# --- Builder stage: compile wheels ---
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install build tools only in builder
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libblas-dev \
        liblapack-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /wheels
# Copy requirements and build all wheels
COPY requirements.txt .
RUN pip wheel --wheel-dir=/wheels --no-cache-dir -r requirements.txt

# --- Runtime stage: minimal image ---
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install only runtime dependencies (BLAS/LAPACK)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libblas-dev \
        liblapack-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Copy built wheels and install without re-building
COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt && \
    rm -rf /wheels

# Copy application code and data artifacts
COPY deploy.py models/ preprocessed_data/ .

# Do NOT hard-code secrets; use env-vars in Railway settings

# Default command
CMD ["python3", "deploy.py"]
