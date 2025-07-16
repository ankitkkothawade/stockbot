FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (build-essential for any compiled packages)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
       build-essential \
       libblas-dev \
       liblapack-dev && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt ./

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . ./

# Do NOT hard-code secrets here; set these via Railway (or env) when deploying
# ENV APCA_API_KEY_ID=<your_key>
# ENV APCA_API_SECRET_KEY=<your_secret>
# ENV APCA_BASE_URL=https://paper-api.alpaca.markets

# Default command to run your deployment script
CMD ["python3", "deploy.py"]
