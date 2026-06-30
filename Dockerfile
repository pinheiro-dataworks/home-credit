FROM python:3.11-slim

WORKDIR /app

# System deps for LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY src/       ./src/
COPY api/       ./api/
COPY params.yaml .
COPY models/    ./models/

# Create dirs expected at runtime
RUN mkdir -p data/processed data/features mlruns

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
