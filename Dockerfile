FROM python:3.11-slim

WORKDIR /app

# System deps for LightGBM + fetching model artifacts at build time
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY src/       ./src/
COPY api/       ./api/
COPY params.yaml .
COPY models/    ./models/

# Trained model artifacts (model.pkl, calibrated_model.pkl, etc.) are large
# binaries and are intentionally not committed to git (see .gitignore). They
# are fetched here from the GitHub Release published alongside this version
# of the code, so the API serves the real trained model instead of falling
# back to demo/mock data. To ship a retrained model, upload the new
# models/artifacts/* files to a new release and bump MODEL_RELEASE_TAG.
ARG MODEL_RELEASE_TAG=model-v1
RUN BASE="https://github.com/pinheiro-dataworks/home-credit/releases/download/${MODEL_RELEASE_TAG}" && \
    for f in model.pkl calibrated_model.pkl enc_state.pkl threshold.json \
             feature_names.json precomputed_stats.json drift_report.json \
             statistical_report.json metrics.json; do \
        curl -fsSL -o "models/artifacts/$f" "$BASE/$f"; \
    done

# Create dirs expected at runtime
RUN mkdir -p data/processed data/features mlruns

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
