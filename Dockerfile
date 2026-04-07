# ── Strategic Negotiation Environment — OpenEnv Dockerfile ──
# Deploys the environment as a FastAPI server on HuggingFace Spaces

FROM python:3.11-slim

WORKDIR /app

# Install dos2unix to handle Windows CRLF line endings
RUN apt-get update && \
    apt-get install -y --no-install-recommends dos2unix curl && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN dos2unix requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY env_wrapper.py .
COPY tasks.py .
COPY app.py .
COPY inference.py .
COPY openenv.yaml .
COPY pyproject.toml .
COPY README.md .
COPY server/ server/

# Fix CRLF on all Python source files (Windows → Linux)
RUN find . -name "*.py" -exec dos2unix {} + && \
    dos2unix openenv.yaml

# Environment variables (set at runtime via HF Spaces secrets/variables)
# API_BASE_URL — The API endpoint for the LLM
# MODEL_NAME   — The model identifier to use for inference
# HF_TOKEN     — Your HuggingFace API key

EXPOSE 7860

# Health check — ensures the server is responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run the OpenEnv server (stays alive for agent connections)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
