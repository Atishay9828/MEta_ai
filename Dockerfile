# ── Strategic Negotiation Environment — OpenEnv Dockerfile ──
# Deploys the environment as a FastAPI server on HuggingFace Spaces

FROM python:3.11

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY env_wrapper.py .
COPY tasks.py .
COPY app.py .
COPY inference.py .
COPY openenv.yaml .
COPY pyproject.toml .
COPY README.md .
COPY server/ server/

# Environment variables (set at runtime via HF Spaces secrets/variables)
# API_BASE_URL — The API endpoint for the LLM
# MODEL_NAME   — The model identifier to use for inference
# HF_TOKEN     — Your HuggingFace API key

EXPOSE 7860

# Run the OpenEnv server (stays alive for agent connections)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
