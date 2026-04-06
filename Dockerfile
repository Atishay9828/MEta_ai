# ── Negotiation Environment — OpenEnv Dockerfile ──
# Person 3: Complete this Dockerfile for HuggingFace Spaces deployment
# Requirements: Python 3.11+, pip dependencies, inference.py entrypoint
# Constraints: CPU only, vcpu=2, memory=8gb, runtime < 20min

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY env_wrapper.py .
COPY tasks.py .
COPY inference.py .
COPY openenv.yaml .
COPY README.md .

# Environment variables (set at runtime, NOT hardcoded)
# API_BASE_URL — The API endpoint for the LLM
# MODEL_NAME   — The model identifier to use for inference
# HF_TOKEN     — Your HuggingFace API key

# Entrypoint
CMD ["python", "inference.py"]
