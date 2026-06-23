# Crux backend — FastAPI + sentence-transformers + Groq.
# The embedding model is baked into the image so the container boots fast
# (no first-request download), which keeps the live demo snappy for clients.

FROM python:3.12-slim

# build tools some wheels need; cleaned up in the same layer to stay small
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install deps first so they cache across code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# pre-download the embedding model into the image (no runtime HF fetch)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY src ./src

# hosts (Render/Fly/HF/Railway) inject $PORT; default to 8000 locally
ENV PORT=8000
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT}"]
