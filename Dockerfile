# Production image for RagIntroLab's web UI (Flask under gunicorn).
#
# The vector store (store.npz) is baked into the image — it's built offline by
# `make index` and committed to the repo, so the running container needs only a
# query-time embedder (the Ollama accessory) + the OpenAI API for generation.

FROM python:3.14-slim

WORKDIR /app

# Install dependencies first (better layer caching).
COPY requirements.txt requirements-prod.txt ./
RUN pip install --no-cache-dir -r requirements-prod.txt

# App code + the prebuilt index (store.npz) + static assets.
COPY . .

# Fail fast at build time if the index wasn't committed.
RUN test -f store.npz || (echo "ERROR: store.npz missing — run 'make index' and commit it." && exit 1)

EXPOSE 8000

# One worker (keeps the in-memory rate limiter and the loaded matrix in a single
# process), several threads to handle concurrent I/O-bound requests. The long
# timeout accommodates slower remote-model responses.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", "--timeout", "120", "server:app"]
