# The FastAPI app image.
FROM python:3.11-slim

WORKDIR /app

# Install the package + the API-route extra (anthropic) so the hard-query route
# works out of the box. The local cross-encoder reranker (.[rerank], pulls torch)
# is intentionally left out to keep the image small — add it if you set
# RERANKER_TYPE=local.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir ".[api]"

COPY scripts/ ./scripts/
COPY eval/ ./eval/

EXPOSE 8000

# The app factory ensures the DB schema exists on startup (lifespan hook).
CMD ["uvicorn", "sourcerer.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
