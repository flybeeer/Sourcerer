# The FastAPI app image. Kept minimal for the scaffold.
FROM python:3.11-slim

WORKDIR /app

# Install the package (deps from pyproject.toml)
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

COPY scripts/ ./scripts/

EXPOSE 8000

# TODO (Phase 1): point this at the real app factory in src/sourcerer/api/main.py
CMD ["uvicorn", "sourcerer.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
