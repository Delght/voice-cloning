# API Gateway — reverse proxy and pipeline orchestrator
# No ML dependencies: routes requests between microservices via HTTP.
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    httpx \
    python-multipart \
    python-dotenv

COPY gateway/ ./gateway/
COPY common/ ./common/
COPY services/__init__.py ./services/__init__.py

EXPOSE 8000

CMD ["uvicorn", "gateway.app:app", "--host", "0.0.0.0", "--port", "8000"]
