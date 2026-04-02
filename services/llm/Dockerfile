# LLM Connector — thin HTTP proxy to Anything-LLM
# No ML dependencies: this service only makes HTTP requests.
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    httpx \
    python-dotenv

COPY services/ ./services/

EXPOSE 8004

CMD ["uvicorn", "services.llm.app:app", "--host", "0.0.0.0", "--port", "8004"]
