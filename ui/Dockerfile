# Gradio UI — connects to Gateway API, no direct model access
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    gradio \
    httpx \
    python-dotenv

COPY ui/ ./ui/

EXPOSE 7860

# Gradio binds to 0.0.0.0 by default; server_name env var is set in app.py
ENV GRADIO_SERVER_NAME=0.0.0.0

CMD ["python", "-m", "ui.app"]
