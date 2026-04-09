FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create models directory (to be mounted)
RUN mkdir -p /app/models

COPY . .

EXPOSE 8005

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005"]
