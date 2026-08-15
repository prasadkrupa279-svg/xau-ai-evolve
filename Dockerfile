FROM python:3.11-slim

WORKDIR /app

# slim image needs nothing extra for pandas/numpy wheels on py3.11
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data memory

ENV PYTHONUNBUFFERED=1 \
    PORT=10000 \
    DATA_DIR=data \
    MEMORY_PATH=memory/global_ai_memory.json

EXPOSE 10000

# 1 worker => exactly ONE evolution loop (avoid duplicate daemons writing memory)
CMD gunicorn dashboard:app --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 8 --timeout 120
