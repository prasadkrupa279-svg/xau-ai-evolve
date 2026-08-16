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
    AGENTS_PER_TF=5 \
    MAX_BARS=0 \
    MAX_M1_BARS=0

EXPOSE 10000

# 1 worker => ONE swarm (90 agents) + dashboard in a single process
CMD gunicorn render_swarm_app:app --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 8 --timeout 180
