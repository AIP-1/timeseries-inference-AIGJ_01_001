FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY collector ./collector
COPY config ./config
ENV PYTHONUNBUFFERED=1 DATA_DIR=/data COLLECTOR_CONFIG=/app/config/collector.yaml STATUS_PORT=8090
HEALTHCHECK --interval=60s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8090/healthz', timeout=4)" || exit 1
CMD ["python", "-m", "collector.main"]
