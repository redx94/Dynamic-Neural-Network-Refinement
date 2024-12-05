# src/metrics.py

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import FastAPI, Response
import time

# Define metrics
REQUEST_COUNT = Counter('app_requests_total', 'Total number of requests')
REQUEST_LATENCY = Histogram('app_request_latency_seconds', 'Latency of requests in seconds')

def setup_metrics(app: FastAPI):
    @app.middleware("http")
    async def prometheus_middleware(request, call_next):
        REQUEST_COUNT.inc()
        start_time = time.time()
        response = await call_next(request)
        REQUEST_LATENCY.observe(time.time() - start_time)
        return response

    @app.get("/metrics")
    async def metrics():
        data = generate_latest()
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)
