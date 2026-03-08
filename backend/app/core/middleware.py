import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from backend.app.core.metrics import HTTP_REQUEST_DURATION


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        # Skip /metrics itself to avoid recursion
        if request.url.path != "/metrics":
            path = request.url.path
            HTTP_REQUEST_DURATION.labels(
                method=request.method,
                endpoint=path,
                status=response.status_code,
            ).observe(duration)

        return response
