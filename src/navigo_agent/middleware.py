"""FastAPI middleware: security headers, rate limiting, request size limit."""

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

MAX_REQUEST_BODY_BYTES = 10_241  # ~10 KB


def configure_security_headers(app: FastAPI) -> None:
    """Attach middleware that injects security headers on every response."""

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), interest-cohort=()"
        )
        return response


def configure_cors(app: FastAPI) -> None:
    """Add CORS middleware with configurable origins."""
    import os

    origins_str = os.getenv("ALLOWED_ORIGINS", "*")
    origins = [o.strip() for o in origins_str.split(",") if o.strip()] if origins_str != "*" else ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )


def configure_rate_limiting(app: FastAPI) -> None:
    """Add rate limiting via slowapi."""
    import os

    default_rate = os.getenv("RATE_LIMIT", "10/minute")
    limiter = Limiter(key_func=get_remote_address, default_limits=[default_rate])
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)


def configure_request_size_limit(app: FastAPI) -> None:
    """Reject request bodies over MAX_REQUEST_BODY_BYTES."""

    @app.middleware("http")
    async def limit_request_size(request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_BODY_BYTES:
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=413,
                content={"success": False, "error": "Request body too large. Maximum is 10 KB."},
            )
        return await call_next(request)


def setup_middleware(app: FastAPI) -> None:
    """One-shot: apply all middleware layers to the FastAPI app."""
    configure_cors(app)
    configure_rate_limiting(app)
    configure_security_headers(app)
    configure_request_size_limit(app)
