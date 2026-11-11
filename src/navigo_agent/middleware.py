"""FastAPI middleware: security headers, rate limiting, request size limit."""

import os

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

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
    origins_str = os.getenv("ALLOWED_ORIGINS", "*")
    origins = [o.strip() for o in origins_str.split(",") if o.strip()] if origins_str != "*" else ["*"]

    # Credentials are only allowed for an explicit origin list: a wildcard
    # origin combined with credentials is rejected by browsers and unsafe.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=origins != ["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )


# ── Rate limiting ───────────────────────────────────────────────────────
#
# Only the expensive endpoints (the two /api/travel routes) are limited —
# app.py decorates them with `travel_rate_limit`. Health checks, the page and
# static assets are never limited: a load balancer probing /health must not
# be able to exhaust the budget and get the instance marked unhealthy.

RATE_LIMIT = os.getenv("RATE_LIMIT", "10/minute")


def _trusted_proxy_count() -> int:
    try:
        return max(0, int(os.getenv("TRUSTED_PROXY_COUNT", "0")))
    except ValueError:
        return 0


def client_ip(request: Request) -> str:
    """Rate-limit key: the real client address.

    Behind a reverse proxy the socket peer is the proxy, so every user would
    share one bucket. With TRUSTED_PROXY_COUNT=N the client is taken from
    X-Forwarded-For, counting N entries from the right — those are the ones
    appended by our own proxies. Entries further left are client-supplied and
    can be forged, so they are never used. With N=0 (default) the header is
    ignored entirely.
    """
    hops = _trusted_proxy_count()
    if hops:
        forwarded = [part.strip() for part in request.headers.get("x-forwarded-for", "").split(",")]
        forwarded = [part for part in forwarded if part]
        if len(forwarded) >= hops:
            return forwarded[-hops]
    return get_remote_address(request)


limiter = Limiter(key_func=client_ip)

# One shared bucket for both travel endpoints (JSON + SSE).
travel_rate_limit = limiter.shared_limit(lambda: RATE_LIMIT, scope="travel")


async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """429 in the API's own JSON shape, so the frontend can show the message."""
    return JSONResponse(
        status_code=429,
        content={
            "success": False,
            "error": f"Too many requests ({exc.detail}). Please wait a moment and try again.",
        },
        headers={"Retry-After": "60"},
    )


def configure_rate_limiting(app: FastAPI) -> None:
    """Register the limiter and its 429 handler."""
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)


def configure_request_size_limit(app: FastAPI) -> None:
    """Reject request bodies over MAX_REQUEST_BODY_BYTES."""

    @app.middleware("http")
    async def limit_request_size(request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_BODY_BYTES:
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
