"""Request limits apply to streamed bytes, not just Content-Length metadata."""

import httpx
import pytest
from fastapi import FastAPI, Request

from navigo_agent.middleware import MAX_REQUEST_BODY_BYTES, configure_request_size_limit


@pytest.fixture
def limited_app():
    app = FastAPI()
    app.state.calls = 0

    @app.post("/echo")
    async def echo(request: Request):
        app.state.calls += 1
        return {"body": (await request.body()).decode()}

    configure_request_size_limit(app)
    return app


@pytest.mark.asyncio
async def test_chunked_body_over_limit_never_reaches_handler(limited_app):
    async def chunks():
        yield b"a" * (MAX_REQUEST_BODY_BYTES // 2)
        yield b"b" * MAX_REQUEST_BODY_BYTES

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=limited_app), base_url="http://test") as client:
        response = await client.post("/echo", content=chunks())
    assert response.status_code == 413
    assert response.json()["success"] is False
    assert limited_app.state.calls == 0


@pytest.mark.asyncio
async def test_oversized_declared_body_never_reaches_handler(limited_app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=limited_app), base_url="http://test") as client:
        response = await client.post("/echo", content=b"a" * (MAX_REQUEST_BODY_BYTES + 1))
    assert response.status_code == 413
    assert limited_app.state.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("length", ["not-a-number", "-1", ""])
async def test_invalid_content_length_is_a_client_error(limited_app, length):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=limited_app), base_url="http://test") as client:
        response = await client.post("/echo", content=b"a", headers={"Content-Length": length})
    assert response.status_code == 400
    assert limited_app.state.calls == 0


@pytest.mark.asyncio
async def test_small_declared_length_cannot_bypass_actual_size_check(limited_app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=limited_app), base_url="http://test") as client:
        response = await client.post("/echo", content=b"a" * (MAX_REQUEST_BODY_BYTES + 1), headers={"Content-Length": "1"})
    assert response.status_code == 413
    assert limited_app.state.calls == 0


@pytest.mark.asyncio
async def test_chunked_body_at_limit_is_replayed_intact(limited_app):
    async def chunks():
        yield b"a" * (MAX_REQUEST_BODY_BYTES - 1)
        yield b"b"

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=limited_app), base_url="http://test") as client:
        response = await client.post("/echo", content=chunks())
    assert response.status_code == 200
    assert response.json()["body"] == "a" * (MAX_REQUEST_BODY_BYTES - 1) + "b"
    assert limited_app.state.calls == 1

