"""FastAPI application for Navigo-Agent travel planner.

Routes:
  GET  /                  Chat interface (HTML)
  POST /api/travel        Submit a travel planning request
  POST /api/travel/stream SSE streaming travel planning request
  GET  /health            Health check
"""

from pathlib import Path
import logging
import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from navigo_agent import run_travel_agent, setup_middleware
from navigo_agent.guardrails import guard_input, guard_output
from navigo_agent.streaming import stream_travel_plan

BASE_DIR = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Navigo AI",
    description="LangGraph Multi-Agent Travel Planner with Supervisor, Guardrails, and MCP Integration",
    version="0.3.0",
)

# ── Middleware ──────────────────────────────────────────────────────────
setup_middleware(app)

# ── Static & Templates ──────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


class TravelRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    thread_id: str | None = Field(default=None, max_length=64)


# ── Routes ──────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the chat interface (HTML page)."""
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.post("/api/travel")
async def travel_planner(request: Request, request_data: TravelRequest):
    """Main endpoint: submit a travel planning request.

    Pipeline:
      1. Guard input (length, injection, PII, topic)
      2. Invoke LangGraph supervisor pipeline
      3. Guard output (HTML strip, sensitive redaction)
      4. Return structured JSON with per-agent results
    """
    try:
        user_message = request_data.message.strip()

        # Whitespace-only check
        if not user_message:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Message cannot be empty."},
            )

        # ── Input Guard ─────────────────────────────────────────────────
        guard_result = guard_input(user_message)
        if not guard_result.passed:
            logger.warning("Input guard blocked request: %s", guard_result.blocked_reason)
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "error": guard_result.blocked_reason or "Input validation failed.",
                },
            )

        result = await run_travel_agent(
            user_input=guard_result.sanitized_input,
            thread_id=request_data.thread_id,
        )

        # ── Output Guard ────────────────────────────────────────────────
        answer = guard_output(result.get("answer", ""))

        return JSONResponse(
            content={
                "success": True,
                "thread_id": result["thread_id"],
                "answer": answer,
                "flight_results": guard_output(result.get("flight_results", "")),
                "hotel_results": guard_output(result.get("hotel_results", "")),
                "weather_results": guard_output(result.get("weather_results", "")),
                "itinerary": guard_output(result.get("itinerary", "")),
                "llm_calls": result.get("llm_calls", 0),
            }
        )

    except Exception:
        logger.exception("Unhandled error processing travel request: %s", request_data.message[:100])
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "An internal error occurred. Please try again later.",
            },
        )


@app.post("/api/travel/stream")
async def travel_planner_stream(request: Request, request_data: TravelRequest):
    """Streaming endpoint: submit a travel request and receive SSE progress.

    Events:
      - progress: {agent, display, icon, status}
      - agent_result: partial result from a completed agent
      - complete: {thread_id, answer, ...}
      - error: {message, thread_id}

    Frontend renders real-time progress indicators as agents complete.
    """
    user_message = request_data.message.strip()

    if not user_message:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Message cannot be empty."},
        )

    # ── Input Guard ─────────────────────────────────────────────────────
    guard_result = guard_input(user_message)
    if not guard_result.passed:
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": guard_result.blocked_reason or "Input validation failed.",
            },
        )

    return StreamingResponse(
        stream_travel_plan(
            user_input=guard_result.sanitized_input,
            thread_id=request_data.thread_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
            "Connection": "keep-alive",
        },
    )


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring / load balancer probes."""
    return {"status": "ok", "message": "Navigo AI Travel Planner API is running"}


@app.get("/favicon.ico")
async def favicon():
    """Browser favicon — returns empty JSON to avoid 404 noise in logs."""
    return JSONResponse(content={})


# ── Entry Point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("app:app", "--host", "127.0.0.1", "--port", "8000", "--reload")
