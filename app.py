"""FastAPI application for Navigo-Agent travel planner.

Routes:
  GET  /                 Chat interface (HTML)
  POST /api/travel       Submit a travel planning request
  GET  /health           Health check
"""

from pathlib import Path
import traceback
import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from navigo_agent import run_travel_agent, validate_input, sanitize_output, setup_middleware

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Navigo AI",
    description="LangGraph Multi-Agent Travel Planner with Supervisor, Guardrails, and MCP Integration",
    version="0.2.0",
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
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.post("/api/travel")
async def travel_planner(request: Request, request_data: TravelRequest):
    try:
        user_message = request_data.message.strip()

        # ── Phase 1: Input Guardrails ──────────────────────────────────
        guard_result = validate_input(user_message)
        if not guard_result.passed:
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "error": guard_result.blocked_reason,
                    "guard_blocked": True,
                },
            )
        user_message = guard_result.sanitized_input or user_message

        if not user_message:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Message cannot be empty."},
            )

        result = await run_travel_agent(
            user_input=user_message,
            thread_id=request_data.thread_id,
        )

        # ── Phase 1: Output Guardrails ─────────────────────────────────
        answer_text, output_meta = sanitize_output(result["answer"])

        return JSONResponse(
            content={
                "success": True,
                "thread_id": result["thread_id"],
                "answer": answer_text,
                "flight_results": result.get("flight_results", ""),
                "hotel_results": result.get("hotel_results", ""),
                "weather_results": result.get("weather_results", ""),
                "itinerary": result.get("itinerary", ""),
                "llm_calls": result.get("llm_calls", 0),
                "output_guard": output_meta,
            }
        )

    except Exception:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "An internal error occurred. Please try again later.",
            },
        )


@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Navigo AI Travel Planner API is running"}


@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={})


# ── Entry Point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
