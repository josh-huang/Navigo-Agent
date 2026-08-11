# TripMate-AI

AI-powered travel companion built on LangGraph multi-agent architecture — intelligent itinerary planning, flight search, hotel discovery, and personalized trip management.

## Architecture

```
User → FastAPI (/api/travel) → LangGraph Workflow
                                    ├── flight_agent   (AviationStack API)
                                    ├── hotel_agent    (Tavily Search)
                                    ├── itinerary_agent (Groq LLM)
                                    └── final_agent    (Groq LLM — polished output)
```

Each agent is a node in a `StateGraph`. The workflow streams through flight search → hotel search → itinerary generation → final polish, with PostgreSQL-backed checkpoint persistence via `langgraph-checkpoint-postgres`.

## Tech Stack

| Layer          | Stack                                                    |
|----------------|----------------------------------------------------------|
| Language       | Python 3.12+                                             |
| Web Framework  | FastAPI + Jinja2 + Uvicorn                               |
| Agent Graph    | LangGraph + LangChain                                    |
| LLM            | Groq (Llama 3.3 70B)                                     |
| Search         | Tavily API (hotels) + AviationStack API (flights)        |
| State Persist  | PostgreSQL (Render) via `langgraph-checkpoint-postgres` |
| Geodata        | airportsdata + pycountry                                 |

## Project Structure

```
TripMate-AI/
├── app.py                  # FastAPI server, API routes, static mount
├── backend.py              # LangGraph state machine, agent nodes, DB connection
├── mcp_client.py           # MCP multi-server client (Tavily + AviationStack)
├── mcp_client_test.py      # MCP client test runner
├── pyproject.toml          # Project metadata & dependencies (uv)
├── uv.lock                 # Locked dependency versions
├── tools/
│   ├── __init__.py
│   ├── flight_tool.py      # AviationStack flight search + location resolution
│   └── tavily_tool.py      # Tavily web search wrapper
├── static/
│   ├── style.css           # Chat UI styles
│   └── script.js           # Chat UI logic, SSE streaming
├── templates/
│   └── index.html          # Chat interface
├── .env                    # API keys & DB URL (gitignored)
└── .venv/                  # Virtual environment (gitignored)
```

## Setup

```bash
# 1. Clone & install dependencies
git clone https://github.com/josh-huang/TripMate-AI.git
cd TripMate-AI
uv sync

# 2. Create .env
cp .env.example .env
# Fill in:
#   GROQ_API_KEY=
#   TAVILY_API_KEY=
#   AVIATIONSTACK_API_KEY=
#   DATABASE_URL=postgresql://...
#   DEFAULT_ORIGIN_IATA=DAC   (optional, defaults to DAC)
```

## Environment Variables

| Variable                | Required | Description                                      |
|-------------------------|----------|--------------------------------------------------|
| `GROQ_API_KEY`          | Yes      | Groq API key for LLM inference                   |
| `TAVILY_API_KEY`         | Yes      | Tavily API key for hotel/web search              |
| `AVIATIONSTACK_API_KEY`  | Yes      | AviationStack API key for flight data            |
| `DATABASE_URL`           | Yes      | PostgreSQL URL for LangGraph checkpoint storage  |
| `DEFAULT_ORIGIN_IATA`    | No       | Default departure airport (default: `DAC`)       |

## Running

```bash
# Development (auto-reload)
uv run python app.py
# or
uv run uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Open `http://127.0.0.1:8000` — the chat UI loads at `/`.

## API

| Endpoint        | Method | Description                              |
|-----------------|--------|------------------------------------------|
| `/`             | GET    | Chat interface (HTML)                    |
| `/api/travel`   | POST   | Send a travel request, get full plan     |
| `/health`       | GET    | Health check                             |

### `POST /api/travel`

```json
{
  "message": "Plan a 5-day trip to Tokyo from Dhaka",
  "thread_id": null
}
```

Response:

```json
{
  "success": true,
  "thread_id": "uuid-...",
  "answer": "### Trip Summary\n...",
  "flight_results": "[...]",
  "hotel_results": "[...]",
  "itinerary": "...",
  "llm_calls": 2
}
```

Pass the returned `thread_id` in follow-up requests to continue the same conversation session.

## License

MIT © 2026 Josh Huang
