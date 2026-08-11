# Navigo-Agent

AI-powered multi-agent travel planner built on LangGraph with Supervisor routing, ReAct agent loops, MCP tool integration, and production-grade guardrails.

## Architecture

```
User → FastAPI → Input Guard → LangGraph StateGraph
                                    ├── Intent Classifier (regex fast-path + LLM fallback)
                                    ├── Supervisor Agent (LLM JSON routing, max 3 loops)
                                    ├── ✈️ Flight Agent (ReAct over Tavily web search)
                                    ├── 🏨 Hotel Agent  (ReAct over Tavily web search)
                                    ├── 🌤️ Weather Agent (OpenWeather MCP)
                                    ├── 🗺️ Itinerary Agent (ReAct over Tavily + peer data)
                                    └── ✨ Final Synthesizer (polished output)
                                          ↓
                                    Output Guard → JSON Response / SSE Stream
```

**Key Design Decisions:**

- **ReAct Multi-Hop Search**: Each search agent runs up to 3 iterations of Reason → Search → Observe → Repeat. The LLM varies the Tavily search query (not the API endpoint) for multi-hop retrieval — broad search → gap detection → targeted search → final answer.
- **Supervisor without native tool calling**: Groq's `bind_tools` support is incomplete → supervisor uses prompt-engineered JSON output with Pydantic parsing and 3-level fallback.
- **Shared ReAct infrastructure**: `react_utils.py` provides a generic `run_react_loop()` used by flight, hotel, and itinerary agents — DRY, testable, and extensible.
- **Guardrails**: Input guard (length, injection detection, PII scanning, topic boundary) + Output guard (HTML strip, sensitive redaction, content safety).
- **SSE Streaming**: Real-time agent progress via `POST /api/travel/stream` — frontend renders progress indicators as each agent completes.

## Tech Stack

| Layer            | Stack                                                         |
|------------------|---------------------------------------------------------------|
| Language         | Python 3.13+                                                   |
| Web Framework    | FastAPI + Jinja2 + Uvicorn                                     |
| Agent Graph      | LangGraph + LangChain                                          |
| LLM              | Groq (Llama 3.3 70B Versatile)                                 |
| Search           | Tavily MCP (web search) + OpenWeather MCP (weather)            |
| MCP Protocol     | Multi-server: Tavily (HTTP), AviationStack (stdio), Weather (stdio) |
| State Persist    | PostgreSQL (Render) via `langgraph-checkpoint-postgres`        |
| Tracing          | LangSmith (optional, set env vars to enable)                   |
| Geodata          | airportsdata + pycountry                                       |
| Security         | Guardrails (input + output), CORS, rate limiting, security headers |

## Project Structure

```
Navigo-Agent/
├── app.py                         # FastAPI entry point, API routes
├── dockerfile                     # Multi-stage Docker build (Python 3.13-slim)
├── docker-compose.yml             # Local dev: app + PostgreSQL
├── pyproject.toml                 # Project metadata & dependencies (uv)
├── uv.lock                        # Locked dependency versions
├── .github/workflows/ci.yml       # CI/CD: lint → test → docker build
├── src/navigo_agent/              # Core package
│   ├── __init__.py                #   Public API: run_travel_agent()
│   ├── config.py                  #   LLM factory, env vars, LangSmith
│   ├── state.py                   #   TravelState TypedDict
│   ├── middleware.py              #   CORS, security headers, rate limiting
│   ├── streaming.py               #   SSE streaming wrapper for graph execution
│   ├── logging_config.py          #   Structured JSON logging
│   ├── guardrails/                #   Input/output safety
│   │   ├── input_guard.py         #     Length, injection, PII, topic boundary
│   │   └── output_guard.py        #     HTML strip, sensitive redaction, safety
│   ├── prompts/                   #   Prompt management (YAML-based)
│   │   ├── loader.py              #     YAML loader + variable interpolation
│   │   ├── flight.yaml            #     Flight ReAct system prompt
│   │   ├── hotel.yaml             #     Hotel ReAct system prompt
│   │   ├── itinerary.yaml         #     Itinerary ReAct system prompt
│   │   ├── supervisor.yaml        #     Supervisor routing prompt
│   │   ├── intent.yaml            #     Intent classifier prompt
│   │   └── final.yaml             #     Final synthesizer prompt
│   ├── mcp/                       #   MCP integration layer
│   │   ├── client.py              #     Multi-server MCP client
│   │   └── weather_server.py      #     Custom Weather MCP Server (OpenWeather)
│   ├── graph/                     #   LangGraph orchestration
│   │   ├── builder.py             #     StateGraph construction, checkpointing
│   │   ├── supervisor.py          #     Supervisor node (JSON routing decisions)
│   │   ├── routing.py             #     Intent classifier + conditional routing
│   │   └── agents/                #     Specialized sub-agents
│   │       ├── react_utils.py     #       Shared ReAct loop infrastructure
│   │       ├── flight.py          #       Flight agent (ReAct)
│   │       ├── hotel.py           #       Hotel agent (ReAct)
│   │       ├── weather.py         #       Weather agent
│   │       ├── itinerary.py       #       Itinerary agent (ReAct)
│   │       └── final.py           #       Final synthesizer
│   ├── evaluation/                #   LLM-as-Judge evaluation harness
│   │   ├── evaluator.py           #     Eval runner + LLM quality scoring
│   │   └── test_cases.py          #     11 curated test cases
│   └── memory/                    #   User preference persistence
│       └── user_profile.py        #     PostgreSQL-backed preference storage
├── tests/                         # Test suite (27 tests)
├── static/                        # Frontend assets
│   ├── style.css
│   └── script.js                  # SSE streaming + markdown rendering
└── templates/
    └── index.html                 # Chat interface
```

## Setup

### Local Development (with Docker)

```bash
# Clone & start
git clone https://github.com/josh-huang/Navigo-Agent.git
cd Navigo-Agent

# Create .env with your API keys
cp .env.example .env

# Start app + PostgreSQL
docker compose up --build
```

### Local Development (without Docker)

```bash
# 1. Clone & install
git clone https://github.com/josh-huang/Navigo-Agent.git
cd Navigo-Agent
uv sync

# 2. Create .env
cp .env.example .env
# Fill in:
#   GROQ_API_KEY=        (required)
#   TAVILY_API_KEY=      (required)
#   OPENWEATHER_API_KEY= (required)
#   DATABASE_URL=        (required — PostgreSQL)
#   LANGCHAIN_API_KEY=   (optional — LangSmith tracing)
#   LANGCHAIN_PROJECT=   (optional — defaults to navigo-agent)
```

## Environment Variables

| Variable                | Required | Description                                            |
|-------------------------|----------|--------------------------------------------------------|
| `GROQ_API_KEY`          | Yes      | Groq API key for LLM inference                         |
| `TAVILY_API_KEY`         | Yes      | Tavily API key for web search (flight + hotel + itinerary) |
| `OPENWEATHER_API_KEY`    | Yes      | OpenWeatherMap API key for weather data                |
| `DATABASE_URL`           | Yes      | PostgreSQL URL for LangGraph checkpoint storage        |
| `DEFAULT_ORIGIN_IATA`    | No       | Default departure airport (default: `DAC`)            |
| `LANGCHAIN_TRACING_V2`   | No       | Set to `true` to enable LangSmith tracing              |
| `LANGCHAIN_API_KEY`       | No       | LangSmith API key (required for tracing)               |
| `LANGCHAIN_PROJECT`       | No       | LangSmith project name (default: `navigo-agent`)       |
| `ALLOWED_ORIGINS`         | No       | CORS origins (default: `*`)                            |
| `RATE_LIMIT`              | No       | Rate limit string (default: `10/minute`)               |

## Running

```bash
# Development (auto-reload)
uv run python app.py

# Production
uv run uvicorn app:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000` — the chat UI loads at `/`.

## API

| Endpoint               | Method | Description                                    |
|------------------------|--------|------------------------------------------------|
| `/`                    | GET    | Chat interface (HTML)                          |
| `/api/travel`          | POST   | Submit a travel request, get full plan         |
| `/api/travel/stream`   | POST   | SSE streaming — real-time agent progress       |
| `/health`              | GET    | Health check                                   |

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
  "thread_id": "user_abc123...",
  "answer": "## Trip Summary\n...",
  "flight_results": "...",
  "hotel_results": "...",
  "weather_results": "...",
  "itinerary": "...",
  "llm_calls": 5
}
```

### `POST /api/travel/stream`

Same request body as `/api/travel`. Returns `text/event-stream`:

```
event: progress
data: {"agent":"weather_agent","display":"Checking weather","icon":"🌤️","status":"completed"}

event: agent_result
data: {"weather_results":"Current Weather:\n..."}

event: progress
data: {"agent":"flight_agent","display":"Searching flights","icon":"✈️","status":"completed"}

event: complete
data: {"thread_id":"user_abc123...","answer":"## Trip Summary\n...","llm_calls":5}
```

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ -v --cov=src/navigo_agent --cov-report=term-missing
```

## Evaluation

```bash
# Run the LLM-as-Judge evaluation suite
uv run python -m navigo_agent.evaluation.evaluator
```

The evaluator measures:
- **Tool accuracy**: did the supervisor dispatch the correct agents?
- **Quality score**: LLM-judged relevance, completeness, faithfulness (1-5)
- **Latency**: end-to-end response time
- **Error handling**: graceful degradation vs crash

## License

MIT © 2026 Josh Huang
