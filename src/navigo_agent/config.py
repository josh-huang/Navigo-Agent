"""Centralized configuration for Navigo-Agent.

Loads environment variables and provides LLM factory + constants.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Required API Keys ─────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is missing. Add your Render PostgreSQL URL to .env.")

# ── Optional Settings ─────────────────────────────────────────────────
DEFAULT_ORIGIN_IATA = os.getenv("DEFAULT_ORIGIN_IATA", "DAC")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AVIATIONSTACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# ── Constants ─────────────────────────────────────────────────────────
MAX_SUPERVISOR_LOOPS = 3
MAX_INPUT_LENGTH = 1000
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 4096
LLM_TIMEOUT_SECONDS = 60
LLM_MAX_RETRIES = 2


# ── LLM Factory ───────────────────────────────────────────────────────
def get_llm(
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
    timeout: int = LLM_TIMEOUT_SECONDS,
    max_retries: int = LLM_MAX_RETRIES,
):
    """Return a configured ChatGroq instance.

    Parameters are configurable so sub-agents can tune for their task:
      - flight_agent: default temp (factual, no creativity needed)
      - itinerary_agent: higher temp (0.5) for creative planning
      - supervisor: low temp (0.1) for deterministic routing
    """
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=LLM_MODEL,
        api_key=GROQ_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
    )


def get_checkpointer_dsn() -> str:
    """Return the PostgreSQL connection string with sslmode=require enforced."""
    dsn = DATABASE_URL
    if "sslmode=" not in dsn:
        separator = "&" if "?" in dsn else "?"
        dsn = f"{dsn}{separator}sslmode=require"
    return dsn
