"""User preference memory: store and retrieve travel preferences across sessions.

Uses the existing PostgreSQL connection (via config.get_checkpointer_dsn())
to persist preferences. The checkpointer already maintains a connection pool,
so for simplicity we use a lightweight approach:
  - Preferences are extracted via LLM after each trip plan
  - Stored in a JSON-serializable format alongside thread_id
  - Injected into the system prompt on subsequent requests
"""

import json
import logging
import psycopg
from pydantic import BaseModel, Field

from navigo_agent.config import get_checkpointer_dsn, get_llm

logger = logging.getLogger(__name__)

# ── Schema ───────────────────────────────────────────────────────────

class UserPreferences(BaseModel):
    """Extracted user travel preferences."""
    home_airport: str | None = Field(default=None, description="User's home airport IATA code")
    budget_level: str | None = Field(default=None, description="budget | mid | luxury")
    preferred_airline: str | None = Field(default=None, description="Preferred airline name")
    travel_style: str | None = Field(default=None, description="adventure | relaxation | cultural | business")
    dietary_restrictions: str | None = Field(default=None)
    notes: str | None = Field(default=None, description="Any other notable preferences")


# ── Database Setup ───────────────────────────────────────────────────

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS user_preferences (
    thread_id TEXT PRIMARY KEY,
    preferences JSONB NOT NULL DEFAULT '{}',
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
"""


async def _ensure_table():
    """Create the user_preferences table if it doesn't exist."""
    dsn = get_checkpointer_dsn()
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
        await conn.execute(CREATE_TABLE_SQL)
    logger.debug("user_preferences table ready.")


# ── Preference Extraction (LLM) ──────────────────────────────────────

EXTRACTION_PROMPT = """Extract the user's travel preferences from this query and response.

Return a JSON object with:
- home_airport: the user's home airport code (IATA, e.g. "JFK", "DAC"), or null
- budget_level: "budget", "mid", or "luxury", or null
- preferred_airline: the airline name, or null
- travel_style: "adventure", "relaxation", "cultural", or "business", or null
- dietary_restrictions: any food restrictions mentioned, or null
- notes: any other notable preferences, or null

User Query: {query}
Agent Response (first 500 chars): {response_snippet}

Output ONLY the JSON object."""


async def extract_preferences(query: str, response: str) -> UserPreferences:
    """Use LLM to extract user preferences from query + response."""
    llm = get_llm(temperature=0.1)
    snippet = response[:500]

    try:
        result = await llm.ainvoke(EXTRACTION_PROMPT.format(query=query, response_snippet=snippet))
        text = result.content if hasattr(result, "content") else str(result)

        # Parse JSON from response
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]
        data = json.loads(text)
        return UserPreferences(**data)
    except Exception as e:
        logger.warning("Failed to extract preferences: %s", e)
        return UserPreferences()


# ── Storage ───────────────────────────────────────────────────────────

UPSERT_SQL = """
INSERT INTO user_preferences (thread_id, preferences, last_updated)
VALUES (%s, %s, NOW())
ON CONFLICT (thread_id)
DO UPDATE SET preferences = EXCLUDED.preferences, last_updated = NOW();
"""

SELECT_SQL = """
SELECT preferences FROM user_preferences WHERE thread_id = %s;
"""


async def save_preferences(thread_id: str, prefs: UserPreferences) -> None:
    """Persist user preferences to PostgreSQL."""
    await _ensure_table()
    dsn = get_checkpointer_dsn()
    try:
        async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
            await conn.execute(UPSERT_SQL, (thread_id, json.dumps(prefs.model_dump())))
        logger.info("Preferences saved for thread %s", thread_id)
    except Exception as e:
        logger.error("Failed to save preferences: %s", e)


async def load_preferences(thread_id: str) -> UserPreferences | None:
    """Load user preferences from PostgreSQL."""
    await _ensure_table()
    dsn = get_checkpointer_dsn()
    try:
        async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
            async with conn.cursor() as cur:
                await cur.execute(SELECT_SQL, (thread_id,))
                row = await cur.fetchone()
                if row:
                    data = row[0] if isinstance(row[0], dict) else json.loads(row[0])
                    return UserPreferences(**data)
    except Exception as e:
        logger.error("Failed to load preferences: %s", e)
    return None


# ── Prompt Injection ─────────────────────────────────────────────────

def inject_preferences_into_prompt(base_prompt: str, prefs: UserPreferences) -> str:
    """Inject user preferences into a system prompt."""
    if not any([prefs.home_airport, prefs.budget_level, prefs.preferred_airline, prefs.travel_style]):
        return base_prompt

    lines = ["\n\nUser Preferences (from previous sessions):"]
    if prefs.home_airport:
        lines.append(f"- Home airport: {prefs.home_airport}")
    if prefs.budget_level:
        lines.append(f"- Budget level: {prefs.budget_level}")
    if prefs.preferred_airline:
        lines.append(f"- Preferred airline: {prefs.preferred_airline}")
    if prefs.travel_style:
        lines.append(f"- Travel style: {prefs.travel_style}")
    if prefs.dietary_restrictions:
        lines.append(f"- Dietary: {prefs.dietary_restrictions}")
    if prefs.notes:
        lines.append(f"- Notes: {prefs.notes}")

    return base_prompt + "\n".join(lines)
