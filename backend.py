import os
import certifi
from dotenv import load_dotenv

load_dotenv()

os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

from typing import TypedDict, Annotated
import operator
import uuid

import psycopg
from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from tools.tavily_tool import tavily_search as _tavily_search
from tools.flight_tool import search_flights as _search_flights

# =========================
# Config
# =========================

def get_database_url():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. Please add your Render PostgreSQL External Database URL to .env"
        )
    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"
    return database_url


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please add your Groq API key to .env")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY
)

# =========================
# Tools (LangChain native)
# =========================

@tool
def search_flights(query: str) -> str:
    """
    Search for flights between two locations.
    Use this when the user asks about flights, air tickets, or getting from one city/country to another.
    Input: a natural-language query describing departure and destination, e.g. "flights from Dhaka to Tokyo".
    Returns: flight schedule data or an error message.
    """
    return _search_flights(query)


@tool
def search_hotels(query: str) -> str:
    """
    Search for hotels and accommodations.
    Use this when the user asks about hotels, places to stay, accommodation recommendations, or lodging options.
    Input: a natural-language search query, e.g. "best budget hotels in Tokyo".
    Returns: a list of hotel suggestions with descriptions and URLs.
    """
    return _tavily_search(f"Best hotels: {query}")


@tool
def search_web(query: str) -> str:
    """
    Search the web for general travel information: attractions, visa requirements, weather, local tips, etc.
    Use this for questions that are NOT about flights or hotels — things like tourist spots, food, transport, safety.
    Input: a natural-language search query.
    Returns: search result snippets with titles and URLs.
    """
    return _tavily_search(query)


TOOLS = [search_flights, search_hotels, search_web]

# =========================
# State
# =========================

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


# =========================
# ReAct Agent node
# =========================

SYSTEM_PROMPT = """You are TripMate, an intelligent AI travel assistant.

Your job is to help users plan travel. You have access to the following tools:
- search_flights: find flight information between locations
- search_hotels: find hotel and accommodation options
- search_web: search for general travel info (attractions, visa, weather, tips)

Follow the ReAct pattern:
1. Think about what the user needs
2. Call the appropriate tool(s) to gather information
3. After receiving tool results, decide if you need more data or if you can answer
4. When you have enough information, compose a complete, well-structured answer

Guidelines:
- If the user only asks about flights, don't search hotels. If only hotels, don't search flights.
- You can call multiple tools in sequence — e.g. flights → hotels → general info.
- Format your final answer with clear sections (Trip Summary, Flights, Hotels, Itinerary, Budget, Tips).
- If a tool returns an error, tell the user and suggest alternatives.
- Be practical, budget-aware, and give actionable recommendations.
- Reply in the same language the user used."""


def agent_node(state: TravelState) -> dict:
    """The ReAct agent: calls LLM with bound tools. LLM decides whether to use a tool or respond."""
    messages = state["messages"]

    # Prepend system prompt on first call
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

    llm_with_tools = llm.bind_tools(TOOLS)
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# =========================
# Graph construction
# =========================

graph = StateGraph(TravelState)

graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(TOOLS))

graph.add_edge(START, "agent")

# Conditional routing: if agent emits tool_calls → execute tools → loop back to agent
# Otherwise → END
graph.add_conditional_edges(
    "agent",
    tools_condition,
    {"tools": "tools", END: END}
)
graph.add_edge("tools", "agent")  # tool_node always returns to agent_node

# =========================
# Checkpointer
# =========================

DATABASE_URL = get_database_url()
_conn = psycopg.connect(DATABASE_URL, autocommit=True, row_factory=dict_row)
checkpointer = PostgresSaver(_conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)

# =========================
# API Interface (unchanged)
# =========================

def run_travel_agent(user_input: str, thread_id: str | None = None):
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = travel_graph.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )

    # Extract LLM call count from message types
    llm_calls = sum(
        1 for m in result["messages"]
        if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None)
    ) + sum(
        1 for m in result["messages"]
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
    )

    # The last AI message (non-tool-call) is the final answer
    final_answer = ""
    for m in reversed(result["messages"]):
        if isinstance(m, AIMessage) and m.content:
            final_answer = m.content
            break

    # Collect tool result content
    flight_results = ""
    hotel_results = ""
    for m in result["messages"]:
        if isinstance(m, ToolMessage):
            if m.name == "search_flights":
                flight_results += m.content + "\n"
            elif m.name == "search_hotels":
                hotel_results += m.content + "\n"

    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": flight_results.strip(),
        "hotel_results": hotel_results.strip(),
        "itinerary": "",
        "llm_calls": llm_calls,
    }
