import os
import certifi
from dotenv import load_dotenv

load_dotenv()

os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

from typing import TypedDict, Annotated
import operator
import uuid
import asyncio
import psycopg
from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq
# from tools.tavily_tool import tavily_search
# from tools.flight_tool import search_flights
from mcp_client import tavily_mcp_search, aviation_mcp_call, forecast_mcp_search, weather_mcp_search, extract_destination



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
# State
# =========================

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    weather_results: str
    itinerary: str
    llm_calls: int
    

# Flight Tool Router Prompt
FLIGHT_AGENT_PROMPT = """
You are a travel flight expert.

User Query:
{query}

Airport Information:
{airport_data}

Airline Information:
{airline_data}

Generate:

1. Likely departure airport
2. Likely arrival airport
3. Airlines serving this route
4. Typical flight duration
5. Estimated airfare range
6. Peak season pricing warning
7. Booking advice

Return concise travel guidance.
"""




# Flight Agent
async def flight_agent(state: TravelState):
    print("\nINSIDE FLIGHT AGENT\n")

    query = state["user_query"]

    try:

        airports = await aviation_mcp_call(
            "list_airports"
        )

        airlines = await aviation_mcp_call(
            "list_airlines"
        )


        print("\nAIRPORTS:", airports)
        print("\nAIRLINES:", airlines)

        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=str(airports)[:3000],
            airline_data=str(airlines)[:3000]
        )

        response = await llm.ainvoke([
            SystemMessage(
                content="You are an expert travel flight planner."
            ),
            HumanMessage(content=prompt)
        ])

        flight_data = response.content

    except Exception as e:

        flight_data = f"Flight information unavailable: {str(e)}"

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(
                content="Flight recommendations generated"
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

    

async def hotel_agent(state: TravelState) -> TravelState:
    query = f"Best hotels for {state['user_query']}"
    #hotel_data = tavily_search(query)
    hotel_data = await tavily_mcp_search(query)
    
    return {
        "hotel_results": hotel_data,
        "messages": [
            AIMessage(content="Hotel results fetched.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }
    

async def weather_agent(state: TravelState):

    city = extract_destination(state["user_query"])

    weather_data = await weather_mcp_search(city)

    forecast_data = await forecast_mcp_search(city)

    return {
        "weather_results": f"""
        Current Weather:
        {weather_data}

        Forecast:
        {forecast_data}
        """,
        "messages": [
            AIMessage(
                content="Weather information fetched"
            )
        ]
    }

    

async def itinerary_agent(state: TravelState) -> TravelState:
    prompt = f"""
Create a complete travel itinerary based on the following information:
User Query: {state['user_query']}
Flight Results: {state['flight_results']}
Hotel Results: {state['hotel_results']}
Weather Results: {state['weather_results']}

Make the itinerary practical, budget-aware, and easy to follow.
"""
    response = await llm.ainvoke([
        SystemMessage(content="You are a travel assistant that creates detailed itineraries based on user queries and search results."),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }
    
    
async def final_agent(state: TravelState) -> TravelState:
    final_prompt = f"""
Generate a final travel plan based on the following information:
User Query: {state['user_query']}
Flight Results: {state['flight_results']}
Hotel Results: {state['hotel_results']}
Weather Results: {state['weather_results']}
Itinerary: {state['itinerary']}

Format the final answer beautifully using these sections:

1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Weather Information
5. Day-by-Day Itinerary
6. Estimated Budget
7. Final Recommendations

Important:
- Be clear and practical.
- Mention that live flight API may not provide ticket prices if pricing is unavailable.
- Keep the response useful for real travel planning.
"""

    response = await llm.ainvoke([
        SystemMessage(content="You are a professional travel assistant that creates a final travel plan based on user queries and search results."),
        HumanMessage(content=final_prompt)
    ])
    
    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "weather_agent")
graph.add_edge("weather_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)


DATABASE_URL = get_database_url()


async def _init_checkpointer():
    """Create async Postgres connection and checkpointer.

    Called once at module level via asyncio.run() — this runs
    outside any request handler so it will not conflict with
    the FastAPI/Uvicorn event loop.
    """
    async_conn = await psycopg.AsyncConnection.connect(
        DATABASE_URL,
        autocommit=True,
        row_factory=dict_row,
    )
    checkpointer = AsyncPostgresSaver(async_conn)
    await checkpointer.setup()
    return checkpointer


import selectors

checkpointer = asyncio.run(
    _init_checkpointer(),
    loop_factory=lambda: asyncio.SelectorEventLoop(
        selectors.SelectSelector()
    ),
)

travel_graph = graph.compile(checkpointer=checkpointer)

# =========================
# Function for FastAPI
# =========================

async def run_travel_agent(user_input: str, thread_id: str | None = None):
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = await travel_graph.ainvoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "itinerary": "",
            "llm_calls": 0
        },
        config=config
    )

    final_answer = result["messages"][-1].content

    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "itinerary": result.get("itinerary", ""),
        "llm_calls": result.get("llm_calls", 0),
    }
