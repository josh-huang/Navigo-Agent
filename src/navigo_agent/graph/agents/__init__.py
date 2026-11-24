"""Sub-agents of the Navigo orchestrator.

Each sub-agent is a graph node the orchestrator delegates to through a tool
call of the same name (see delegation.py for the contract and the registry).
"""

from navigo_agent.graph.agents.delegation import SUBAGENT_NAMES, SUBAGENTS
from navigo_agent.graph.agents.flight import flight_agent
from navigo_agent.graph.agents.hotel import hotel_agent
from navigo_agent.graph.agents.itinerary import itinerary_agent
from navigo_agent.graph.agents.weather import weather_agent

# Node implementation of every sub-agent in the registry.
SUBAGENT_NODES = {
    "flight_agent": flight_agent,
    "hotel_agent": hotel_agent,
    "weather_agent": weather_agent,
    "itinerary_agent": itinerary_agent,
}
assert tuple(SUBAGENT_NODES) == SUBAGENT_NAMES, "every registered sub-agent needs a node"

__all__ = [
    "SUBAGENTS",
    "SUBAGENT_NAMES",
    "SUBAGENT_NODES",
    "flight_agent",
    "hotel_agent",
    "itinerary_agent",
    "weather_agent",
]
