"""Agent nodes for the Navigo travel planning graph."""

from navigo_agent.graph.agents.final import final_synthesizer
from navigo_agent.graph.agents.flight import flight_agent
from navigo_agent.graph.agents.hotel import hotel_agent
from navigo_agent.graph.agents.itinerary import itinerary_agent
from navigo_agent.graph.agents.weather import weather_agent

__all__ = [
    "final_synthesizer",
    "flight_agent",
    "hotel_agent",
    "itinerary_agent",
    "weather_agent",
]
