"""MCP integration layer — convenience re-exports from client.py."""

from navigo_agent.mcp.client import (
    tavily_mcp_search,
    aviation_mcp_call,
    weather_mcp_search,
    forecast_mcp_search,
    extract_destination,
    extract_mcp_text,
    get_all_tools,
    get_aviation_tools,
)

__all__ = [
    "tavily_mcp_search",
    "aviation_mcp_call",
    "weather_mcp_search",
    "forecast_mcp_search",
    "extract_destination",
    "extract_mcp_text",
    "get_all_tools",
    "get_aviation_tools",
]
