"""MCP integration layer — convenience re-exports from client.py."""

from navigo_agent.mcp.client import (
    aviation_mcp_call,
    extract_destination,
    extract_mcp_text,
    forecast_mcp_search,
    get_all_tools,
    get_aviation_tools,
    tavily_mcp_search,
    weather_mcp_search,
)

__all__ = [
    "aviation_mcp_call",
    "extract_destination",
    "extract_mcp_text",
    "forecast_mcp_search",
    "get_all_tools",
    "get_aviation_tools",
    "tavily_mcp_search",
    "weather_mcp_search",
]
