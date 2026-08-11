"""MCP integration layer — re-exports from the root mcp_client module.

During Phase 2, the MCP client stays at the project root for backward compatibility
while the navigo_agent package imports from it.

Phase 3+: mcp_client.py can be moved into this directory.
"""

# Re-export all MCP convenience functions from the root module
from mcp_client import (
    tavily_mcp_search,
    aviation_mcp_call,
    weather_mcp_search,
    forecast_mcp_search,
    extract_destination,
    get_all_tools,
)

__all__ = [
    "tavily_mcp_search",
    "aviation_mcp_call",
    "weather_mcp_search",
    "forecast_mcp_search",
    "extract_destination",
    "get_all_tools",
]
