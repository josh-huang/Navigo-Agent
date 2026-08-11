"""Navigo-Agent graph: supervisor-orchestrated multi-agent travel planning."""

__all__ = ["build_travel_graph", "get_compiled_graph"]


def build_travel_graph():
    from navigo_agent.graph.builder import build_travel_graph as _build_travel_graph

    return _build_travel_graph()


def get_compiled_graph():
    from navigo_agent.graph.builder import get_compiled_graph as _get_compiled_graph

    return _get_compiled_graph()
