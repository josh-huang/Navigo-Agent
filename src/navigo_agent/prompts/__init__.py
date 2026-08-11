"""Prompt management: load, version, and inject prompts from YAML files.

Usage:
    from navigo_agent.prompts import load_prompt, render_prompt

    prompt = load_prompt("flight.react_system")
    rendered = render_prompt(prompt, destination="Tokyo", user_query="...")
"""

from navigo_agent.prompts.loader import (
    load_prompt,
    render_prompt,
    list_prompts,
    PROMPT_DIR,
)

__all__ = [
    "load_prompt",
    "render_prompt",
    "list_prompts",
    "PROMPT_DIR",
]
