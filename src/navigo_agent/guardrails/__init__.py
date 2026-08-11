"""Guardrails: input sanitisation + output safety for Navigo-Agent.

Two-layer defence:
  - Input guard: length limits, prompt injection detection, PII scanning
  - Output guard: HTML sanitisation, sensitive info redaction, content safety
"""

from navigo_agent.guardrails.input_guard import GuardResult, guard_input
from navigo_agent.guardrails.output_guard import guard_output

__all__ = [
    "GuardResult",
    "guard_input",
    "guard_output",
]
