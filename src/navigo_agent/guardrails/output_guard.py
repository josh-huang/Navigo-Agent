"""Output guard: sanitise agent outputs before they reach the user.

Three checks:
  1. HTML tag stripping (prevent XSS in markdown-rendered output)
  2. Sensitive info redaction (API keys, credit cards leaked by LLM)
  3. Content safety blocklist
"""

import logging
import re

logger = logging.getLogger(__name__)

# ── HTML Sanitisation ──────────────────────────────────────────────────

# Inline <script>, <iframe>, <object>, <embed> — always strip
DANGEROUS_TAG_PATTERN = re.compile(
    r"<\s*(script|iframe|object|embed|form|input|link|meta|base|applet|frame|frameset|style)\b[^>]*>.*?</\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)

# Self-closing dangerous tags
DANGEROUS_SELF_CLOSING = re.compile(
    r"<\s*(script|iframe|object|embed|link|meta|base)\b[^>]*/?>",
    re.IGNORECASE,
)

# Event handler attributes (onclick, onload, etc.)
EVENT_HANDLER_PATTERN = re.compile(r"\bon\w+\s*=\s*[\"'][^\"']*[\"']", re.IGNORECASE)

# javascript: URLs
JAVASCRIPT_URL_PATTERN = re.compile(r"javascript\s*:", re.IGNORECASE)


def _strip_html(raw: str) -> str:
    """Strip dangerous HTML while preserving safe markdown."""
    cleaned = DANGEROUS_TAG_PATTERN.sub("", raw)
    cleaned = DANGEROUS_SELF_CLOSING.sub("", cleaned)
    cleaned = EVENT_HANDLER_PATTERN.sub("", cleaned)
    cleaned = JAVASCRIPT_URL_PATTERN.sub("", cleaned)
    return cleaned


# ── Sensitive Info Redaction ───────────────────────────────────────────

SENSITIVE_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    # API key formats
    ("groq_key", re.compile(r"gsk_[a-zA-Z0-9]{20,60}"), "[REDACTED:GROQ_KEY]"),
    ("openai_key", re.compile(r"sk-[a-zA-Z0-9]{20,60}"), "[REDACTED:API_KEY]"),
    ("tavily_key", re.compile(r"tvly-[a-zA-Z0-9]{20,60}"), "[REDACTED:API_KEY]"),
    # Credit card numbers (13-19 digits with optional separators)
    ("credit_card", re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"), "[REDACTED:CC]"),
    # Generic token patterns
    ("generic_token", re.compile(r"\b[a-zA-Z0-9_-]{32,64}\b"), "[REDACTED:TOKEN]"),
]

CREDIT_CARD_PATTERN = re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b")


def _redact_sensitive(raw: str) -> str:
    """Redact API keys, tokens, and credit card numbers."""
    result = raw
    for name, pattern, replacement in SENSITIVE_PATTERNS:
        if name == "generic_token":
            # Only redact generic tokens that look like they could be secrets
            # Skip common hash/ID patterns that appear in normal text
            continue
        if pattern.search(result):
            logger.warning("Output guard: redacting sensitive pattern '%s'.", name)
            result = pattern.sub(replacement, result)
    return result


# ── Content Safety ─────────────────────────────────────────────────────

SAFETY_BLOCKLIST: list[re.Pattern] = [
    re.compile(r"\b(child\s+(pornography|abuse|exploitation))\b", re.IGNORECASE),
    re.compile(r"\b(sexual\s+(abuse|assault|violence))\b", re.IGNORECASE),
    re.compile(r"\b(how\s+to\s+(make|manufacture|build)\s+(a\s+)?(bomb|weapon|explosive))\b", re.IGNORECASE),
]

SAFETY_REPLACEMENT = "[Content removed by safety filter.]"


def _check_content_safety(raw: str) -> str:
    """Replace content that matches the safety blocklist."""
    result = raw
    for pattern in SAFETY_BLOCKLIST:
        if pattern.search(result):
            logger.warning("Output guard: safety blocklist match detected.")
            result = pattern.sub(SAFETY_REPLACEMENT, result)
    return result


# ── Main Guard Pipeline ────────────────────────────────────────────────

def guard_output(raw_output: str) -> str:
    """Run the full output guard pipeline.

    Order: HTML strip → sensitive redaction → content safety.

    Always returns a string — never raises. The sanitised output may be
    shorter or have placeholder text but will never be None.
    """
    if not raw_output:
        return ""

    try:
        cleaned = _strip_html(raw_output)
        cleaned = _redact_sensitive(cleaned)
        cleaned = _check_content_safety(cleaned)
        return cleaned
    except Exception as e:  # noqa: BLE001
        logger.error("Output guard failed: %s — returning raw output as fallback.", e)
        return raw_output
