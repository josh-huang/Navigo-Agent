"""Output guard: sanitize LLM responses before returning to client.

Three layers:
  1. HTML tag stripping (via bleach)
  2. Credential / API key redaction (regex)
  3. Harmful content keyword screening
"""

import re
import logging
from bleach import clean

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Patterns matching common API key / token formats
CREDENTIAL_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("groq_api_key", re.compile(r"gsk_[a-zA-Z0-9]{30,60}")),
    ("openai_api_key", re.compile(r"sk-[a-zA-Z0-9]{32,60}")),
    ("github_token", re.compile(r"gh[pousr]_[a-zA-Z0-9]{36,}")),
    ("generic_token", re.compile(r"[a-zA-Z0-9_-]{32,}={0,2}")),  # base64-ish tokens; noisy but safe
    ("credit_card", re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b")),
]

# Harmful content keyword categories
HARMFUL_CONTENT_PATTERNS: dict[str, list[str]] = {
    "self_harm": ["kill yourself", "commit suicide", "end your life", "self-harm"],
    "illegal": ["how to make a bomb", "instructions for manufacturing", "illegal drug recipe"],
    "hate": ["ethnic cleansing", "racial superiority", "exterminate all"],
}

BLEACH_TAGS: list[str] = [
    "p", "br", "strong", "em", "b", "i", "u", "s", "a", "ul", "ol", "li",
    "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "code", "hr",
    "table", "thead", "tbody", "tr", "th", "td", "span", "div", "img",
    "sup", "sub", "del", "ins", "mark", "small",
]
BLEACH_ATTRIBUTES: dict[str, list[str]] = {
    "a": ["href", "title", "target", "rel"],
    "img": ["src", "alt", "title", "width", "height"],
    "td": ["align", "valign"],
    "th": ["align", "valign"],
    "span": ["class"],
    "div": ["class"],
    "code": ["class"],
    "pre": ["class"],
}


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def strip_html(text: str) -> str:
    """Remove disallowed HTML tags and attributes while preserving markdown-safe ones."""
    cleaned = clean(
        text,
        tags=BLEACH_TAGS,
        attributes=BLEACH_ATTRIBUTES,
        strip=True,
    )
    return cleaned


def redact_credentials(text: str) -> tuple[str, list[str]]:
    detected: list[str] = []
    for name, pattern in CREDENTIAL_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            detected.append(name)
            text = pattern.sub(f"[REDACTED_{name.upper()}]", text)
    if detected:
        logger.warning("Credential patterns redacted from output: %s", detected)
    return text, detected


def check_harmful_content(text: str) -> tuple[bool, str | None]:
    text_lower = text.lower()
    for category, keywords in HARMFUL_CONTENT_PATTERNS.items():
        for keyword in keywords:
            if keyword in text_lower:
                logger.warning("Harmful content detected in output: category=%s keyword=%s", category, keyword)
                return False, f"Output contained potentially harmful content ({category}). The response has been redacted."
    return True, None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def sanitize_output(llm_output: str) -> tuple[str, dict]:
    """Run the full output sanitization pipeline.

    Returns (sanitized_text, metadata).
    metadata includes:
      - html_stripped: whether HTML tags were cleaned
      - credentials_redacted: list of credential types detected
      - harmful_content: whether harmful content was found
      - original_length: original text length
      - sanitized_length: sanitized text length
    """
    metadata: dict = {
        "html_stripped": False,
        "credentials_redacted": [],
        "harmful_content": False,
        "original_length": len(llm_output),
    }

    text = strip_html(llm_output)
    if len(text) != len(llm_output):
        metadata["html_stripped"] = True

    text, creds_detected = redact_credentials(text)
    metadata["credentials_redacted"] = creds_detected

    safe, reason = check_harmful_content(text)
    if not safe:
        metadata["harmful_content"] = True
        text = f"[Content filtered: {reason}]"

    metadata["sanitized_length"] = len(text)
    return text, metadata
